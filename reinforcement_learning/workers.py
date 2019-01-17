from pysc2.lib import actions
from pysc2.agents import base_agent
import tensorflow as tf
import numpy as np
from .networks import AC_Network
from .preprocessing import get_env_featutures
from .env_interactions import RolloutsManager, select_army, compute_reward, pick_action_over_policy, build_dict_action_args, build_action_args
from .tensorflow_functions import initialize_uninitialized, update_target_graph, build_histo_summary

class AC_Worker(base_agent.BaseAgent):

    def __init__(self, id=1, session=None, map_name='MoveToBeacon', restore=False, dict_params=None, dict_network_params=None):
        super(AC_Worker, self).__init__()
        self.id = id
        self.sess = session
        self.name = "worker_" + str(id)
        self.map_name = map_name
        self.restore = restore

        self.gamma = dict_params['gamma']
        self.rollout_size = dict_params['rollout_size']
        self.with_random_policy = dict_params['with_random_policy']
        self.policy_type = 'random' if self.with_random_policy else 'a3c'
        self.random_policy = 1 if self.with_random_policy else 0

        self.episode = 1
        self.population_value = 0
        self.kill_score = 0
        self.episode_cumulated_reward = 0

        self.current_episode_step_count = 0
        self.previous_episode_actions = []
        self.current_episode_actions = []
        self.current_episode_rewards  = []
        self.current_episode_values = []
        self.current_episode_available_actions = []

        self.current_episode_unique_actions = {}
        self.previous_episode_unique_actions = {}
        self.previous_actions_kept = {}
        self.previous_actions_ratio = 0
        self.previous_cumulated_reward = 0
        self.batch_values = []

        #Rollouts management :
        self.rollouts_manager = RolloutsManager()

        #Networks params
        self.rl_network = AC_Network(scope=self.name, dict_params=dict_network_params)
        self.actions_spectrum = dict_network_params['actions_spectrum']
        self.filter_actions = dict_network_params['filter_actions']
        self.reduce_units_dim = dict_network_params['reduce_units_dim']
        if self.reduce_units_dim  == True:
            #49 is the highest ID between marines (48+1) and banelings (9+1)
            #In other cases the highest unit id is 1913 => computation become very big
            self.units_dimension = self.rl_network.dict_features_dim['screen']['unit_type']

        self.summary_writer = tf.summary.FileWriter("train\\%s\\%s\\summaries\\%s"%(self.map_name, self.policy_type,str(self.name)))
        self.batch_writer = tf.summary.FileWriter("train\\%s\\%s\\batches\\%s"%(self.map_name, self.policy_type, str(self.name)))
        self.batch_id = 0
        self.update_local_ops = update_target_graph('global',self.name)
        if not self.restore:
            self.sess.run(tf.global_variables_initializer())
        else:
            initialize_uninitialized(self.sess, self.name)


    def step(self, obs):
        super(AC_Worker, self).step(obs)
        self.sess.run(self.update_local_ops)
        self.population_value = obs.observation['player'][5]
        if obs.first():
            self.first_action = True
            return select_army(obs)
        else:
            if not self.first_action:
                self.reward = compute_reward(obs.observation["score_cumulative"][0], self.episode_cumulated_reward, self.first_action)
                self.episode_cumulated_reward+= self.reward
                self.current_episode_rewards.append(self.reward)
                self.rollouts_manager.fill_dict_rollouts('reward', {'agent_reward':self.reward})
            self.first_action = False

            if self.rollouts_manager.dict_rollouts['size'] == self.rollout_size or obs.last():
                if obs.last():
                    self.current_episode_unique_actions = set(self.current_episode_actions)
                    self.previous_actions_kept = self.current_episode_unique_actions.intersection(self.previous_episode_unique_actions)
                    self.previous_actions_ratio = len(self.previous_actions_kept)/len(self.current_episode_unique_actions)

                    self.previous_episode_unique_actions = self.current_episode_unique_actions
                    self.previous_episode_actions = self.current_episode_actions
                    train_state_value = np.float32(0)
                else:
                    train_state_value = self.value
                self.agent_return, self.advantages, self.value_loss, self.global_policy_loss, self.entropy, self.network_loss, self.grad_norms, self.var_norms =\
                self.train(train_state_value)
                self.rollouts_manager.empty_dict_rollouts()
                self.average_value = np.mean(self.batch_values)
                self.batch_values = []

                #Local network update :
                self.sess.run(self.update_local_ops)
                #Summary parameters :
                summary = tf.Summary()
                summary.value.add(tag='Perf/1_Reward', simple_value=float(self.episode_cumulated_reward))
                summary.value.add(tag='Perf/3_Average advantage', simple_value=float(np.mean(self.advantages)))
                summary.value.add(tag='Perf/5_Average value', simple_value=float(self.average_value))
                summary.value.add(tag='Perf/7_Average agent return', simple_value=float(np.mean(self.agent_return)))
                summary.value.add(tag='Perf/8_Random policy', simple_value=float(self.random_policy))
                summary.value.add(tag='Losses/1_Value loss', simple_value=float(self.value_loss))
                summary.value.add(tag='Losses/2_Policy loss', simple_value=float(self.global_policy_loss))
                summary.value.add(tag='Losses/3_Entropy loss', simple_value=float(self.entropy))
                summary.value.add(tag='Losses/4_Network loss', simple_value=float(self.network_loss))
                #summary.value.add(tag='Losses/5_Grad norm', simple_value=float(self.grad_norms))
                #summary.value.add(tag='Losses/6_Var norm', simple_value=float(self.var_norms))

                for label in self.dict_policy.keys():
                    policy = self.dict_policy[label][0]
                    policy_len = len(policy)
                    indexed_label = self.index_label(label)+'|(%s)'%policy_len
                    summary.value.add(tag=indexed_label, histo=build_histo_summary(policy, policy_len))
                self.batch_writer.add_summary(summary, self.batch_id)
                self.batch_writer.flush()
                self.batch_id+=1

            return self.act(obs)


    def act(self, obs):
        self.dict_env_inputs, raw_available_actions = get_env_featutures(obs, self.units_dimension, self.actions_spectrum)
        for action in raw_available_actions:
            if not action in self.current_episode_available_actions:
                self.current_episode_available_actions.append(action)

        self.rollouts_manager.fill_dict_rollouts('env_inputs', self.dict_env_inputs)

        feed_dict = {
        self.rl_network.minimap_input:self.dict_env_inputs['minimap_features'],
        self.rl_network.screen_input:self.dict_env_inputs['screen_features'],
        self.rl_network.available_actions_input:self.dict_env_inputs['available_actions'],
        self.rl_network.non_spatial_state_input:self.dict_env_inputs['non_spatial_state'],
        self.rl_network.actions_filter_input:self.dict_env_inputs['actions_filter']
        }

        [policy_action, screen, screen2, minimap, build_queue_id, control_group_act, control_group_id,\
        queued, select_add, select_point_act, select_unit_act, select_unit_id, select_worker, unload_id, value]\
        = self.sess.run([self.rl_network.masked_policy_action, self.rl_network.policy_arg_screen, self.rl_network.policy_arg_screen2,\
        self.rl_network.policy_arg_minimap, self.rl_network.policy_arg_build_queue_id, self.rl_network.policy_arg_control_group_act,\
        self.rl_network.policy_arg_control_group_id, self.rl_network.policy_arg_queued, self.rl_network.policy_arg_select_add,\
        self.rl_network.policy_arg_select_point_act, self.rl_network.policy_arg_select_unit_act, self.rl_network.policy_arg_select_unit_id,\
        self.rl_network.policy_arg_select_worker, self.rl_network.policy_arg_unload_id, self.rl_network.value],\
        feed_dict=feed_dict)

        self.L_policy = [policy_action, screen, screen2, minimap, build_queue_id, control_group_act, control_group_id,\
        queued, select_add, select_point_act, select_unit_act, select_unit_id, select_worker, unload_id]
        self.L_policy_labels = ['policy_action', 'screen', 'screen2', 'minimap', 'build_queue_id', 'control_group_act', 'control_group_id',\
        'queued', 'select_add', 'select_point_act', 'select_unit_act', 'select_unit_id', 'select_worker', 'unload_id']

        self.dict_policy = {label:policy for label, policy in zip(self.L_policy_labels, self.L_policy)}
        action_id = pick_action_over_policy(policy_action[0], self.with_random_policy)
        if self.filter_actions:
            action_id = self.actions_spectrum[action_id]

        self.value = value[0]
        self.batch_values.append(value)
        self.current_episode_actions.append(action_id)
        self.current_episode_values.append(value)
        self.current_episode_step_count = obs.observation['game_loop'][0]
        self.dict_action_args = build_dict_action_args(self.dict_policy, self.with_random_policy)
        L_args_for_action, self.dict_coeffs_args = build_action_args(action_id, self.dict_policy, self.dict_action_args, self.rl_network.resolution)

        self.rollouts_manager.fill_dict_rollouts('action_args', self.dict_action_args)
        self.rollouts_manager.fill_dict_rollouts('action', {'agent_action':action_id})
        self.rollouts_manager.fill_dict_rollouts('policy_coeffs', self.dict_coeffs_args)
        self.rollouts_manager.dict_rollouts['size']+=1

        return actions.FunctionCall(action_id, L_args_for_action)


    def train(self, train_state_value):
        feed_dict = {}
        agent_return = []
        state_value_discount = train_state_value.copy()

        for reward in reversed(self.rollouts_manager.dict_rollouts['agent_reward']):
            state_value_discount = reward + self.gamma * state_value_discount
            agent_return.append(state_value_discount)
        agent_return.reverse()
        agent_return = np.array(agent_return)

        for key in self.rl_network.dict_variables_to_feed.keys():
            tf_variable = self.rl_network.dict_variables_to_feed[key]
            if key == 'agent_return':
                value = agent_return
            else:
                value = np.array(self.rollouts_manager.dict_rollouts[key])
            feed_dict[tf_variable] = value

        #Global network update :
        network_values, advantages, value_loss, global_policy_loss, entropy, network_loss, grad_norms, var_norms, apply_grads = \
        self.sess.run([self.rl_network.value, self.rl_network.advantages, self.rl_network.value_loss, self.rl_network.global_policy_loss, self.rl_network.entropy,
        self.rl_network.network_loss, self.rl_network.grad_norms, self.rl_network.var_norms, self.rl_network.apply_grads],
        feed_dict = feed_dict)

        #return agent_return, agent_advantages, value_loss, global_policy_loss, entropy,  network_loss, grad_norms, var_norms
        return agent_return, advantages, value_loss, global_policy_loss, entropy,  network_loss, grad_norms, var_norms

    def index_label(self, label):
        for index, policy_label in enumerate(self.L_policy_labels):
            if label == policy_label:
                label = str(index+1)+'_'+policy_label
        return label
