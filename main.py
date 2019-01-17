import pysc2
from pysc2.env import sc2_env
from pysc2.lib import features
from absl import app
import time
import tensorflow as tf
import numpy as np
import threading
import os
import pickle
import parameters_default
import parameters_custom
from reinforcement_learning.networks import AC_Network
from reinforcement_learning.workers import AC_Worker
from reinforcement_learning.tensorflow_functions import build_histo_summary
from exec_functions import clean_sc2_temp_folder, build_path, load_map_config
from context import tmp_maps_path


MAP_TO_TRAIN = "MoveToBeacon" #DefeatBanelings
custom_params = True
restore = True
MAP_TO_RESTORE = "MoveToBeacon"
restored_policy_type = 'a3c' #Choose between ['a3c', 'random']

params, training_path = load_map_config(MAP_TO_TRAIN, custom_params, restore, MAP_TO_RESTORE, restored_policy_type)

dict_workers_gpu = {1:0.25, 2:0.35, 3:0.45, 4:0.55, 5:0.65, 6:0.75}
session_config = tf.ConfigProto(device_count = {'GPU': 1}, #On CPU => 'GPU': 0||On GPU => 'GPU': 1
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=dict_workers_gpu[params['n_workers']]))
sess = tf.Session(config=session_config)

master_network = AC_Network(scope='global', dict_params=params['network_params'])
saver = tf.train.Saver(max_to_keep=100)
if restore:
    saver.restore(sess, training_path+"sessions\\model_episode_10.cptk")

def main(unused_argv):
    agents = []
    for process in range(params['n_workers']):
        agent = AC_Worker(id=process, session=sess, map_name=MAP_TO_TRAIN, restore=restore, dict_params=params['worker_params'], dict_network_params=params['network_params'])
        agent.episode = 0
        agents.append(agent)
    threads = []
    for thread_id in range(params['n_workers']):
        print("Starting worker_%s"%thread_id)
        t = threading.Thread(target=run_thread, args=(agents[thread_id], MAP_TO_TRAIN))
        threads.append(t)
        t.daemon = True
        t.start()
        time.sleep(2)

    for t in threads:
        t.join()

def run_thread(agent, map_name):
    while True:
        try:
            print("\nStarting episode %s for agent %s ..."%(agent.episode, agent.id))
            clean_sc2_temp_folder(tmp_maps_path, 8, 90)
            agent.rollouts_manager.empty_dict_rollouts()
            agent.episode_values = []
            agent.episode_cumulated_reward = 0
            agent.episode_step_count = 0
            agent.current_episode_actions = []
            agent.current_episode_rewards  = []
            agent.current_episode_values = []

            L_players = [sc2_env.Agent(sc2_env.Race.terran)]
            with sc2_env.SC2Env(map_name=map_name, players=L_players,
            agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=params['resolution'], minimap=params['resolution']),
            use_feature_units=True), step_mul=params['step_mul'],
            game_steps_per_episode=0, visualize=False, disable_fog=True
            ) as env:
                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                agent.reset()
                global start_time
                start_time = time.time()
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

            print("\nEpisode over for agent %s ..."% agent.id)

            #Summary parameters :
            available_actions_ratio = len(agent.current_episode_unique_actions)/len(agent.current_episode_available_actions)
            summary = tf.Summary()
            summary.value.add(tag='Perf/1_Reward', simple_value=float(agent.episode_cumulated_reward))
            summary.value.add(tag='Perf/2_Distinct actions', simple_value=float(len(agent.current_episode_unique_actions)))
            summary.value.add(tag='Perf/3_Average advantage', simple_value=float(np.mean(agent.advantages)))
            summary.value.add(tag='Perf/4_Previous actions ratio', simple_value=float(agent.previous_actions_ratio))
            summary.value.add(tag='Perf/5_Average value', simple_value=float(agent.average_value))
            summary.value.add(tag='Perf/6_Available actions ratio', simple_value=float(available_actions_ratio))
            summary.value.add(tag='Perf/7_Average agent return', simple_value=float(np.mean(agent.agent_return)))
            summary.value.add(tag='Perf/8_Random policy', simple_value=float(agent.random_policy))
            summary.value.add(tag='Perf/9_Episode length', simple_value=float(agent.current_episode_step_count))
            summary.value.add(tag='Losses/1_Value loss', simple_value=float(agent.value_loss))
            summary.value.add(tag='Losses/2_Policy loss', simple_value=float(agent.global_policy_loss))
            summary.value.add(tag='Losses/3_Entropy loss', simple_value=float(agent.entropy))
            summary.value.add(tag='Losses/4_Network loss', simple_value=float(agent.network_loss))
            #summary.value.add(tag='Losses/5_Grad norm', simple_value=float(agent.grad_norms))
            #summary.value.add(tag='Losses/6_Var norm', simple_value=float(agent.var_norms))

            for label in agent.dict_policy.keys():
                policy = agent.dict_policy[label][0]
                policy_len = len(policy)
                indexed_label = agent.index_label(label)+' | (%s)'%policy_len
                summary.value.add(tag=indexed_label, histo=build_histo_summary(policy, policy_len))
            agent.summary_writer.add_summary(summary, agent.episode)
            agent.summary_writer.flush()

            if agent.episode > 0 and agent.episode % 20 == 0 :
                session_path = training_path+"sessions\\model_episode_%s.cptk"%(str(agent.episode))
                build_path(session_path)
                saver.save(sess, session_path)
                print("\nModel saved")
            agent.episode+=1

        except KeyboardInterrupt:
            break

        except pysc2.lib.remote_controller.RequestError:
            print("\n\npysc2.lib.remote_controller.RequestError for worker %s\n\n"%agent.name)
            env.close()
            print("\n\nenvironment closed for worker %s\n\n"%agent.name)
            time.sleep(2)
            pass
        except pysc2.lib.remote_controller.ConnectError:
            print()
        except pysc2.lib.protocol.ConnectionError:
            print("\n\npysc2.lib.protocol.ConnectionError for worker %s\n\n"%agent.name)
            #Picked from "https://github.com/inoryy/reaver-pysc2/blob/master/reaver/envs/sc2.py#L57-L69"
            # hacky fix from websocket timeout issue...
            # this results in faulty reward signals, but I guess it beats completely crashing...
            env.close()

if __name__ == "__main__":
  app.run(main)
