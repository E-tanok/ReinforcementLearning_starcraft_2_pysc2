import numpy as np
from pysc2.lib.actions import FUNCTIONS
from pysc2.lib import actions
_NO_OP = actions.FUNCTIONS.no_op.id


def can_do(obs, action):
    return action in obs.observation.available_actions

def select_army(obs):
    if can_do(obs, actions.FUNCTIONS.select_army.id):
        return actions.FUNCTIONS.select_army("select")
    else:
        return actions.FunctionCall(_NO_OP, [])

def compute_reward(game_score, episode_cumulated_reward, is_first_action):
    if is_first_action:
        #The first action is to select the army
        reward = 0
    else:
        if episode_cumulated_reward != game_score:
            reward = game_score - episode_cumulated_reward
        else:
            reward = 0
    return reward

def compute_custom_reward(obs, kill_score, population_value):
    #TODO : testing it on a custom minigame
    killed_value_units  = obs.observation['score_cumulative'][5]
    killed_value_structures  = obs.observation['score_cumulative'][6]
    population_value = obs.observation['player'][5]

    kill_score = killed_value_units+killed_value_structures
    reward = 0
    if kill_score != kill_score:
        #Positive delta if enenmy units or buildings are killed
        kill_score_delta = kill_score - kill_score
        kill_score+= kill_score_delta
        reward+= kill_score_delta

    if population_value <= population_value:
        #Negative delta if the army decrease
        population_delta = population_value - population_value
        population_value+= population_delta
        #1 population unit less equals "-50 points"
        reward+= 15*population_delta

    return reward

def pick_action_over_policy(policy, with_random_policy):
    """
    This fuction goal is to pick an action from a policy.
    - 'with_random_policy' :
        --> If "True" : the choice of the argument within the policy will be random
        --> If "False" : the choice of the argument within the policy will be based on its probabilities
    """
    if not with_random_policy:
        #We choose a random probability within the policy, based on the policy probabilities :
        proba = np.random.choice(policy,p=policy)
        #Many arguments can have the same probabilities : they are all eligible for the agent selection :
        eligible_args = np.array(policy==proba)
    else:
        #If the argument is choosen randomply, eligible args are thoose with a probability>0 :
        eligible_args = np.array(policy>0)

    eligible_indexes = np.array([idx if check == True else -1 for idx, check in zip(range(len(eligible_args)),eligible_args)])
    eligible_indexes = eligible_indexes[eligible_indexes!=-1]
    random_arg = np.random.randint(len(eligible_indexes))
    action_arg = eligible_indexes[random_arg]

    return action_arg

def build_dict_action_args(dict_policy, with_random_policy):
    dict_action_args = {}
    #We first build a dictionnary which maps each "dict_policy" keys with their chosen args :
    for arg_name in dict_policy.keys():
        if not arg_name in ['policy_action', 'value']:
            arg = pick_action_over_policy(dict_policy[arg_name][0], with_random_policy)
            dict_action_args[arg_name] = arg
    return dict_action_args


def build_action_args(action_id, dict_policy, dict_action_args, spatial_resolution):
    #Now we take interest about the arguments linked to the chosen action :
    action_args = FUNCTIONS._func_list[action_id].args
    arg_list = []
    L_action_args = []

    L_coeffs = ['screen','screen2','minimap','build_queue_id','control_group_act',
    'control_group_id','queued','select_add','select_point_act','select_unit_act',
    'select_unit_id','select_worker','unload_id']
    dict_coeffs_args = {key:0 for key in L_coeffs}

    #If the action don't takes arguments :
    if len(action_args) == 0:
        return L_action_args, dict_coeffs_args
    else:
        #For each argument of the function :
        for arg_type in action_args:
            arg_name = arg_type.name
            dict_coeffs_args[arg_name] = 1
            if arg_name in ['screen', 'screen2', 'minimap']:
                arg = [dict_action_args[arg_name] % spatial_resolution, dict_action_args[arg_name] // spatial_resolution]
            else:
                arg = [dict_action_args[arg_name]]
            arg_list.append(arg_name)
            L_action_args.append(arg)
    return L_action_args, dict_coeffs_args


class RolloutsManager():
    """
    This class allows to manage the agent rollouts
    """
    def __init__(self):
        self.dict_rollouts = {'size':0, 'minimap_features':[], 'screen_features':[],'available_actions':[],
        'actions_filter':[], 'non_spatial_state':[], 'agent_action':[], 'agent_reward':[],
        'agent_return':[], 'agent_arg_screen':[], 'agent_arg_screen2':[],
        'agent_arg_minimap':[], 'agent_arg_build_queue_id':[],
        'agent_arg_control_group_act':[], 'agent_arg_control_group_id':[],
        'agent_arg_queued':[], 'agent_arg_select_add':[],
        'agent_arg_select_point_act':[], 'agent_arg_select_unit_act':[],
        'agent_arg_select_unit_id':[], 'agent_arg_select_worker':[],
        'agent_arg_unload_id':[], 'coeff_arg_screen':[], 'coeff_arg_screen2':[],
        'coeff_arg_minimap':[], 'coeff_arg_build_queue_id':[], 'coeff_arg_control_group_act':[],
        'coeff_arg_control_group_id':[], 'coeff_arg_queued':[], 'coeff_arg_select_add':[],
        'coeff_arg_select_point_act':[], 'coeff_arg_select_unit_act':[],
        'coeff_arg_select_unit_id':[], 'coeff_arg_select_worker':[], 'coeff_arg_unload_id':[]
        }

    def fill_dict_rollouts(self, arg, sub_dict):
        for key in sub_dict.keys():
            if arg == 'env_inputs':
                value = sub_dict[key][0]
            else:
                value = sub_dict[key]
            if arg == 'action_args':
                key = 'agent_arg_%s'%key
            elif arg == 'policy_coeffs':
                key = 'coeff_arg_%s'%key
            self.dict_rollouts[key].append(value)

    def empty_dict_rollouts(self):
        for key in self.dict_rollouts.keys():
            if key == 'size':
                self.dict_rollouts[key] = 0
            else:
                self.dict_rollouts[key] = []
