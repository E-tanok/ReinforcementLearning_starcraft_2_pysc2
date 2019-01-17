import numpy as np

def flatten(matrix):
    flat_matrix = matrix.flatten()
    if len(flat_matrix)==0:
        flat_matrix = np.array([0])
    return flat_matrix

def one_hot_and_reduce(categorical_array, one_hot_dim):
    try:
        one_hot_array = np.eye(one_hot_dim)[categorical_array]
        agregated_array = np.sum(one_hot_array, axis=0)
    except IndexError:
        #print("index error : the one hot array will be replaced by an array of 0")
        agregated_array = np.zeros(one_hot_dim)
    return agregated_array

def preprocess_last_actions(array):
    #Because 'last_actions' can have a variable length we need to add a
    #default value of '-1' when 'last_actions' length is '0' or '1' :
    if len(array) == 0:
        output = np.array([0, 0])
    elif len(array) == 1 :
        output = np.append(np.array([0]), array)
    else:
        output = array
    return output

def preprocess_non_spatial(matrix, layer):
    if len(matrix) == 0:
        if layer == 'action':
            output = np.array([0])
        elif layer == 'multi_select':
            output = np.array([[0,0,0,0,0,0,0]])
    else:
        output = matrix
    return output

def preprocess_game_loop(array):
    array[array==0]=1
    array = np.log(array)
    return array

def preprocess_alerts(array):
    if len(array) == 0 :
        output = np.array([0, 0])
    elif len(array) == 1:
        output = np.append(array, np.array([0]))
    elif len(array) == 2:
        output = np.array(array)
    else:
        output = np.array(array[0:2])
    return output

def preprocess_quantitative_arrays(array, arg):
    output_array = []
    for index, value in enumerate(array):
        if value > 0:
            if arg == 'score_cumulative':
                if index > 0:
                    value = np.log(value)
            elif arg == 'player':
                value = np.log(value)
        output_array.append(value)
    return output_array

def preprocess_control_groups(control_groups, hot_encoded_dim):
    units = control_groups[:,0]
    units_hot_encoded = np.eye(hot_encoded_dim)[units]
    units_number = np.array([control_groups[:,1]])
    hot_encoded_control_groups = np.concatenate((units_hot_encoded, units_number.T), axis=1)
    hot_encoded_control_groups = np.reshape(hot_encoded_control_groups, -1)
    return hot_encoded_control_groups

def get_env_featutures(obs, units_dimension, actions_spectrum):
    """
    This function goal is to build the enviroment features in order to feed the
    agent neural ntework
    """
    #Spatial features :
    feature_minimap = np.array([obs.observation['feature_minimap']])
    feature_screen = np.array([obs.observation['feature_screen']])
    #Non spatial features raw :
    non_spatial_state=[]
    last_actions = obs.observation['last_actions']
    action_result = obs.observation['action_result']
    game_loop = obs.observation['game_loop']
    score_cumulative = obs.observation['score_cumulative']
    player = obs.observation['player']
    control_groups = obs.observation['control_groups']
    available_actions = obs.observation['available_actions']
    raw_available_actions = available_actions.copy()
    single_select = obs.observation['single_select']
    multi_select = obs.observation['multi_select']
    alerts = obs.observation['alerts']
    #Non spatial features preprocessed:
    last_actions = one_hot_and_reduce(preprocess_last_actions(last_actions), 541) #one_hot dim 541 | '541' is based on pysc2/lib/actions.py
    action_result = one_hot_and_reduce(preprocess_non_spatial(action_result, 'action'), 215) #one_hot dim 215 |# '215' is based on https://github.com/Blizzard/s2client-proto/blob/master/s2clientprotocol/error.proto
    game_loop = preprocess_game_loop(game_loop)
    score_cumulative = preprocess_quantitative_arrays(score_cumulative, 'score_cumulative')
    player = preprocess_quantitative_arrays(player[1:], 'player') #We don't take the "player id which is at index 0"
    control_groups = preprocess_control_groups(control_groups, units_dimension)
    available_actions = one_hot_and_reduce(available_actions, 541)
    single_select = one_hot_and_reduce(single_select[:,0], units_dimension)
    multi_select = one_hot_and_reduce(preprocess_non_spatial(multi_select, 'multi_select')[:,0], units_dimension)
    alerts = preprocess_alerts(alerts)
    L_non_spatial_inputs = [last_actions, action_result, game_loop, score_cumulative, player, control_groups, available_actions, single_select, multi_select, alerts]
    for input in L_non_spatial_inputs:
        non_spatial_state+=list(input)
    non_spatial_state = np.array([non_spatial_state])

    actions_filter = np.array([[1 if action in raw_available_actions else 0 for action in actions_spectrum]])
    available_actions = np.array([available_actions])

    dict_env_inputs = {'minimap_features':feature_minimap, 'screen_features':feature_screen,
    'available_actions':available_actions, 'non_spatial_state':non_spatial_state, 'actions_filter':actions_filter}

    return dict_env_inputs, raw_available_actions
