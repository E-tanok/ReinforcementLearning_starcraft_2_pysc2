import pickle
import tensorflow as tf

#Environment parameters :
step_mul = 10
n_workers = 1

#Workers parameters :
gamma = 0.99
rollout_size = 40
with_random_policy = False
dict_worker_params = {'gamma':gamma, 'rollout_size':rollout_size, 'with_random_policy':with_random_policy}

#Networks parameters :
resolution = 32
minimap_spectrum = ['height_map','visibility_map','creep','camera','player_id','player_relative','selected']
screen_spectrum = ['height_map','visibility_map','creep','power','player_id','player_relative','unit_type','selected',
'unit_hit_points','unit_hit_points_ratio','unit_energy','unit_energy_ratio','unit_shields',
'unit_shields_ratio','unit_density','unit_density_aa','effects']
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, decay=0.99, epsilon=1e-5)
beta = 0.001

reduce_units_dim = True
units_dim_reduction = 49
filter_actions = True
dict_actions_spectrums = {'move':[331],
'nothing':[0],
'move_and_nothing':[0, 331],
'blizzard_minigames':[0, 1, 2, 3, 4, 6, 7, 12, 13, 140, 168, 261, 274, 331, 332, 333, 334, 451, 452, 453],
'blizzard_minigames_and_stimpack':[0, 1, 2, 3, 4, 6, 7, 12, 13, 140, 168, 234, 235, 236, 237, 238, 261, 274, 331, 332, 333, 334, 451, 452, 453],
'try_marines_hit_and_run':[0, 12, 234, 331],
'full':list(range(541))
}
actions_spectrum = dict_actions_spectrums['move']

dict_activation_functions = {'None':None, 'relu':tf.nn.relu}
dict_policy_losses = {'based_on_logits':'logits','based_on_softmax':'softmax'}
policy_loss = dict_policy_losses['based_on_logits']
conv_activation = dict_activation_functions['None']
dense_activation = dict_activation_functions['None']


dict_network_params = {'resolution':resolution, 'minimap_spectrum':minimap_spectrum, 'screen_spectrum':screen_spectrum,
'optimizer':optimizer, 'beta':beta,'filter_actions':filter_actions, 'actions_spectrum':actions_spectrum,
'reduce_units_dim':reduce_units_dim, 'units_dim_reduction':units_dim_reduction, 'convolution_activation':conv_activation, 'dense_activation':dense_activation ,'policy_loss':policy_loss}

dict_custom_params = {'resolution':resolution, 'n_workers':n_workers, 'step_mul':step_mul, 'worker_params':dict_worker_params, 'network_params':dict_network_params}
