from pysc2.lib import features
from pysc2.lib import actions

#Features latent dimensions (after the one hot encoding expansion) :
L_minimap_features = ['height_map','visibility_map','creep','camera','player_id','player_relative','selected']
L_screen_features = ['height_map','visibility_map','creep','power','player_id','player_relative','unit_type','selected',
                    'unit_hit_points','unit_hit_points_ratio','unit_energy','unit_energy_ratio','unit_shields',
                     'unit_shields_ratio','unit_density','unit_density_aa','effects']


dict_features_idx = {'minimap':{}, 'screen':{}}
dict_features_dim = {'minimap':{}, 'screen':{}}

for idx, val in enumerate(L_minimap_features):
    dict_features_idx['minimap'][val]  = idx
    dict_features_dim['minimap'][val]  = features.MINIMAP_FEATURES[idx].scale

for idx, val in enumerate(L_screen_features):
    dict_features_idx['screen'][val]  = idx
    dict_features_dim['screen'][val]  = features.SCREEN_FEATURES[idx].scale

#Actions arguments :
dict_action_args = {}
for arg_type in actions.TYPES:
    name = arg_type.name
    orig_size =size = arg_type.sizes
    size = orig_size[0]
    dict_action_args[name] = size

dict_features_types = {'minimap' : {'height_map':'scalar', 'visibility_map':'categorical',
'creep':'categorical', 'camera':'categorical', 'player_id':'categorical', 'player_relative':'categorical', 'selected':'categorical'},

'screen': {'height_map':'scalar', 'unit_hit_points':'scalar', 'unit_hit_points_ratio':'scalar',
'unit_energy':'scalar', 'unit_energy_ratio':'scalar', 'unit_shields':'scalar', 'unit_shields_ratio':'scalar',
'unit_density':'scalar', 'unit_density_aa':'scalar', 'visibility_map':'categorical', 'creep':'categorical',
'power':'categorical', 'player_id':'categorical', 'player_relative':'categorical', 'unit_type':'categorical',
'selected':'categorical', 'effects':'categorical'}
}
