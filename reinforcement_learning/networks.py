import tensorflow as tf
from tensorflow.contrib import layers
from pysc2.lib.actions import _FUNCTIONS
import numpy as np
from . import features_metadata

class AC_Network:
    def __init__(self, scope='global', dict_params=None):
        #Metadatas :
        self.minimap_depth = 7
        self.screen_depth = 17
        self.dict_features_idx = features_metadata.dict_features_idx # Feature positions for slicing
        self.dict_features_dim = features_metadata.dict_features_dim # Categorical dimensions scale
        self.dict_action_args = features_metadata.dict_action_args # Number of arguments for each part of the policy
        self.dict_features_types = features_metadata.dict_features_types # Datatype of each feature

        #General parameters :
        self.scope = scope
        self.resolution = dict_params['resolution']
        self.minimap_spectrum = dict_params['minimap_spectrum']
        self.screen_spectrum = dict_params['screen_spectrum']
        self.optimizer = dict_params['optimizer']
        self.beta = dict_params['beta']
        self.filter_actions = dict_params['filter_actions']
        self.actions_spectrum = dict_params['actions_spectrum']
        self.reduce_units_dim = dict_params['reduce_units_dim']
        self.convolution_activation = dict_params['convolution_activation']
        self.dense_activation = dict_params['dense_activation']
        self.policy_loss = dict_params['policy_loss']
        if self.reduce_units_dim :
            self.dict_features_dim['screen']['unit_type'] = dict_params['units_dim_reduction']

        with tf.variable_scope(self.scope):
            #Inputs definition
            with tf.name_scope("inputs"):
                with tf.name_scope("spatial_inputs"):
                    self.minimap_input = tf.placeholder(shape=[None, self.minimap_depth, self.resolution, self.resolution], dtype=tf.float32, name="minimap_input")
                    self.screen_input = tf.placeholder(shape=[None, self.screen_depth, self.resolution, self.resolution], dtype=tf.float32, name="screen_input")

                with tf.name_scope("non_spatial_inputs"):
                    self.available_actions_input = tf.placeholder(shape=[None, 541], dtype=tf.float32, name="available_actions_input")
                    self.actions_filter_input = tf.placeholder(shape=[None, len(self.actions_spectrum)], dtype=tf.float32, name="actions_filter_input")
                    self.non_spatial_state_input = tf.placeholder(shape=[None, self.compute_non_spatial_shape()], dtype=tf.float32, name="non_spatial_state")

            #Non spatial features preprocessing:
            with tf.name_scope("non_spatial"):
                self.non_spatial_representation = tf.identity(self.non_spatial_state_input, name="non_spatial_representation")

            #Minimap features preprocessing :
            with tf.name_scope("mimimap_features"):
                L_minimap_features = [self.preprocess_spatial_features(self.minimap_input, 'minimap', key) for key in self.minimap_spectrum]
                self.minimap_input_clean = tf.stack(L_minimap_features, -1)

            with tf.name_scope("screen_features"):
                L_screen_features = [self.preprocess_spatial_features(self.screen_input, 'screen', key) for key in self.screen_spectrum]
                self.screen_input_clean = tf.stack(L_screen_features, -1)

            #Features reshape :
            self.minimap_features = tf.reshape(self.minimap_input_clean, [-1, self.resolution, self.resolution, self.minimap_depth], name="minimap_reshaped")
            self.screen_features = tf.reshape(self.screen_input_clean, [-1, self.resolution, self.resolution, self.screen_depth], name="screen_reshaped")
            self.non_spatial_features = tf.identity(self.broadcast_non_spatial_features(self.non_spatial_representation), name="non_spatial_reshaped")

            #FullyConv building :
            ##The state representation is then formed by the concatenation of the
            ##screen and minimap network outputs, as well as the broadcast vector statistics, along the channel dimension
            self.minimap_conv_32_3_3 = self.compute_spatial_input_convolution(self.minimap_features, "minimap")
            self.screen_conv_32_3_3 = self.compute_spatial_input_convolution(self.screen_features, "screen")

            self.state_representation = tf.concat([self.minimap_conv_32_3_3, self.screen_conv_32_3_3, self.non_spatial_features], 3, name="state_representation")
            self.flattened_state_representation = layers.flatten(self.state_representation)

            #To compute the baseline and policies over categorical (non-spatial) actions, the state representation
            #is first passed through a fully-connected layer with 256 units and ReLU activations, followed by fully-connected linear layers :
            self.fully_connected = layers.fully_connected(self.flattened_state_representation, num_outputs=256, activation_fn=tf.nn.relu)

            with tf.name_scope("outputs"):
                with tf.name_scope("policy"):
                    if self.filter_actions :
                        self.policy_action, self.logits_action = self.compute_non_spatial_policy(self.fully_connected, len(self.actions_spectrum), "policy_action")
                    else:
                        self.policy_action, self.logits_action = self.compute_non_spatial_policy(self.fully_connected, len(_FUNCTIONS), "policy_action")
                    self.masked_policy_action = self.mask_on_available_actions(self.policy_action)
                    with tf.name_scope("policy_spatial"):
                        self.policy_arg_screen, self.logits_screen = self.compute_spatial_policy(self.state_representation, "screen")
                        self.policy_arg_screen2, self.logits_screen2  = self.compute_spatial_policy(self.state_representation, "screen2")
                        self.policy_arg_minimap, self.logits_minimap  = self.compute_spatial_policy(self.state_representation, "minimap")

                    with tf.name_scope("policy_non_spatial"):
                        self.policy_arg_build_queue_id, self.logits_build_queue_id = self.compute_non_spatial_policy(self.fully_connected, self.dict_action_args['build_queue_id'], "build_queue_id")
                        self.policy_arg_control_group_act, self.logits_control_group_act = self.compute_non_spatial_policy(self.fully_connected, self.dict_action_args['control_group_act'], "control_group_act")
                        self.policy_arg_control_group_id, self.logits_control_group_id = self.compute_non_spatial_policy(self.fully_connected, self.dict_action_args['control_group_id'], "control_group_id")
                        self.policy_arg_queued, self.logits_queued = self.compute_non_spatial_policy(self.fully_connected, self.dict_action_args['queued'], "queued")
                        self.policy_arg_select_add, self.logits_select_add = self.compute_non_spatial_policy(self.fully_connected, self.dict_action_args['select_add'], "select_add")
                        self.policy_arg_select_point_act, self.logits_select_point_act = self.compute_non_spatial_policy(self.fully_connected, self.dict_action_args['select_point_act'], "select_point_act")
                        self.policy_arg_select_unit_act, self.logits_select_unit_act = self.compute_non_spatial_policy(self.fully_connected, self.dict_action_args['select_unit_act'], "select_unit_act")
                        self.policy_arg_select_unit_id, self.logits_select_unit_id = self.compute_non_spatial_policy(self.fully_connected, self.dict_action_args['select_unit_id'], "select_unit_id")
                        self.policy_arg_select_worker, self.logits_select_worker = self.compute_non_spatial_policy(self.fully_connected, self.dict_action_args['select_worker'], "select_worker")
                        self.policy_arg_unload_id, self.logits_unload_id = self.compute_non_spatial_policy(self.fully_connected, self.dict_action_args['unload_id'], "unload_id")

                with tf.name_scope("value"):
                    self.value = tf.reshape(tf.layers.dense(inputs=self.fully_connected, activation=None, units=1), [-1], name="output_value")

            if self.scope != 'global':
                with tf.name_scope("%s_inputs"%self.scope):
                    self.agent_action = tf.placeholder(shape=[None], dtype=tf.int32, name="agent_action")

                    self.agent_arg_screen = tf.placeholder(shape=[None], dtype=tf.int32, name="agent_arg_screen")
                    self.agent_arg_screen2 = tf.placeholder(shape=[None], dtype=tf.int32, name="agent_arg_screen2")
                    self.agent_arg_minimap = tf.placeholder(shape=[None], dtype=tf.int32, name="agent_arg_minimap")
                    self.agent_arg_build_queue_id = tf.placeholder(shape=[None], dtype=tf.int32, name="agent_arg_build_queue_id")
                    self.agent_arg_control_group_act = tf.placeholder(shape=[None], dtype=tf.int32, name="agent_arg_control_group_act")
                    self.agent_arg_control_group_id = tf.placeholder(shape=[None], dtype=tf.int32, name="agent_arg_control_group_id")
                    self.agent_arg_queued = tf.placeholder(shape=[None], dtype=tf.int32, name="agent_arg_queued")
                    self.agent_arg_select_add = tf.placeholder(shape=[None], dtype=tf.int32, name="agent_arg_select_add")
                    self.agent_arg_select_point_act = tf.placeholder(shape=[None], dtype=tf.int32, name="agent_arg_select_point_act")
                    self.agent_arg_select_unit_act = tf.placeholder(shape=[None], dtype=tf.int32, name="agent_arg_select_unit_act")
                    self.agent_arg_select_unit_id = tf.placeholder(shape=[None], dtype=tf.int32, name="agent_arg_select_unit_id")
                    self.agent_arg_select_worker = tf.placeholder(shape=[None], dtype=tf.int32, name="agent_arg_select_worker")
                    self.agent_arg_unload_id = tf.placeholder(shape=[None], dtype=tf.int32, name="agent_arg_unload_id")

                    """self.agent_advantages = tf.placeholder(shape=[None], dtype=tf.float32, name="advantages")"""

                    self.agent_return = tf.placeholder(shape=[None], dtype=tf.float32, name="agent_return")
                    self.advantages = tf.stop_gradient(self.agent_return - self.value)

                with tf.name_scope("%s_loss_coeffs_inputs"%self.scope):
                    self.coeff_arg_screen = tf.placeholder(shape=[None], dtype=tf.float32, name="coeff_arg_screen")
                    self.coeff_arg_screen2 = tf.placeholder(shape=[None], dtype=tf.float32, name="coeff_arg_screen2")
                    self.coeff_arg_minimap = tf.placeholder(shape=[None], dtype=tf.float32, name="coeff_arg_minimap")
                    self.coeff_arg_build_queue_id = tf.placeholder(shape=[None], dtype=tf.float32, name="coeff_arg_build_queue_id")
                    self.coeff_arg_control_group_act = tf.placeholder(shape=[None], dtype=tf.float32, name="coeff_arg_control_group_act")
                    self.coeff_arg_control_group_id = tf.placeholder(shape=[None], dtype=tf.float32, name="coeff_arg_control_group_id")
                    self.coeff_arg_queued = tf.placeholder(shape=[None], dtype=tf.float32, name="coeff_arg_queued")
                    self.coeff_arg_select_add = tf.placeholder(shape=[None], dtype=tf.float32, name="coeff_arg_select_add")
                    self.coeff_arg_select_point_act = tf.placeholder(shape=[None], dtype=tf.float32, name="coeff_arg_select_point_act")
                    self.coeff_arg_select_unit_act = tf.placeholder(shape=[None], dtype=tf.float32, name="coeff_arg_select_unit_act")
                    self.coeff_arg_select_unit_id = tf.placeholder(shape=[None], dtype=tf.float32, name="coeff_arg_select_unit_id")
                    self.coeff_arg_select_worker = tf.placeholder(shape=[None], dtype=tf.float32, name="coeff_arg_select_worker")
                    self.coeff_arg_unload_id = tf.placeholder(shape=[None], dtype=tf.float32, name="coeff_arg_unload_id")

                self.dict_policies = {'action':self.masked_policy_action,'screen':self.policy_arg_screen, 'screen2':self.policy_arg_screen2,
                'minimap':self.policy_arg_minimap, 'build_queue_id':self.policy_arg_build_queue_id,
                'control_group_act':self.policy_arg_control_group_act, 'control_group_id':self.policy_arg_control_group_id,
                'queued':self.policy_arg_queued, 'select_add':self.policy_arg_select_add, 'select_point_act':self.policy_arg_select_point_act,
                'select_unit_act':self.policy_arg_select_unit_act, 'select_unit_id':self.policy_arg_select_unit_id,
                'select_worker':self.policy_arg_select_worker, 'unload_id':self.policy_arg_unload_id}

                self.dict_logits = {'action':self.logits_action,'screen':self.logits_screen, 'screen2':self.logits_screen2,
                'minimap':self.logits_minimap, 'build_queue_id':self.logits_build_queue_id,
                'control_group_act':self.logits_control_group_act, 'control_group_id':self.logits_control_group_id,
                'queued':self.logits_queued, 'select_add':self.logits_select_add, 'select_point_act':self.logits_select_point_act,
                'select_unit_act':self.logits_select_unit_act, 'select_unit_id':self.logits_select_unit_id,
                'select_worker':self.logits_select_worker, 'unload_id':self.logits_unload_id}

                self.dict_actions = {'action':self.agent_action,'screen':self.agent_arg_screen, 'screen2':self.agent_arg_screen2,
                'minimap':self.agent_arg_minimap, 'build_queue_id':self.agent_arg_build_queue_id,
                'control_group_act':self.agent_arg_control_group_act, 'control_group_id':self.agent_arg_control_group_id,
                'queued':self.agent_arg_queued, 'select_add':self.agent_arg_select_add, 'select_point_act':self.agent_arg_select_point_act,
                'select_unit_act':self.agent_arg_select_unit_act, 'select_unit_id':self.agent_arg_select_unit_id,
                'select_worker':self.agent_arg_select_worker, 'unload_id':self.agent_arg_unload_id}

                self.dict_policy_coeffs = {'screen':self.coeff_arg_screen, 'screen2':self.coeff_arg_screen2,
                'minimap':self.coeff_arg_minimap, 'build_queue_id':self.coeff_arg_build_queue_id,
                'control_group_act':self.coeff_arg_control_group_act, 'control_group_id':self.coeff_arg_control_group_id,
                'queued':self.coeff_arg_queued, 'select_add':self.coeff_arg_select_add, 'select_point_act':self.coeff_arg_select_point_act,
                'select_unit_act':self.coeff_arg_select_unit_act, 'select_unit_id':self.coeff_arg_select_unit_id,
                'select_worker':self.coeff_arg_select_worker, 'unload_id':self.coeff_arg_unload_id}

                self.global_policy_loss, self.value_loss, self.entropy, self.network_loss = self.compute_losses()

                #Get gradients from local network using local losses
                self.local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
                self.grads = self.optimizer.compute_gradients(self.network_loss, self.local_vars)
                #Apply local gradients to global network
                self.global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.cliped_grads = []
                for grad, _ in self.grads:
                    grad = tf.clip_by_norm(grad, 0.1)
                    self.cliped_grads.append(grad)
                self.grad_norms = tf.constant(0) #TO MODIFY
                self.var_norms = tf.constant(0) #TO MODIFY
                self.apply_grads = self.optimizer.apply_gradients(zip(self.cliped_grads, self.global_vars))

                #Dictionary which maps labels to the tensorflow variables which
                #will allow to train the network from the "workers.py" file :
                self.dict_variables_to_feed = {'minimap_features':self.minimap_input, 'screen_features':self.screen_input,
                'available_actions':self.available_actions_input, 'actions_filter':self.actions_filter_input,
                'non_spatial_state':self.non_spatial_state_input, 'agent_return':self.agent_return}

                for key in self.dict_actions.keys():
                    value = self.dict_actions[key]
                    if key == 'action':
                        prefix = 'agent_'
                    else:
                        prefix = 'agent_arg_'
                    key = prefix+key
                    self.dict_variables_to_feed[key] = value

                for key in self.dict_policy_coeffs.keys():
                    value = self.dict_policy_coeffs[key]
                    key = 'coeff_arg_%s'%key
                    self.dict_variables_to_feed[key] = value

    def compute_losses(self):
        """
        This function goal is to build the tensorflow's graph which allows to
        compute the neural network losses .
        """
        policy_loss = 0
        cross_entropy = 0
        entropy = 0
        for key in self.dict_policies.keys():
            policy = self.dict_policies[key]
            logits = self.dict_logits[key]
            if key =='action':
                if self.filter_actions:
                    one_hot_dim = len(self.actions_spectrum)
                else:
                    one_hot_dim = len(_FUNCTIONS)

                policy_coeff = 1
            elif key in ['screen','screen2','minimap']:
                one_hot_dim = self.resolution**2
            else:
                one_hot_dim = self.dict_action_args[key]
                policy_coeff = self.dict_policy_coeffs[key]

            agent_action = self.dict_actions[key]
            agent_action_one_hot = tf.stop_gradient(tf.cast(tf.one_hot(agent_action, one_hot_dim), tf.float32))
            if self.policy_loss=='softmax':
                log_policy = tf.log(tf.clip_by_value(policy, 1e-12, 1))
                cross_entropy+= -tf.reduce_sum(agent_action_one_hot*log_policy, axis=1)*policy_coeff
                entropy+= -tf.reduce_sum(policy*log_policy, axis=1)*policy_coeff
            elif self.policy_loss=='logits':
                cross_entropy+= tf.nn.softmax_cross_entropy_with_logits_v2(labels=agent_action_one_hot, logits=logits)*policy_coeff
                log_policy = tf.log(tf.clip_by_value(policy, 1e-12, 1))
                entropy+= -tf.reduce_sum(policy*log_policy, axis=1)*policy_coeff

        with tf.name_scope("network_outputs"):
            entropy = tf.identity(tf.reduce_mean(entropy, axis=0), name="entropy")
            policy_loss = tf.identity(tf.reduce_mean(cross_entropy*self.advantages, axis=0), name="policy_loss")
            value_loss =  tf.identity(tf.reduce_mean(tf.square(self.agent_return - self.value)), name="value_loss")
            #network_loss = tf.identity(policy_loss + value_loss + 0.001*entropy, name="network_loss")
            network_loss = tf.identity(policy_loss + value_loss + self.beta*entropy, name="network_loss")

        return policy_loss, value_loss, entropy, network_loss

    def preprocess_spatial_features(self, input, input_type, feature_label):
        """
        This function goal is to preprocess the spatial features which comes
        from the pysc2 observations.
        - Minimap sends a self.resolution*self.resolution*7 tensor
        - Screen sends a self.resolution*self.resolution*17 tensor
        - 'self.dict_features_idx' allows to slice a screen or a minimap tensor on a specific feature
        - 'self.dict_features_types' allows to contextualize the preprocessing of the specific feature (scalar or categorical preprocessing)
        - 'self.dict_features_dim' allows to expand categorical features to their respective dimensions (ie : their number of categories)
        - scalar features are transformed with a log
        - categorical features are one-hot expanded before to be passed in a 1x1 convolution product
        """
        slicing_index = self.dict_features_idx[input_type][feature_label]
        feature_type = self.dict_features_types[input_type][feature_label]
        one_hot_dimension = self.dict_features_dim[input_type][feature_label]
        name = input_type+"_"+feature_label
        input = tf.squeeze(tf.slice(input, [0,slicing_index,0,0],[-1,1,self.resolution,self.resolution]))

        if feature_type == 'scalar':
            with tf.name_scope("%s_scalar"%input_type):
                with tf.name_scope("%s_scalar_raw"%input_type):
                    input = tf.clip_by_value(input, 1e-12, 10000)
                with tf.name_scope("%s_scalar_clean"%input_type):
                    output = tf.log(input)
            return output

        elif feature_type == 'categorical':
            with tf.name_scope("%s_categorical"%input_type):
                with tf.name_scope("%s_categorical_raw"%input_type):
                    input = tf.cast(input,tf.int32, name="%s_raw_%s"%(input_type,feature_label))
                with tf.name_scope("%s_categorical_hot_encoded"%input_type):
                    input = tf.reshape(tf.one_hot(input, one_hot_dimension, axis=-1), [-1, self.resolution, self.resolution, one_hot_dimension], name="reshaped_%s"%name)
                with tf.name_scope("%s_categorical_conv_1_1"%input_type):
                    input = tf.layers.conv2d(inputs=input, strides = (1, 1), filters=1, kernel_size=1, padding="same", activation=self.convolution_activation, name="conv_1_1_%s"%name)
                    #input = tf.clip_by_value(input, 1e-12, 10000)
                with tf.name_scope("%s_categorical_clean"%input_type):
                    input_clean = tf.squeeze(input, name="%s_clean_%s"%(input_type, feature_label))
            return input_clean

    def mask_on_available_actions(self, policy):
        """
        This function goal is to use the "available_actions" which  comes from
        the pysc2 observations in order to multiply the policy corresponding indexes by 0.
        The policy is then re-normalized by it's sum.
        """
        if self.filter_actions:
            masked_policy = policy*self.actions_filter_input
        else:
            masked_policy = policy*self.available_actions_input
        masked_policy = masked_policy / tf.reduce_sum(masked_policy, axis=1, keepdims=True)
        return masked_policy

    def compute_spatial_input_convolution(self, spatial_input, name):
        """
        This function goal is to apply the convolutions products to the preprocessed and
        stacked minimap and screen features
        """
        ##Convolution with 16 5x5 filters :
        conv_16_5_5_layer = tf.layers.conv2d(
        inputs = spatial_input,
        strides = (1, 1),
        filters = 16,
        kernel_size = 5,
        padding = "same",
        activation = self.convolution_activation,
        name="conv_16_5_5_%s"%name)
        ## Screen Convolution with 32 3x3 filters :
        conv_32_3_3 = tf.layers.conv2d(
        inputs = conv_16_5_5_layer,
        strides = (1, 1),
        filters = 32,
        kernel_size = 3,
        padding = "same",
        activation = self.convolution_activation,
        name="conv_32_3_3_%s"%name)
        return conv_32_3_3

    def broadcast_non_spatial_features(self, non_spatial_tensor):
        """
        This function goal is to broadcast the "non_spatial_input" tensor which
        comes from pysc2 observations. An input tensor with shape (X,) will have it's
        values broadcased in order to build a tensor with a shape of (self.resolution, self.resolution, X).

        For computational performances, this tensor is followed by a 1x1 convolution product
        (Just comment this last part if you have enough computational power)
        """
        broadcasted = tf.tile(tf.expand_dims(tf.expand_dims(non_spatial_tensor, 1), 2),
        tf.stack([1, self.resolution, self.resolution, 1]))
        broadcasted = tf.layers.conv2d(inputs=broadcasted, strides = (1, 1), filters=1, kernel_size=1, padding="same", activation=self.convolution_activation, name="broadcasted_conv_1_1")
        return broadcasted

    def compute_spatial_policy(self, state_representation, name):
        """
        This function goal is to compute the spatial policies for 'screen', 'screen2'
        and 'minimap' args.
        It outputs 'logits' in order to compute losses and 'distribution'
        in order to allow the agent to pick actions.
        """
        #Finally, a policy over spatial actions is obtained using 1 x 1 convolution of the state representation with a single output channel
        conv_1_1_layer = tf.squeeze(tf.layers.conv2d(
        inputs = state_representation,
        strides = (1, 1),
        filters = 1,
        kernel_size = 1,
        padding = "same",
        activation = self.convolution_activation), name = "conv_1_1_%s"%name)
        reshaped = tf.reshape(conv_1_1_layer, [-1, self.resolution**2], name="flattened_%s"%name)
        distribution = tf.nn.softmax(reshaped, name = "softmax_%s"%name)
        logits = reshaped
        return distribution, logits

    def compute_non_spatial_policy(self, fully_connected, output_size, name):
        """
        This function goal is to compute the non-spatial policies (an action or
        it's distinct args).
        It outputs 'logits' in order to compute losses and 'distribution'
        in order to allow the agent to pick actions.
        """
        dense_layer = tf.squeeze(tf.layers.dense(
        inputs=fully_connected,
        units=output_size,
        activation=self.dense_activation),
        name="dense_%s"%name)
        reshaped = tf.reshape(dense_layer, [-1, output_size], name="flattened_%s"%name)
        distribution = tf.nn.softmax(reshaped, name = "softmax_%s"%name)
        logits = reshaped
        return distribution, logits

    def compute_non_spatial_shape(self):
        """
        This function goal is to compute the shape of the non_spatial input vector
        This values directly depends on the "workers.py" observations so look out
        if you modify them
        """
        last_actions_dim = 541
        action_result_dim = 215
        game_loop_dim = 1
        score_cumulative_dim = 13
        player_dim = 10
        control_groups_dim = 10*(self.dict_features_dim['screen']['unit_type']+1)
        available_actions_dim = 541
        single_select_dim = self.dict_features_dim['screen']['unit_type']
        multi_select_dim = self.dict_features_dim['screen']['unit_type']
        alert_dim = 2
        total_dim = sum([last_actions_dim, action_result_dim, game_loop_dim, score_cumulative_dim,
        player_dim, control_groups_dim, available_actions_dim, single_select_dim, multi_select_dim, alert_dim])

        return total_dim
