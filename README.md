# The project :
This project allows to run reinforcement learning agents in pysc2 environments.
The features which provides from the API are preprocessed like Deepmind described it in [their paper](https://arxiv.org/abs/1708.04782)
Implemented agents (for now) :
- A3C

The pysc2 version used to train the agents is at the root of the project.

*The project is initially a project of end of studies but I really enjoyed reinforcement learning and I would like to develop it more*

## How to use it :
In order to start training :
- 1 : Go at 'main.py' :

      - 'MAP_TO_TRAIN' [str] : refers to the map where you want to train your agent
      - 'custom_params' [bool] :
            - 'True' : means that you want to use custom parameters which comes from 'parameters_custom.py'
            - 'False' : means that you want to use default parameters which comes from 'parameters_default.py'
      - 'restore' [bool] :
            - 'True' : means that you want to restore an older tensorflow session and its weights trained on the map 'MAP_TO_RESTORE'
            - 'False' : means that you want to create a new tensorflow session, with new weights (WARNING: the oldest weights will be remove if you choose this option, since their is not (yet?) a program to archive weights, you have to do it manually)
      - 'MAP_TO_RESTORE' [str] : The map where you trained weights you want to restore
      - 'restored_policy_type' [str] : refers to the type of policy you want to restore from your old session (right now, there are just 'a3c' and 'random')

- 2 : If you want to set custom parameters (or to modify the default parameters) :

      - Go at 'parameters_custom.py' (resp 'parameters_default.py'; both files have the same structure)
      - 'Environment parameters':
            - 'step_mul' [int] : The environment steps multiplier : the higher it is, the less the agent has time to react when the game sends inputs (more details at https://github.com/deepmind/pysc2/blob/master/docs/environment.md#game-speed)
            - 'n_workers' [int] : The number of workers you want to use

      - 'Workers parameters' :
            - 'gamma' [0>float<=1] : The rewards discount factor
            - 'rollout_size' [int] : The agent rollouts size (transitions {s, a, r})
            - 'with_random_policy' [bool] : "True" if you want to run a random agent else "False"
            - 'dict_worker_params' [dict] : A dictionnary which contains the parameters common to all the workers

      - 'Networks parameters' :
            - 'resolution' [int] : The screen AND the minimap resolutions
            - 'minimap_spectrum' [list of strings] : This list contains the minimap features labels YOU WANT TO USE in your neural networks. Just chose which to remove.
            - 'screen_spectrum' [list of strings] : (idem to 'minimap_spectrum', bet with the screen features labels)
            - 'optimizer' : The optimizer you want to use for the backpropagation
            - 'beta' : The entropy term in the A3C loss computation
            - 'reduce_units_dim' [bool] : "True" if you want to reduce the units dimension while using one-hot expansion, else "False"
                  - Explanations : We do a 1x1 convolution product after using one-hot expansion on the screen feature of categorical unit id's.
                  The maximum label of unit id's is 1912, while the units ID's of Banelings, Marines, Zerglings and Roaches are respectively 9, 49, 105 and 110
                  => Use a one-hot expansion on 1912 id's is inefficient on the blizzard minigames perimeter : If you use a custom map with other units, check your maximum id numeric value and reduce the units dimension by this number
            - 'units_dim_reduction' [int] : The maximum numeric value of your units id's if "reduce_units_dim"==True
            - 'filter_actions' [bool] : "True" if you want to filter the number of actions inside the neural network's policy, else "False"
            - 'dict_actions_spectrums' [dict] : This is an arbitrary dictionary which maps labels to lists of actions ids you ONLY want to use. It hels to filter the number of actions (for your agent and in the neural network). Feel free to modify it.
            - 'actions_spectrum' [list] : The actions id's list you want to take from "dict_actions_spectrums"
            - 'dict_activation_functions' [dict] : activations functions metadatas for the neural network
            - 'dict_policy_losses' [dict] : 'based_on_logits' if you want to base your loss on the policy logits outputs| 'based_on_softmax' if you want to base your loss on the softmax applied on the policy logits outputs
            - 'policy_loss' [str] : directly depends on "dict_policy_losses"
            - 'conv_activation' [tf_activation] : the activation function you want to use in the neural network convolution products
            - 'dense_activation' [tf_activation] : the activation function you want to use in the neural network dense layers
            - 'dict_network_params' [dict] : A dictionnary which contains the parameters common to all the networks

            - 'dict_custom_params' (resp 'dict_default_params') [dict] : A dictionnary which contains the parameters for the environment, the parameters common to all the networks and the parameters common to all the workers


- 3 : Once again, in 'main.py' :

      - 'dict_workers_gpu' [dict] : this is just a dictionnary which maps number of workers to the tensorflow "per_process_gpu_memory_fraction" GPUOptions . Feel free to modify it
      - 'master_network' [Neural Network] : This is the master network which will be updated by the workers .
      - The 'main' function associate workers to threads ; each thread runs the 'run_thread' function
      - All results are stored in "train/MAP_TO_TRAIN/agent.policy_type" where agent.policy_type = 'random' if "with_random_policy" else 'a3c'

- 4 : Run "python main.py"

## Results :
- MoveToBeacon (with the action "move_screen" for now) :
![alt text](https://github.com/E-tanok/projects_pictures/blob/master/ReinforcementLearning/starcraft_2_pysc2/a3c/a3c_5_workers_action_move_screen.png)
*The agent equalize the Deepmind's performances on the game MoveToBeacon ; this results provides from 5 agents trained with the actions spectrum restricted on ['331']/"move_screen"*
- Other mini-games :
*In future pushes*


# Acknowledgement :

I would like to thank my tutor, [Judith Abécassis](bit.ly/judith_abecassis), which accompanied me in my journey of data science student.

I would also like to thank [Arthur Juliani](https://twitter.com/awjuliani) whose tutorials learn't me a lot to understand the reinforcement learning field. My A3C implementation is inspired from  [it's own implementation](http://bit.ly/a_jul_a3c), with some modifications.

Finally, I would like to thank the [psys2 community](bit.ly/discord_pysc2), particularly :
  - [Ring Roman](https://github.com/inoryy). About that, don't hesitate to check it's own project, [reaver](https://github.com/inoryy/reaver-pysc2) which allows to train agents in starcraft2 and in other games too.
  - [Steven Brown](https://chatbotslife.com/@skjb). He proposes tutorials to build scripted pysc2 agents : a good way to understand the library.
  - "petitTofu42" and other members of the discord!
