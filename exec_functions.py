def clean_sc2_temp_folder(tmp_maps_path, clean_limit, time_limit):
    """
    This function goal is to clean the starcraftII temp files folder which is filled when game instances are launched

    'tmp_maps_path' : is the path of your starcraftII temp files folder
    'clean_limit' : defines the limit number of files from which you  want to start cleaning the folder
    'time_limit' : defines the time limit, in seconds, for a file/folder seniority before being deletable
    """
    import os
    from os import listdir
    import shutil
    import time
    try :
        L_content = listdir(tmp_maps_path)
        if len(L_content)>=clean_limit:
            for content in L_content:
                content_creation_time = os.stat(tmp_maps_path+content).st_mtime

                seniority = time.time()-content_creation_time
                if seniority>time_limit:
                    try:
                        shutil.rmtree(tmp_maps_path+content, ignore_errors=True)
                        os.remove(tmp_maps_path+content)
                    except FileNotFoundError:
                        pass
                    except:
                        pass

    except FileNotFoundError:
        pass

def build_path(mypath):
    import os
    if not os.path.exists(mypath):
        os.makedirs(mypath)

def remove_path(mypath):
    import shutil
    try:
        shutil.rmtree(mypath)
    except FileNotFoundError:
        pass

def load_map_config(map_to_train, custom_params, restore, map_to_restore, restored_policy_type):
    """
    - 'map_to_train' [str]: The map where you want to train your agents
    - 'custom_params' [bool]:
        - If True : you want to import custom params from "parameters_custom.py"
        - If False : you want to import the default parameters from "parameters_default.py"
    - 'restore' [bool]:
        - If True : you want to restore a pretrained session
        - If False : you want to train a session from scratch
    - 'map_to_restore' [str] : From which pre-trained map do you want to restore your session
    - 'restored_policy_type' ['a3c' or 'random']: What kind of policy do you want to use
    """
    import pickle
    import parameters_default
    import parameters_custom
    from context import train_path

    if custom_params:
        params = parameters_custom.dict_custom_params
    else:
        params = parameters_default.dict_default_params
    to_print = "\n\nTRAINING IN : %s\nIMPORTING CUSTOM PARAMS : %s\nRESTORING WEIGHTS : %s"%(map_to_train, custom_params, restore)

    if restore:
        loading_path = train_path+"%s\\%s\\"%(map_to_restore, restored_policy_type)
        params = pickle.load(open(loading_path+'dict_params.p', "rb"))
        to_print+= " | FROM : %s"%(map_to_restore)

    training_path = train_path+"%s\\%s\\"%(map_to_train, 'random' if params['worker_params']['with_random_policy'] else 'a3c')
    if not restore:
        remove_path(training_path)

    build_path(training_path)
    pickle.dump(params, open(training_path+'dict_params.p', "wb"))

    print("\nTRAINING INFORMATIONS : "+to_print+"\n\n")

    return params, training_path
