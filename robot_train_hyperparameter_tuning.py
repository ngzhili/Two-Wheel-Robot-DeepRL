import yaml
import argparse
from robot_move import MoveRobot

if __name__ == "__main__":

    parser = argparse.ArgumentParser('robot Train', add_help=False)

    # model config   
    parser.add_argument("--config-file", 
        default="./configs/main.yaml", 
        metavar="FILE", help="path to config file")    
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    
    args = parser.parse_args()
    
    model_config_path = args.config_file

    # reads model config
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
        
    # read config file variables
    model_config['config_path'] = model_config_path
    model_config['gpu_id'] = args.gpu
    model_config['mode'] = 'train'

    print(f'\n ===== Config File Path given: {model_config_path} =====\n')   
    print(yaml.dump(model_config, sort_keys=False))
    
    # get hyperparameter tuning variables from config file
    hyperparam_type = model_config['hyperparameter_tuning']['type']
    hyperparameter_tuning_variable = model_config['hyperparameter_tuning']['tuning_variable']
    hyperparameter_value_list = model_config['hyperparameter_tuning']['value_list']

    print(" ===== Hyperparameter Tuning ===== ")
    print(f"hyperparameter_tuning_variable = {hyperparameter_tuning_variable}")
    print(f"hyperparameter_value_list = {hyperparameter_value_list}\n")

    for run_idx, hyperparameter_value in enumerate(hyperparameter_value_list):
        print(f" ===== Run {run_idx} : {hyperparameter_tuning_variable} = {hyperparameter_value} ===== ")
        # overwrites hyperparameter_tuning_variable
        model_config[hyperparam_type][hyperparameter_tuning_variable] = hyperparameter_value
        robot_trainer = MoveRobot(model_config,
                                  hyperparameter_tuning_variable = hyperparameter_tuning_variable, \
                                  hyperparameter_value = hyperparameter_value)
        robot_trainer.run()



