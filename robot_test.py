import yaml
import argparse
from robot_move import MoveRobot

if __name__ == "__main__":

    parser = argparse.ArgumentParser('robot Test', add_help=False)

    # model config   
    parser.add_argument("--config-file", 
        default="./configs/spider.yaml", 
        metavar="FILE", help="path to config file")    
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    
    args = parser.parse_args()
    
    model_config_path = args.config_file

    # reads model config
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    model_config['config_path'] = model_config_path
    model_config['gpu_id'] = args.gpu
    model_config['mode'] = 'test'

    print(f'\n ===== Config File Path given: {model_config_path} =====\n')   
    print(yaml.dump(model_config, sort_keys=False))

    robot_trainer = MoveRobot(model_config,)
    robot_trainer.run()

