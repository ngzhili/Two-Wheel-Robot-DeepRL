# Balancing, Moving, and Traversing Slopes with a Two-Wheel Robot using Deep Reinforcement Learning

The objective of this project is to train a Two-Wheel Robot (TWR) to balance and move to a destination while clearing a slope using Deep Reinforcement Learning. 
This project was done by: Ng Zhili, Foo Jia Yuan, Natasha Soon

![slope_demo](https://user-images.githubusercontent.com/69728128/234353570-e9368a8a-5388-4a23-b4ae-fe567f65436a.gif)

## Setup & Installation of Repository
The code is written using Python 3.8 and used the following libraries:
```
matplotlib==3.7.1
numpy==1.23.5
opencv_python==4.7.0.72
pandas==2.0.0
pybullet==3.2.5
PyYAML==6.0
seaborn==0.12.2
tensorflow==2.12.0
tensorflow_probability==0.19.0
torch==1.7.1
```

1. Create conda environment
``` 
conda create -n two_wheel_robot python=3.8 
```
2. Install Dependencies
``` 
pip install -r requirements.txt 
```

## Models used
| Model | Agent | State Space   | Action Space | Actions per time step   |
| :---: | :---:  | :---: | :---: | :---: |
| DQN | Single | Continuous   | Discrete   | Single (Both Wheels) |
| DQNMA | Single | Continuous   | Discrete   | Multiple (for each wheel) |
| REINFORCE | Single | Continuous   | Discrete   | Single (Both Wheels) | 
| A2CSA | Single | Continuous   | Discrete   | Single (Both Wheels) |
| MAA2C | Multiple | Continuous   | Discrete   | Multiple (for each wheel) |
| SAC | Single | Continuous   | Continuous   | Single (Both Wheels) |

## Folders
### 1. Object Models
This folder stores the two-wheel robot xml file created in Gazebo. The object models for the slope is also stored here. 

### 2. Configs
This folder stores the configuration files (.yaml) for running the training and testing experiments. 

### 3. Results
This folder stores the training and testing results of the model runs and the saved models.

## Source Code

### 1. robot_environment.py
This file contains the robot's environment that interfaces with the PyBullet environment. It loads the Two-Wheel Robot model into the Pybullet environment and sets up the terrain. The code also retrieves the state observations of the Two-Wheel Robot and controls the robot based on the actions given by the model. This class also defines the reward functions to be given to the agent during training.

### 2. robot_move.py
This file contains the MoveRobot class for moving the robot during training and testing.
This is the main file where all classes in the source code interact with each other during training/testing. The class reads the configuration file parameters to initialise other classes. The Two-Wheel Robot is trained by running through episodes. All training and testing metrics are logged and the results are plotted in the results subdirectories.

### 3. robot_agent.py
This defines the operations and functions for the tensorflow models, the class can update the model weights via backpropagating. The class also enables saving of the tensorflow model weights.

### 4. robot_neural_network.py
This file contain both PyTorch and Tensorflow 2 model structures. It defines the Fully-Connected Networks of the Tensorflow models and the model classes of the PyTorch DQN and SAC. The select_action functions for the models are also defined here to select greedy actions or random actions for balancing the exploration and exploitation trade-off.

### 5. robot_train.py
The main code for training an agent to balance and move the Two-Wheel Robot. It takes in a training configuration file which stores all the training parameters. Users can also enable DIRECT mode (Not using GUI) to speed up training.

### 6. robot_test.py
The main code for testing the trained agent. The user can run this file to validate and test the model to visualise the performance of the trained agent.


## Training Robot Agent
1. Define your training config file by creating a new config.yaml under configs directory. Rename the run name for every new run so that a new subdirectory will be created under results/train/<run_name> for storing the training plots and model weights.

For example in main.yaml:
```
# ======== CONFIGURATION FILE FOR TRAINING / TESTING ========
# RENAME THE RUN NAME FOR A NEW RUN
run: 
  name: DQN_Train

model:
  name: DQN
  learning_rate_actor: 1.0e-6
  learning_rate_critic: 1.0e-3
  batch_size: 1 # num of steps before update
  gamma: 0.99
  hidden_layer_size: [512,256,128]
  weight_decay: [0,0,0]
  dropout_rate: [0,0,0]
  tau: 0.005

epsilon_greedy:
  eps_init: 0.95
  eps_end: 0.05
  eps_decay: 1.0e-4
  epsilon_decay_type: 'exponential'
  stretched_A: 0.5
  stretched_B: 0.1
  stretched_C: 0.1

training:
  device: 'cpu'
  num_workers: 40 # num of cpu cores to use for parallel computation
  num_train_episodes: 3000
  max_steps_per_episode: 50000 # 20secs per ep
  n_step: False
  base_results_dir: './results/train/'
  
  save_model_weights: True

testing:
  device: 'auto' # auto, cpu, cuda:0
  num_workers: 40 # num of cpu cores to use for parallel computation
  num_test_episodes: 1000
  max_steps_per_episode: 50000
  base_results_dir: './results/test/' # defines the base directory to create a folder with the config yaml name
  record_video: True

environment:
  render_mode: 'DIRECT' # DIRECT, GUI
  video_mode: False # for recording video in direct mode
  enable_keyboard: False
  environment_type: 'FLAT'
  x_distance_to_goal: 30
  time_step_size: 1/20 # user time step size, controls frequency of action made by the robot
  distance_to_goal_penalty: 0.7
  roll_penalty: 0.5
  yaw_penalty: 0.3
  time_penalty: 0.3
  target_velocity_change: 0.5

plotting:
  plot_trajectories_episode_interval: 100
  record_trajectory_time_step_interval: 5 # decrease to increase rate of recording coordinates

```

2. After editing the config file, run the following python code to train the agent.
```
python robot_train.py --config-file ./config/main.yaml
```

## Testing Robot Agent
1. After training a model, find the path tp the saved config file in the results/train/<run_name> directory after training to load the trained model weigths path.

2. Run the following code to test the model
```
python robot_test.py --config-file ./results/train/hyperparam_DQN_slope_40m/epsilon_decay_type_linear/hyperparam_DQN_slope_40m.yaml
```

## Running Hyperparameter Tuning
1. To run hyperparameter tuning, edit the hyperparameters in the configuration file

For example in main.yaml:
```
# ======= HYPERPARAMETER TUNING ======= #
hyperparameter_tuning:
  hyperparam_type: 'environment'
  hyperparameter_tuning_variable: 'x_distance_to_goal' 
  hyperparameter_value_list: [5, 10, 15]
```
2. Run the following code with the configuration file
```
python robot_train_hyperparameter_tuning.py --config-file ./config/main.yaml
```

## Video Demo
https://user-images.githubusercontent.com/69728128/233765311-ea57ea41-e370-46c6-8069-491c71ef3d8d.mp4

