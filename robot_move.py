import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import shutil
import yaml
import argparse
import json 

from robot_environment import TwoWheelRobot
from utils.metrics import TotalTrainingMetrics
from utils.recordVideoDirectMode import *
from utils.epsilonSchedules import epsilon_function
import torch
import torch.optim as optim
import tensorflow as tf
from utils.general import to_float

class MoveRobot:
    """ Class for moving the Two-Wheel RobotTWR and where all the classes interact with each other.
    """
    def __init__(self, 
                 model_config, 
                hyperparameter_tuning_variable=False,
                hyperparameter_value=False):

        # model config path
        self.model_config_path = model_config['config_path'] # config file path
        self.run_name = model_config['run']['name'] # run name
        self.mode = model_config['mode'] # training or testing mode

        self.record_video = model_config['testing']['record_video'] # whether to record video           
        
        # model specific params
        self.model_name = model_config['model']['name'] # model name
        self.gamma = model_config['model']['gamma'] # discount factor
        self.learning_rate_actor = model_config['model']['learning_rate_actor'] # learning rate of the AdamW optimizer
        self.learning_rate_critic = model_config['model']['learning_rate_critic'] # learning rate of the AdamW optimizer
        self.hidden_layer_size = model_config['model']['hidden_layer_size'] # number of hidden units per layer (list format [first layer num units, second layer num units, ...])
        self.weight_decay = model_config['model']['weight_decay'] # weight decay for learning rate
        self.dropout_rate = model_config['model']['dropout_rate'] # dropout rate for layers
        self.batch_size = model_config['model']['batch_size'] # number of transitions from the replay buffer
        self.tau = model_config['model']['tau'] # update rate of the target network

        import tensorflow as tf
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        
        # Epsilon Greedy Parameters
        self.epsilon_initial = model_config['epsilon_greedy']['eps_init'] # starting value of epsilon
        self.epsilon_end = model_config['epsilon_greedy']['eps_end'] # final value of epsilon
        self.EPS_DECAY = model_config['epsilon_greedy']['eps_decay'] # rate of exponential decay of epsilon
        self.epsilon_decay_type = model_config['epsilon_greedy']['epsilon_decay_type'] # type of epislon schedule
        self.epsilon_A=model_config['epsilon_greedy']['stretched_A']
        self.epsilon_B=model_config['epsilon_greedy']['stretched_B']
        self.epsilon_C=model_config['epsilon_greedy']['stretched_C']

        # initialise training params if mode is training
        if self.mode == 'train':
            self.device_type = model_config['training']['device']
            self.num_workers = model_config['training']['num_workers']
            if torch.cuda.is_available():
                print('cuda available')
            else:
                print('cuda not available')
            if self.device_type == 'auto' or self.device_type == 'gpu': # automatically choose device type
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = model_config['gpu_id']
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = ""
                self.device = torch.device("cpu")

            self.base_results_dir = model_config['training']['base_results_dir']
            self.num_episodes = model_config['training']['num_train_episodes']
            self.max_steps = model_config['training']['max_steps_per_episode']
            self.n_steps = model_config['training']['n_step']
            if self.n_steps == False:
                self.n_steps = self.max_steps
            self.save_model_weights = model_config['training']['save_model_weights']
        
        # initialise testing params if mode is testing
        elif self.mode == 'test':
            self.device_type = model_config['testing']['device']
            self.num_workers = model_config['testing']['num_workers']
            if torch.cuda.is_available():
                print('cuda available')
            else:
                print('cuda not available')
            if self.device_type == 'auto' or self.device_type == 'gpu': # automatically choose device type
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = model_config['gpu_id']
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device("cpu")

            self.base_results_dir = model_config['testing']['base_results_dir']
            self.num_episodes = model_config['testing']['num_test_episodes']
            self.max_steps = model_config['testing']['max_steps_per_episode']
            self.load_model_weight_path = model_config['load_model_weights_path']

            self.epsilon_decay_type = None
            self.save_model_weights = False
            self.record_video = model_config['testing']['record_video']


        # load environment params
        self.video_mode = model_config['environment']['video_mode']
        self.render_mode = model_config['environment']['render_mode']
        self.enable_keyboard = model_config['environment']['enable_keyboard']
        self.environment_type = model_config['environment']['environment_type']
        self.time_step_size = to_float(model_config['environment']['time_step_size'])
        self.x_distance_to_goal=model_config['environment']['x_distance_to_goal']
        self.distance_to_goal_penalty = model_config['environment']['distance_to_goal_penalty']
        self.time_penalty = model_config['environment']['time_penalty']
        self.target_velocity_change = model_config['environment']['target_velocity_change']
        self.goal_type = model_config['environment']['goal_type']
        self.goal_step = model_config['environment']['goal_step']

        if self.goal_step > self.max_steps:
            raise ValueError('goal step cannot be larger than max steps')
        
        # if multi-agent or multi-action model is used
        if self.model_name == 'MAA2C':
            self.multi_action = True
        # else single agent/single action model is used
        else:
            self.multi_action = False
        
        if self.mode == 'test' and self.record_video:
            self.render_mode = 'GUI'
        # create our environment object class
        self.env = TwoWheelRobot(render_mode=self.render_mode,
                        enable_keyboard=self.enable_keyboard,
                        environment_type = self.environment_type,
                        time_step_size = self.time_step_size,
                        x_distance_to_goal = self.x_distance_to_goal,
                        robot_time_penalty = self.time_penalty,
                        target_velocity_change = self.target_velocity_change,
                        record_video = self.record_video,
                        multi_action = self.multi_action,
                        goal_type=self.goal_type
                        )
        
        # get observation dimensions
        self.observation_space_dimension = len(self.env.get_robot_state())
        # get action space dimensions
        self.action_space_dimension = self.env.get_action_space()

        # results dir stores the results of each experiment run
        import os
        self.results_dir = os.path.join(self.base_results_dir,self.run_name)
                
        # if hyperparameter tuning, we create subdirectories to save the results of each run with a new hyperparameter value
        if hyperparameter_tuning_variable:
            self.results_dir = os.path.join(self.results_dir,f"{hyperparameter_tuning_variable}_{hyperparameter_value}")
        
        # clears results directory if exist
        if os.path.exists(self.results_dir) and os.path.isdir(self.results_dir):
            shutil.rmtree(self.results_dir)

        # creates results directory for current run 
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        print(f'Saving results to {self.results_dir}')
        
        if self.model_name == "DQN" or self.model_name == "DQNMA" or self.model_name == 'SAC':
            self.save_model_weights_path = os.path.join(self.results_dir,'best_model.pt')
            model_config['load_model_weights_path'] = self.save_model_weights_path
        else:
            model_config['load_model_weights_path'] = self.results_dir
        
        # save config file to results dir
        with open(os.path.join(self.results_dir, f"{self.run_name}.yaml"), 'w') as f:
            yaml.dump(model_config,f)

        # creates trajectory results directory for current run 
        self.results_dir_trajectory = os.path.join(self.results_dir,'robot_trajectories')
        if not os.path.exists(self.results_dir_trajectory):
            os.makedirs(self.results_dir_trajectory)

         # creates plot results directory for current run 
        self.results_dir_plot = os.path.join(self.results_dir,'graphs')
        if not os.path.exists(self.results_dir_plot):
            os.makedirs(self.results_dir_plot)

        # evaluation params
        self.num_test_episodes = model_config['testing']['num_test_episodes']

        # plotting params
        self.plot_trajectories_episode_interval = model_config['plotting']['plot_trajectories_episode_interval']
        self.record_trajectory_time_step_interval = model_config['plotting']['record_trajectory_time_step_interval']       

        # for testing, use zero epsilon, use full model exploitation
        if self.mode == 'test':
            self.epsilon_initial = 0.0
            self.epsilon_end = 0.0
        
        # initialise epsilon function class for varying epsilon during training
        self.epsilon_function=epsilon_function(self.num_episodes,
                                               eps_init=self.epsilon_initial,
                                               eps_end=self.epsilon_end,
                                               epsilon_decay_type=self.epsilon_decay_type,
                                               A=self.epsilon_A,B=self.epsilon_B,C=self.epsilon_C)
        
        # --------------------------------PURELY FOR MODEL INITIALISATION-------------------------------------------------

        self.tf_model_list = ['A2C','MAA2C','Reinforce']
        self.pt_model_list = ['DQN', 'DQNMA', 'SAC']

        # load DQN based on config model name
        if self.model_name == 'DQN' or self.model_name == 'DQNMA':
            from robot_neural_network import ReplayMemoryDQN, DQN
            if self.model_name == 'DQN':
                self.policy_net = DQN(self.observation_space_dimension, self.action_space_dimension, self.hidden_layer_size).to(self.device)
                self.target_net = DQN(self.observation_space_dimension, self.action_space_dimension, self.hidden_layer_size).to(self.device)
            elif self.model_name == 'DQNMA':
                from utils.general import Differential_Drive
                self.differential_action_scheme = Differential_Drive(self.action_space_dimension,2)
                self.policy_net = DQN(self.observation_space_dimension, len(self.differential_action_scheme.new_mapping), self.hidden_layer_size).to(self.device)
                self.target_net = DQN(self.observation_space_dimension, len(self.differential_action_scheme.new_mapping), self.hidden_layer_size).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
 
            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate_actor, amsgrad=True)
            self.memory = ReplayMemoryDQN(10000)

            # load saved model for testing
            if self.mode == 'test':
                print(f"Loading model from {self.load_model_weight_path}")
                # load saved model weights
                self.policy_net.load_state_dict(torch.load(self.load_model_weight_path, map_location=self.device))
            
            # Set model for parallel computation, move model to devie
            from torch.nn.parallel import DataParallel
            if self.device_type != 'cpu':
                self.policy_net = DataParallel(self.policy_net)
                self.policy_net = self.policy_net.to(self.device)
                self.target_net = DataParallel(self.target_net)
                self.target_net = self.target_net.to(self.device)
            else:
                self.policy_net = DataParallel(self.policy_net)
                self.policy_net = self.policy_net.module.to(self.device)
                self.target_net = DataParallel(self.target_net)
                self.target_net = self.target_net.module.to(self.device)

            # Set the number of worker processes to use for computation
            torch.set_num_threads(self.num_workers)

        elif self.model_name == 'SAC':
            from robot_neural_network import SAC
            from robot_neural_network import ReplayMemorySAC
            self.updates = 0
            self.agent = SAC(num_inputs = self.observation_space_dimension, hidden_layer_size = self.hidden_layer_size[0], learning_rate = self.learning_rate_actor, update_interval = 1, device = self.device, epsilon = self.epsilon_initial, gamma = self.gamma, tau = self.tau)
            self.memory = ReplayMemorySAC(10000)
            # Set model for parallel computation, move model to devie
            from torch.nn.parallel import DataParallel
            if self.device_type != 'cpu':
                self.agent = DataParallel(self.agent)
                self.agent = self.agent.to(self.device)
            else:
                self.agent = DataParallel(self.agent)
                self.agent =self.agent.to(self.device)
            

            if self.mode == 'test':
                print(f"Loading model from {self.load_model_weight_path}")
                load_weight = torch.load(self.load_model_weight_path, map_location=self.device)
                print(load_weight)
                # load saved model weights
                self.agent.load_state_dict(torch.load(self.load_model_weight_path, map_location=self.device))


        elif self.model_name in self.tf_model_list:
            self.num_wheels = 2
            from robot_agent import Agent
            if self.model_name == "MAA2C" or self.model_name == "Reinforce":
                self.agent = Agent(model = self.model_name, 
                            discount_rate = self.gamma, 
                            learning_rate_actor = self.learning_rate_actor, 
                            learning_rate_critic = self.learning_rate_critic,
                            hidden_layer_size=self.hidden_layer_size,
                            weight_decay=self.weight_decay,
                            dropout_rate=self.dropout_rate,
                            action_space_dimension = self.action_space_dimension,
                            observation_space_dimension=self.observation_space_dimension, 
                            epsilon = self.epsilon_initial,
                            num_wheels = self.num_wheels)
            
            elif self.model_name == "A2C":
                self.agent = Agent(model = self.model_name, 
                            discount_rate = self.gamma, 
                            learning_rate_actor = self.learning_rate_actor, 
                            learning_rate_critic = self.learning_rate_critic, 
                            hidden_layer_size=self.hidden_layer_size,
                            weight_decay=self.weight_decay,
                            dropout_rate=self.dropout_rate,
                            action_space_dimension = self.action_space_dimension, 
                            observation_space_dimension=self.observation_space_dimension,
                            epsilon = self.epsilon_initial,
                            num_wheels = self.num_wheels)
            else:
                raise ValueError("Invalid model/algorithm set in config")
            
        else:
            raise ValueError(f"model_name {self.model_name} not recognized")
        
        # initialise training metrics logging function
        self.MetricTotal=TotalTrainingMetrics()
       
        # printing which device used for running
        print(f"Using device: {self.device}")

    def run_episode(self,episode):
        """Run the episode
        """
        steps_original_reward_list = []
        steps_transient_reward_list = []
        x_pos_list = []
        y_pos_list = []

        record_episode = 1

        time_step = 0
        done = 0
        loss = 0

        # get initial robot state
        state = self.env.get_robot_state()
        state = self.env.normalize_observations(state)
        # if DQN, move state tensor to device
        if (self.model_name == 'DQN' or self.model_name == "DQNMA") and not self.enable_keyboard and self.mode == 'train':
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # start logging metrics for episode
        self.MetricTotal.start_episode(0,self.env.robot_base_position)

        # set gravity to -9.8 m/s^2
        p.setGravity(0,0,-9.8)
        
        # run episode by stepping through each step
        for time_step in range(self.max_steps):
            if self.record_video and self.render_mode=='GUI' and (episode == record_episode):
                record_gui_mode(p,time_step)
                
            # capture image frames for DIRECT mode for specified episode
            if self.video_mode and self.render_mode=='DIRECT' and (episode == record_episode): #and time_step % 100 == 0:
                record_direct_mode(p,time_step)
            
            # save trajectories of robot at each time step
            if (time_step % self.record_trajectory_time_step_interval) == 0:
                base_pos, _ = p.getBasePositionAndOrientation(self.env.robotid)
                linear_velocity, angular_velocity = p.getBaseVelocity(self.env.robotid)
                x_pos_list.append(base_pos[0])
                y_pos_list.append(base_pos[1])
                
            """ get action from model and take a step in the environment """
            # enable keyboard manual control for debugging
            if self.enable_keyboard:
                keys = p.getKeyboardEvents()
                if ord('q') in keys:
                    break
                action = self.env.select_manual_action_from_keyboard(keys)
            
            # model to choose action
            else:
                if (self.model_name == 'DQN' or self.model_name == "DQNMA"):
                    from robot_neural_network import select_action_DQN, update_weights
                    if (self.model_name == 'DQN' or self.model_name == "DQNMA") and not self.enable_keyboard and self.mode == 'test':
                        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    
                    action = select_action_DQN(obs = state, n_actions = self.action_space_dimension, policy_net = self.policy_net, device = self.device, epsilon = self.epsilon) # state, self.policy_net, self.device, self.epsilon)
                elif self.model_name == 'SAC':
                    action = self.agent.module.select_action(state, self.epsilon,evaluate=self.mode)  # Sample action from policy
                elif self.model_name in self.tf_model_list:
                    action = self.agent.select_actions(state,mode="train")
                else:
                    print(f'{self.model_name} does not exist')
            # take a step in the environment by applying the action on the robot

            # for multi-action/ multi-agent model
            if self.model_name == 'MAA2C':
                step_action = action
            elif self.model_name == "DQNMA":
                step_action = self.differential_action_scheme.new_mapping[int(action)]
                
            else:
                step_action = action[0]
            # take a step in environment and observe next state and rewards
            next_state, reward, transient_reward, done, suceed, info = \
                self.env.step(step_action, self.model_name, time_step=time_step, goal_step=self.goal_step)
            
            # normalise observations
            next_state = self.env.normalize_observations(next_state)

            # add rewards to list
            steps_original_reward_list.append(reward) # for measuring model performance
            steps_transient_reward_list.append(transient_reward) # use transient reward to update models

            # Check for the maximum displacement travelled by the agent
            base_pos, base_ori = p.getBasePositionAndOrientation(self.env.robotid)
            linear_velocity, angular_velocity = p.getBaseVelocity(self.env.robotid)

            self.MetricTotal.intermediate_episode(base_pos,linear_velocity)
            
            """ ===== update models during training ===== """
            if self.mode == 'train':
                # update weights of SAC model
                if (self.model_name == 'SAC') :
                    if len(self.memory) > self.batch_size:
                        # Number of updates per step in environment
                        updates_per_step = 1
                        for i in range(updates_per_step):
                            # Update parameters of all the networks
                            policy_loss = self.agent.module.update_weights(self.memory, self.batch_size, self.updates)
                            self.updates += 1
                            loss += policy_loss
                    if time_step == self.max_steps:
                        mask = 1 
                    else:
                        mask = float(not done)
                    
                    # Add transition to memory
                    self.memory.push(state, action, reward, next_state, mask) 

                # update weights of DQN model
                elif (self.model_name == 'DQN' or self.model_name == 'DQNMA') and not self.enable_keyboard:
                    
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    transient_reward = torch.tensor(transient_reward, dtype=torch.float32, device=self.device).unsqueeze(0)
                    
                    # Store the transition in memory
                    self.memory.push(state, action, next_state, transient_reward)

                    # Perform one step of the optimization (on the policy network)
                    pyt_loss = update_weights(self.policy_net, self.target_net, self.optimizer, self.memory, self.batch_size, self.gamma, self.device)

                    loss += pyt_loss.item()

                    # Soft update of the target network's weights: θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                    self.target_net.load_state_dict(target_net_state_dict)

                # update weights of tensorflow models
                elif self.model_name in self.tf_model_list:
                    if self.agent.model == "MAA2C":
                        actor_observations=state # Local state observed by the acting mechanism (e.g. wheel)
                        actor_observations_prime = next_state
                        critic_observations = state
                        critic_observations_prime=next_state # Global State observed; >= actor observation 

                        tf_loss = self.agent.update_weights(model_name=self.agent.model,
                                                            observations=critic_observations, 
                                                            reward=reward, 
                                                            observations_prime=critic_observations_prime,
                                                            is_done=done, 
                                                            actor_observations=actor_observations)
                        actor_observations = actor_observations_prime
                        critic_observations = critic_observations_prime

                    # apply gradients for combined hybrid actor critic 
                    elif self.agent.model == "A2C" or  self.agent.model == "Reinforce": 
                        tf_loss = self.agent.update_weights(model_name=self.agent.model, 
                                                  observations=state, 
                                                    reward = reward, 
                                                  observations_prime=next_state, 
                                                  is_done=done)
                    # for plotting of loss over steps
                    if type(tf_loss) == list:
                        loss += tf_loss[0]
                    elif type(tf_loss) == float:
                        loss += tf_loss
                        
            # assign previous state to current state
            state = next_state

            # if episode has ended, log the metrics
            if done:
                # calculate cumulative rewards
                cumulative_reward = sum(steps_original_reward_list)
                cumulative_transient_reward = sum(steps_original_reward_list)

                # get final base position for logging
                base_pos, base_ori = p.getBasePositionAndOrientation(self.env.robotid)
                # if DQN model, we move the transient reward to cpu for logging
                if (self.model_name == "DQN" or self.model_name == 'DQNMA') and not self.enable_keyboard and self.mode =='train':
                    transient_reward = transient_reward.to('cpu').numpy()[0]

                # add all logging metrics to the MetricTotal class at the end of the episode
                self.MetricTotal.end_episode(transient_reward, cumulative_transient_reward, reward, cumulative_reward, time_step, base_pos, self.epsilon,loss)
                
                # add robot final trajectory position (x,y) coordinates to lists
                x_pos_list.append(base_pos[0])
                y_pos_list.append(base_pos[1])
                break

        # plot the episode trajectory
        plt.plot(y_pos_list, x_pos_list, color=self.robot_trajectory_color)
        
        # sitch video for DIRECT mode for specified episode
        if ((self.record_video and self.render_mode =='GUI') or \
            (self.video_mode and self.render_mode =='DIRECT')) \
                and episode == record_episode:
            stitch_video_direct_mode(episode)
        
        return steps_transient_reward_list[-1], suceed, time_step

    def plot_trajectories(self,episode_index):
        plt.xlabel('Y position (m)')
        # plt.xlim(-3, 3)
        plt.ylabel('X position (m)')
        # plt.ylim(0, 12)
        plt.title(f'robot path over episodes_{episode_index}')
        # plt.show()
        plt.savefig(os.path.join(self.results_dir_trajectory,f'robot_path_over_episodes_{episode_index}.png'))
        # Clear the current axes.
        plt.cla() 
        # Clear the current figure.
        plt.clf() 
        # Closes all the figure windows.
        plt.close('all')
        import gc
        gc.collect() 
        plt.figure()

    def run(self):
        """ Run the training/testing loop.
        """
        print(f"goal type: {self.goal_type}")
        mode_caption = 'TRAINING' if self.mode == 'train' else 'TESTING'            
        print(f"\n ===== START {mode_caption} ===== ")
        
        total_success = 0
        import time
        start_time = time.time()
        self.robot_trajectory_color = np.random.rand(3,)
        plt.figure()

        # loop through episodes
        for episode in range(self.num_episodes):
            if episode != 0:
                self.env.reset()

            self.epsilon = self.epsilon_function.get_current_epsilon(episode)
           
            ep_reward, is_succeed, end_step = self.run_episode(episode)            
            total_success += is_succeed
            print(f"\r\033[KEpisode: {episode+1}/{self.num_episodes} | Sucesses so far: {total_success}/{episode+1} | Sucess Rate: {round(total_success*100/(episode+1),2)}% | epsilon threshold: {round(self.epsilon,5):.5f} | transient rewards: {round(ep_reward,5):.5f} | survival_time: {round(end_step*self.time_step_size,2):.2f} | end_step: {end_step}", end='', flush=True)
            
            # plot trajectories for TWR at an episode interval
            if episode != 0 and episode% self.plot_trajectories_episode_interval==0:
                self.plot_trajectories(episode)
                if episode+1 != self.num_episodes:
                    self.robot_trajectory_color = np.random.rand(3,)
        
        # plot last episode trajectories
        self.plot_trajectories(episode+1)
        self.MetricTotal.end_training()
        self.MetricTotal.plot_all(self.time_step_size, self.results_dir_plot)
        
        plt.clf()
        plt.figure().clear()
        plt.close('all')

        print(f"\r\033[KEpisode: {episode+1}/{self.num_episodes} | Sucesses so far: {total_success}/{episode+1} | Sucess Rate: {round(total_success*100/(episode+1),2)}% | epsilon threshold: {round(self.epsilon,5):.5f}  | transient rewards: {round(ep_reward,5):.5f} | survival_time (secs): {round(end_step*self.time_step_size,2):.2f} | end_step: {end_step} ")

        from datetime import timedelta
        elapsed = time.time() - start_time
        print(f"--- {mode_caption} took {timedelta(seconds=elapsed)} , HH : MM : SS ---")
        print(f"===== {mode_caption} COMPLETED ===== \n\n")

        # save DQN policy net weights
        if self.save_model_weights and (self.model_name=="DQN" or self.model_name=='SAC' or self.model_name == "DQNMA"):
            print(f"saved model weights to {self.save_model_weights_path}")

            if self.model_name=="DQN" or self.model_name == "DQNMA":
                model_class = self.policy_net
            # SAC model
            else: 
                model_class = self.agent.module.policy

            if self.device_type !='cpu':
                torch.save(model_class.module.state_dict(), self.save_model_weights_path)
            else:
                torch.save(model_class.state_dict(), self.save_model_weights_path)
        elif self.save_model_weights:
            self.agent.save_models(self.results_dir)

        return
