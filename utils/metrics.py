import os
import numpy as np
import matplotlib.pyplot as plt

from .generics import euclidean_distance

def average(list):
    return sum(list)/len(list)

class TotalTrainingMetrics():
    def __init__(self):
        # self.metrics contains all the episodes' metrics in the entire training loop
        self.metrics = {'total_transient_reward':[],
                        'total_reward':[],
                        'total_num_steps':[],
                        'total_displacement':[],
                        'max_displacement': [],
                        'epsilon':[],
                        'velocity':[],
                        'loss':[]}
        

    def displacement(self,init_pos, curr_pos):
        return (curr_pos[0]-init_pos[0]) # Returns only the displacement in x-axis

    def start_episode(self,step,init_pos):
        self.init_pos = init_pos
        self.each_episode_cumulative_reward = 0.0
        self.each_episode_num_steps = 0
        self.episode_max_disp = 0.0
        self.Vx = []

        return
    
    def intermediate_episode(self,curr_pos,linear_velocity):
        curr_step_disp = self.displacement(self.init_pos, curr_pos)
        self.episode_max_disp = max(curr_step_disp,self.episode_max_disp)

        self.Vx.append(linear_velocity[0])

        return

    def end_episode(self,episode_transient_reward, episode_cumulative_transient_reward, \
                    episode_reward, episode_cumulative_reward, \
                    end_step, end_pos, epsilon,loss):
        # at the end of the episode, we add the metrics from the episode to the training metric dictionary
        self.metrics['total_transient_reward'].append(episode_transient_reward)
        self.metrics['total_reward'].append(episode_reward)
        self.metrics['total_num_steps'].append(end_step)
        self.metrics['total_displacement'].append(self.displacement(self.init_pos, end_pos))
        self.metrics['max_displacement'].append(self.episode_max_disp)
        self.metrics['epsilon'].append(epsilon)
        self.metrics['velocity'].append((average(self.Vx),
                                         min(self.Vx),
                                         max(self.Vx)))
        self.metrics['loss'].append(loss)
        return 

    def end_training(self):
        # at the end of episode we calculate the average metrics
        self.episode_count = len(self.metrics['total_transient_reward'])
        self.episode_list = [i for i in range(self.episode_count)]
        self.avg_transient_reward = average(self.metrics['total_transient_reward'])
        self.avg_reward = average(self.metrics['total_reward'])
        self.avg_disp = average(self.metrics['total_displacement'])
        return 

    def plot_reward(self,path=""):
        plt.figure()
        x = self.episode_list
        y = self.metrics['total_transient_reward']
        plt.plot(x,y)
        plt.xlabel('Episodes')
        plt.ylabel('Transient Reward')
        plt.title(f"Transient Reward Over {self.episode_count} Episodes")
        plt.savefig(os.path.join(path,f'transient_reward_over_episode_{self.episode_count}.png'))
        plt.clf()
        
    def plot_rolling_avg_transient_reward(self,path=""):
        plt.figure()
        eps = []
        rwds_subset = []
        rolling_window_interval = int(self.episode_count * 0.1) #100
        x = self.episode_list[::rolling_window_interval]

        rwds = self.metrics['total_transient_reward']
        y = [np.mean(rwds[:m]) for m in range(self.episode_count)][::rolling_window_interval]
        plt.plot(x, y)
        plt.xlabel('Episodes')
        plt.ylabel('Rolling transient average reward')
        plt.title(f"Rolling average transient reward Over {self.episode_count} Episodes")
        plt.savefig(os.path.join(path,f'rolling_avg_transient_reward_over_episode_{self.episode_count}.png'))
        plt.clf()

    def plot_step(self,path=""):
        plt.figure()
        x = self.episode_list
        y = self.metrics['total_num_steps']
        plt.plot(x,y)
        plt.xlabel('Episodes')
        plt.ylabel('Timesteps')
        plt.title(f"Timesteps Over {self.episode_count} Episodes")
        plt.savefig(os.path.join(path,f'timestep_over_episode_{self.episode_count}.png'))
        plt.clf()
        return
    
    def plot_loss(self,path=""):
        plt.figure()
        x = self.episode_list

        # Loss Per Timestep
        # y = [a / b for a, b in zip(self.metrics['loss'], self.metrics['total_num_steps'])]

        # Absolute Loss
        y = self.metrics['loss']
        plt.plot(x,y)
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.title(f" Loss Per Timestep Over {self.episode_count} Episodes")
        plt.savefig(os.path.join(path,f'loss_timestep_over_episode_{self.episode_count}.png'))
        return

    def plot_duration_alive(self,time_step_size, path=""):
        plt.figure()
        print(time_step_size)
        x = self.episode_list
        y = [episode_steps * time_step_size for episode_steps in self.metrics['total_num_steps']]
        plt.plot(x,y)
        plt.xlabel('Episodes')
        plt.ylabel('Seconds Alive')
        plt.title(f"Num of Seconds Alive Over {self.episode_count} Episodes")
        plt.savefig(os.path.join(path,f'seconds_alive_over_episode_{self.episode_count}.png'))
        plt.clf()
        return
    
    def plot_disp(self,path=""):
        plt.figure()
        x = self.episode_list
        y = self.metrics['total_displacement']
        plt.plot(x,y)
        plt.xlabel('Episodes')
        plt.ylabel('Displacement')
        plt.title(f"Displacement Over {self.episode_count} Episodes")
        plt.savefig(os.path.join(path,f'displacement_over_episode_{self.episode_count}.png'))
        plt.clf()
        return

    def plot_disp_max(self,path=""):
        plt.figure()
        x = self.episode_list
        y = self.metrics['max_displacement']
        plt.plot(x,y)
        plt.xlabel('Episodes')
        plt.ylabel('Displacement')
        plt.title(f"Maximum Displacement Achieved Over {self.episode_count} Episodes")
        plt.savefig(os.path.join(path,f'max_displacement_over_episode_{self.episode_count}.png'))
        return

    def plot_velocities(self,path=""):
        # Extract individual lists for each value
        t = self.episode_list  # time
        y1 = [t[0] for t in self.metrics['velocity']]  # red
        y2 = [t[1] for t in self.metrics['velocity']]  # blue
        y3 = [t[2] for t in self.metrics['velocity']]  # green

        # Plot the data
        plt.plot(t, y1, 'r', label='Average Velocity')
        # plt.plot(t, y2, 'b', label='Minimum Velocity')
        # plt.plot(t, y3, 'g', label='Maximum Velocity')
        plt.xlabel('Episodes')
        plt.ylabel('Velocity')
        plt.legend()
        plt.title(f"Velocity Over {self.episode_count} Episodes")
        plt.savefig(os.path.join(path,f'velocity_over_episode_{self.episode_count}.png'))
        return

    
    def plot_goal_percentage_reached(self, path=""):
        """Get the Goal Percentage Reached in episode intervals.

        Args:
            epi_interval (int, optional): Determine amount to increase episodes by for each bar. Defaults to 100.
        """
        epi_interval = int(self.episode_count * 0.1)
        # dictionary to store interval_x_labels, goal_reached_percentage, epi_interval
        self.goal_percentage_reached_dict={}
        # list to store goal_reached_percentage
        goal_reached_percentage = []
        # list to store end episodes of interval
        interval_x_labels = []

        # initialise start episode and end episode range for first bin interval
        start_epi = 0
        end_epi = epi_interval

		# Loop through each increment
        num_bins = int(self.episode_count/epi_interval)
        for i in range(num_bins):
			# counter variable when goal is reached set to 0
            total_goal_reached_count = 0
			# Loop in the episode range for this increment
            for j in range(start_epi, int(end_epi)): # For every goal reached, increase counter
                if self.metrics['total_reward'][j] == 1:
                    total_goal_reached_count += 1

            # add goal reached percentage for each interval to goal_reached_percentage list
            goal_reached_percentage.append(total_goal_reached_count*100/epi_interval)
            # add end episode to interval x labels
            interval_x_labels.append(end_epi)
            # for next interval, assign start episode as current interval's end episode
            start_epi = end_epi
            # for next interval, assign end episode by adding episode_interval to current interval's end episode
            end_epi += epi_interval

        # assign values to goal_percentage_reached_dict
        self.goal_percentage_reached_dict['goal_reached_percentage'] = goal_reached_percentage
        self.goal_percentage_reached_dict['interval_x_labels'] = interval_x_labels
        self.goal_percentage_reached_dict['epi_interval'] = epi_interval
        
        plt.figure()
        plt.bar(self.goal_percentage_reached_dict['interval_x_labels'],  
                self.goal_percentage_reached_dict['goal_reached_percentage'], 
                width = - 0.9*self.goal_percentage_reached_dict['epi_interval'], align = "edge")
        plt.xlabel("Episodes")
        plt.ylabel("Goal Reached Percentage (%)")
        plt.title(f"Goal Reached Percentage during training with bin interval of {self.goal_percentage_reached_dict['epi_interval']} episodes")
        plt.savefig(os.path.join(path,f'goal_reached_percentage_over_episode_{self.episode_count}.png'))
        plt.clf()
        return
    
    def plot_epsilon_decay(self,path=""):
        plt.figure()
        x = self.episode_list
        y = self.metrics['epsilon']
        plt.plot(x,y)
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.title(f"Epsilon Over {self.episode_count} Episodes")
        plt.savefig(os.path.join(path,f'Epsilon_over_episode_{self.episode_count}.png'))
        plt.clf()

    def plot_all(self, time_step_size, path=""):
        self.plot_reward(path)
        self.plot_step(path)
        self.plot_disp(path)
        self.plot_disp_max(path)
        self.plot_duration_alive(time_step_size,path)
        self.plot_epsilon_decay(path)
        self.plot_loss(path)

        self.plot_goal_percentage_reached(path)
        self.plot_rolling_avg_transient_reward(path)

        self.plot_velocities(path)
        return

    def __getattribute__(self, __name: str):
        if __name=="size":
            return (len(self.metrics))
        elif __name=="reward":
            self.avg_reward=sum([m['cumulative_reward'] for m in self.metrics]) / len(self.metrics)
            print(f"The average reward is {self.avg_reward}")
        elif __name=="steps":
            self.avg_steps = sum([m['num_steps'] for m in self.metrics]) / len(self.metrics)
            print(f"The average reward is {self.avg_steps}")
        else:
            return super().__getattribute__(__name)

# Create an instance of the TotalTrainingMetrics class
total_metrics = TotalTrainingMetrics()
