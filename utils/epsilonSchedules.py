
import math
import numpy as np

def exponential_decay_epsilon(episode, num_train_episodes, epsilon_initial, epsilon_end):
    epsilon = (1 - (episode/num_train_episodes) ** 2) * (epsilon_initial - epsilon_end) + epsilon_end
    return epsilon

def linear_decay_epsilon(episode, num_train_episodes, epsilon_initial, epsilon_end):
    epsilon = epsilon_end + (epsilon_initial - epsilon_end) * max(0, (num_train_episodes - episode)) / self.num_train_episodes
    return epsilon


class epsilon_function():
    def __init__(self,
                 num_episodes,
                 eps_init=0.95,
                 eps_end=0.05,
                 eps_decay=1e-4,
                 A = 0.5,
                 B = 0.1,
                 C = 0.1,
                 epsilon_decay_type='linear') -> None:
        """Setting the Epsilon Function for use througout the run

        Args:
            num_episodes (int): 
                Total Number Of Episodes
            eps_init (float, optional): 
                Initial Probability to Explore. 
                Defaults to 0.95.
            eps_end (float, optional): 
                Final Probability to Explore. 
                Defaults to 0.05.
            eps_decay (float, optional): 
                Rate of Decay.
                (Not in USE AFAIK). 
                Defaults to 1e-4.
            A (float, optional): 
                Tendency to Explore vs Exploit 0<A<1. 
                Closer to 1 indicates more likely to explore.
                Defaults to 0.5.
            B (float, optional): 
                Slope of Transition between Exploration and Exploitation. 
                Defaults to 0.1.
            C (float, optional): 
                Steepness of Left and Right Tail. 
                Defaults to 0.1.
            epsilon_decay_type (str, optional): 
                Choose Between 'linear', 'exponential', and 'stretched'. 
                Defaults to 'linear'.
        """
        
        self.num_episodes = num_episodes
        self.epsilon_initial = eps_init
        self.epsilon_end = eps_end
        self.epsilon_decay = eps_decay
        self.epsilon_decay_type = epsilon_decay_type

        if self.epsilon_decay_type == 'stretched':
            self.A,self.B,self.C = A,B,C

    def linear_decay_epsilon(self,episode):
        epsilon = self.epsilon_end + (self.epsilon_initial - self.epsilon_end) * max(0, (self.num_episodes - episode)) / self.num_episodes
        return float(epsilon)
    
    def exponential_decay_epsilon(self,episode):
        epsilon = (1 - (episode/self.num_episodes) ** 2) * (self.epsilon_initial - self.epsilon_end) + self.epsilon_end
        return float(epsilon)

    def stretched_decay_epsilon(self,episode):
        """
        Highly Influenced by https://medium.com/analytics-vidhya/stretched-exponential-decay-function-for-epsilon-greedy-algorithm-98da6224c22f
        Returns the epsilon value for current episode
        Problem: Inital values higher than 1
        Solution: Normalise it
        Args:
            episode (int): Current Episode

        Returns:
            epsilon (float): epsilon value
        """

        normalised_episode=(episode-self.A*self.num_episodes)/(self.B*self.num_episodes)
        cosh=np.cosh(math.exp(-normalised_episode))
        epsilon=1.1-(1/cosh+(episode*self.C/self.num_episodes))
        return epsilon
    
    def stretched_decay_epsilon_normalised(self,episode) -> float:
        """Gets the normalised epsilon decay

        Args:
            episode (int): Current Episode

        Returns:
            float: stretched epsilon, normalised between maximum and minimum
        """
        stretch_max = self.stretched_decay_epsilon(0)
        stretch_min= self.stretched_decay_epsilon(self.num_episodes)

        epsilon_raw=self.stretched_decay_epsilon(episode)
        normalised= (epsilon_raw-stretch_min)/(stretch_max-stretch_min)

        return float((normalised*(self.epsilon_initial-self.epsilon_end))+ self.epsilon_end)
    
    def get_current_epsilon(self,episode):
        if self.epsilon_decay_type=='linear':
            return self.linear_decay_epsilon(episode)
        elif self.epsilon_decay_type=='exponential':
            return self.exponential_decay_epsilon(episode)
        elif self.epsilon_decay_type=='stretched':
            return self.stretched_decay_epsilon_normalised(episode)           
        else:
            return 0.0
        
if __name__=="__main__":
    eps_fn = epsilon_function(10000)

    print(eps_fn.get_current_epsilon(4000))