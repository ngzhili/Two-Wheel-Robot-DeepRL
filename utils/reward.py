import numpy as np
import math
def euclidean_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

class reward_scheme():
    def __init__(self,init_pos) -> None:
        self.fall = -1
        self.reached_goal = 1
        self.neutral = 0
        self.init_pos = init_pos
        self.counter = 0

    def get_state_reward(self,end_episode,is_succeed):

        if is_succeed == True:
            return self.reached_goal
        else:
            if end_episode == True:
                return self.fall
            else:
                return self.neutral
            
    def get_shaped_reward(self,obs,step):


        self.bike_base_position = obs

        # return reward
        return 0
    

    def get_transient_reward(self,obs,step):
        if step == 0:
            self.prev_obs = self.init_pos
        else:
            pass
        # reward = self.stationary_penalty(obs,self.prev_obs)

        self.prev_obs = obs

        # return reward
        return 0
    
    def stationary_penalty(self,obs,prev_obs):

        distance = euclidean_distance(obs, prev_obs)

        return(0.01*euclidean_distance)

    def survival_reward(self,step):
        return(0.001*step)

   
        

