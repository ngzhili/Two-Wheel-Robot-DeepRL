import re
import itertools

def to_float(str):
    is_frac = bool(re.search("/", str))
    if is_frac:
        num_den = str.split("/")
        return float(num_den[0]) / float(num_den[1])
    else:
        return float(str)
    

class Differential_Drive():
    def __init__(self,action_space,wheels) -> None:
        
        self.action_space=action_space
        
        self.new_mapping = list(itertools.permutations(range(self.action_space),wheels))
