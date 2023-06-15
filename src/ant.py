"""

"""
import numpy as np
import random
class AntPopulation:
    def __init__(self,
                 population_size:int,
                 r_path,
                 l_path):
        
        self.first_branch = None
        self.right = 0
        self.left = 0
        
        self.right_path = r_path
        self.left_path = l_path
        
        self.population_size = population_size
        
        self.k = 20
        self.d = 2
           
            
            
    def probs(self):
        self.pr = ((self.right + self.k)**self.d)/(((self.right + self.k)**self.d) + ((self.left+self.k)**self.d))
        self.pl = 1-self.pr
        weight_right = 1/self.right_path
        weight_left = 1/self.left_path
        if self.right_path is not None:
            self.pr = self.pr + weight_right
            self.pl = self.pl + weight_left

        
    def choice(self):
        self.probs()
        r = np.random.random()
        if r<=self.pr:
            self.right += 1
        else:
            self.left += 1
        
    def main(self):
        for i in range(self.population_size):
            self.choice()
        
        print(f'Right path: {self.right} ants')
        print(f'Left path:  {self.left} ants')
        
    
        
        
        
        
        
        
        
ants = AntPopulation(2000,10,1)
ants.main()
        
        
        