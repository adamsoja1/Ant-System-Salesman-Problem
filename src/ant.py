import numpy as np

class AntPopulation:
    def __init__(self, 
                 population_size: int, 
                 r_path:float, 
                 l_path:float):
        
        self.first_branch = None
        self.right = 0
        self.left = 0

        self.right_path = r_path
        self.left_path = l_path

        self.population_size = population_size

        self.k = 20
        self.d = 2

    def probs(self):
        dist_right = self.right_path  
        dist_left = self.left_path 
        

        w_right = 1 / dist_right
        w_left = 1 / dist_left


        self.pr = (((self.right + self.k) ** self.d) * w_right) / (
                    (((self.right + self.k) ** self.d) * w_right) + (((self.left + self.k) ** self.d) * w_left))
        
        self.pl = 1 - self.pr

    def choice(self):
        self.probs()
        r = np.random.random()
        if r <= self.pr:
            self.right += 1
        else:
            self.left += 1

    def main(self):
        for i in range(self.population_size):
            self.choice()

        print(f'Right path: {self.right} ants')
        print(f'Left path:  {self.left} ants')


right = 100
left = 50

ants = AntPopulation(2000,right,left)
ants.main()