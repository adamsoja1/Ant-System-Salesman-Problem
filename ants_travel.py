import numpy as np
import random
from distance_matrix import prepare_matrix,init_pheromones

DIST_MATRIX = prepare_matrix('cities_4.txt')

PHEROMONES = init_pheromones(DIST_MATRIX)


class Ant:
    def __init__(self,
                 cities_count:int):
        
        self.beta = 5
        self.alpha = 1
        
        self.cities_count = cities_count
        self.start_city = np.random.randint(0,cities_count-1)
        
        self.visited_cities = []
        self.cities_to_visit = [i for i in range(10)]
        self.current_city = None
        
        """Appending starting city to list of visited cities
        Method to visit a city"""
        
        self.visit(self.start_city)

    def visit(self,
              city:int):
        self.visited_cities.append(city)
        self.cities_to_visit.remove(city)
    
    def total_cost(self):
        if len(self.visited_cities) == 10:
            route = self.visited_cities
            vals = []
            for i in range(len(route)-1):
                vals.append(DIST_MATRIX[route[i]][route[i+1]])
            vals.append(DIST_MATRIX[route[0]][route[self.cities_count-1]])
        return np.sum(vals)
       
    def deposit_pheromones(self,pheromones):
        """1/length"""
        pass


    def get_propabilities(self):
        if self.current_city is None:
            self.current_city = self.start_city
        temp_phero = []
        distances = []
        for i in range(len(self.cities_to_visit)):
            distances.append(DIST_MATRIX[self.current_city][self.cities_to_visit[i]])
            temp_phero.append(PHEROMONES[self.current_city][self.cities_to_visit[i]])
            
        distances = np.array(distances)
        distances = 1/distances**self.beta
        
        temp_phero = np.array(temp_phero)
        temp_phero = temp_phero**self.alpha
        
        temp_sum = np.sum(distances*temp_phero)
        
        a_ij = []
        print(self.cities_to_visit)
        for i in range(len(self.cities_to_visit)):
            temp = (PHEROMONES[self.current_city][self.cities_to_visit[i]]**self.alpha) * ((1/DIST_MATRIX[self.current_city][self.cities_to_visit[i]]) ** self.beta)
            a_ij.append(temp/temp_sum)
        
        probs = []
        for i in range(len(a_ij)):
            probs.append(a_ij[i]/np.sum(a_ij))
        return probs

    def decision(self):
        
        probs = self.get_propabilities()
        
        random_number = np.random.uniform(0,1)
        number = 0
        for i in range(len(probs)):
            if number >= random_number:
                break
            number += probs[i]
        return self.cities_to_visit[i]   
    
    def move(self):
        for i in range(len(self.cities_to_visit)):
            
            city = self.decision()
            print(city)
            self.visit(city)
        



ant = Ant(10)
ant.move()
ant.total_cost()




class Population:
    def __init__(self,
                 population_size:int,
                 cities_count:int):
        
        self.ants = []
        for specimen in range(population_size):
            self.ants.append(Ant(cities_count))
        
        
        
population = Population(100,10)   
 
class Path:
    def __init__(self,
                 population_size:int):
        self.popopulation = []
        
        """Initialization of pheromones matrix"""
        self.pheromones = self.init_pheromoes()
        
    def init_pheromoes(self):
        """Creates matrix of pheromones level between cities (indexes)"""
        pheromone_matrix = {}
        ln = len(DIST_MATRIX)
        for i in range(ln):
            for j in range(i+1, ln):
                dist = 0.1
                
                if i not in pheromone_matrix:
                    pheromone_matrix[i] = {}
                if j not in pheromone_matrix:
                    pheromone_matrix[j] = {}
                
                pheromone_matrix[i][j] = dist
                pheromone_matrix[j][i] = dist
        return pheromone_matrix
    
    
    

path = Path(500)
xd = path.pheromones       
        