import numpy as np
import random
from distance_matrix import prepare_matrix,init_pheromones,load_file
from copy import deepcopy
import time
import matplotlib.pyplot as plt

FILE_PATH = 'src/cities_4.txt'
DIST_MATRIX = prepare_matrix(FILE_PATH)
PHEROMONES = init_pheromones(DIST_MATRIX)

class Ant:
    def __init__(self,
                 cities_count:int):
        
        self.beta = 5
        self.alpha = 1
        
        self.cities_count = cities_count
        self.start_city = np.random.randint(0,cities_count-1)
        self.current_city = None
        self.routes = []
        
        self.visited_cities = []
        self.cities_to_visit = [i for i in range(cities_count)]
        
        
        """Appending starting city to list of visited cities
        Method to visit a city"""
        
        self.visit(self.start_city)

    def visit(self,
              city:int):
        self.visited_cities.append(city)
        self.cities_to_visit.remove(city)
        self.current_city = city
        
    def total_cost(self):
        if len(self.visited_cities) == self.cities_count:
            route = self.visited_cities
            vals = []
            for i in range(len(route)-1):
                vals.append(DIST_MATRIX[route[i]][route[i+1]])
            vals.append(DIST_MATRIX[route[-1]][route[0]])
            return np.sum(vals)
        else:
            raise ValueError("Length of visited cities isnt equal to count_cities")
       


    def get_propabilities(self):
        if self.current_city is None:
            self.current_city = self.start_city
       
        temp_phero = []
        distances = []
        for i in self.cities_to_visit:
            distances.append(DIST_MATRIX[self.current_city][i])
            temp_phero.append(PHEROMONES[self.current_city][i])
            
        distances = np.array(distances)
        distances = 1/distances
        distances = np.power(distances,self.beta)  
        
        temp_phero = np.array(temp_phero)
        temp_phero = temp_phero**self.alpha
        
        temp_sum = np.sum(distances*temp_phero)
        
        a_ij = []
        for city in self.cities_to_visit:
            temp = (PHEROMONES[self.current_city][city]**self.alpha) * ((1/DIST_MATRIX[self.current_city][city]) ** self.beta)
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
            number += probs[i]
            if number >= random_number:
                return self.cities_to_visit[i]   
    
    def move(self):
        for i in range(len(self.cities_to_visit)):
            self.visit(self.decision())
                
        
    
    def create_pher_matrix(self):
        """Creates matrix of pheromones level between cities (indexes)"""
        pheromone_matrix = {}
        ln = len(DIST_MATRIX)
        for i in range(ln):
            for j in range(i+1, ln):
                dist = 0
                
                if i not in pheromone_matrix:
                    pheromone_matrix[i] = {}
                if j not in pheromone_matrix:
                    pheromone_matrix[j] = {}
                
                pheromone_matrix[i][j] = dist
                pheromone_matrix[j][i] = dist
        return pheromone_matrix
    
    def create_temporary_pheromones(self):
        ant_phero = self.create_pher_matrix()
        for i in range(len(self.visited_cities) - 1):
            
            ant_phero[self.visited_cities[i]][self.visited_cities[i+1]] += (1/self.total_cost())
            ant_phero[self.visited_cities[i+1]][self.visited_cities[i]] += (1/self.total_cost())
        ant_phero[self.visited_cities[-1]][self.visited_cities[0]] += (1/self.total_cost())
        ant_phero[self.visited_cities[0]][self.visited_cities[-1]] += (1/self.total_cost())
        cities = self.visited_cities
        return ant_phero


    def reset_to_default(self):
        self.routes.append(self.visited_cities)
        self.current_city = None
        self.visited_cities = []
        self.cities_to_visit = [i for i in range(10)]
        self.visit(self.start_city)

    def get_current_stats(self):
        print(f'Start city: {self.start_city}')
        print(f'cities_to_visit: {self.cities_to_visit}')

class Population:
    def __init__(self,
                 population_size:int,
                 cities_count:int):
        self.evap = 0.5
        self.ants = []
        for specimen in range(population_size):
            self.ants.append(Ant(cities_count))
        
    def get_starting_cities(self):
        for ant in self.ants:
            print(ant.start_city)
            
    def move_ants(self):
        for ant in self.ants:
            ant.move()
     
    def get_best_ant(self):
        costs = [self.ants[i].total_cost() for i in range(len(self.ants))]
        return self.ants[costs.index(min(costs))].visited_cities,min(costs)
    
    def reset(self):
        for ant in self.ants:
            ant.reset_to_default()
        
    def update_pheromone(self):
        
        ln = len(PHEROMONES)
        for i in range(ln):
            for j in range(i+1, ln):
                dist = 0
                
                if i not in PHEROMONES:
                    PHEROMONES[i] = {}
                if j not in PHEROMONES:
                    PHEROMONES[j] = {}
                
                PHEROMONES[i][j] = PHEROMONES[i][j] * self.evap
                PHEROMONES[j][i] = PHEROMONES[j][i] * self.evap
        
        for ant in self.ants:
            temp = ant.create_temporary_pheromones()

            for i in range(ln):
                for j in range(i+1, ln):
                    
                    if i not in PHEROMONES:
                        PHEROMONES[i] = {}
                    if j not in PHEROMONES:
                        PHEROMONES[j] = {}
                    
                    PHEROMONES[i][j] +=  temp[i][j]
                    PHEROMONES[j][i] +=  temp[j][i]
                    


class System:
    def __init__(self,
                 population_size:int,
                 cities_count:int,
                 n_generations:int)->None:
        
        self.population = Population(population_size,cities_count)
        self.n_generations = n_generations
        
        self.best_ant = []
    
    def go(self):
        self.population.move_ants()
        self.population.update_pheromone()
        self.best_ant.append(self.population.get_best_ant())
        self.population.reset()
        
    def run_system(self):
        for i in range(self.n_generations):
            self.go()

        
    def get_results(self):
        return self.best_ant
    
    def get_best_ant(self):
        results = self.get_results()
        ants = []
        path_cost = []
        for ant,cost in results:
            ants.append(ant)
            path_cost.append(cost)
        
        best_ant = path_cost.index(min(path_cost))
        return ants[best_ant],min(path_cost)
        
    def visualize_path_all(self):
        x,y=load_file(FILE_PATH)
        self.draw_route_all_best(x,y)
    
    
    def visualize_path_best(self):
        x,y=load_file(FILE_PATH)
        self.draw_only_best(x,y)
    
    
    def visualize_path_all_no_best(self):
        x,y=load_file(FILE_PATH)
        self.draw_route_all(x,y)
    
    

        
    
    def draw_route_all_best(self,x,y):
        import matplotlib.pyplot as plt
        
        path_cost = self.get_best_ant()[1]
        fig, ax = plt.subplots(figsize = (25,14))
        
        
        for ant in self.population.ants:
            for route_indexes in ant.routes:
                for i in range(len(route_indexes)-1):
                    idx1 = route_indexes[i]
                    idx2 = route_indexes[i+1]
                    ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], 'k-',linewidth =0.01,color = 'blue')
                idx1 = route_indexes[0]
                idx2 = route_indexes[-1]    
                ax.plot([x[idx1] , x[idx2] ], [y[idx1],  y[idx2] ],'k-',linewidth =0.01,color = 'blue')
        
        
        
        
        route_indexes = self.get_best_ant()[0]

        for i in range(len(x)):
            ax.plot(x[i], y[i], marker='x', markersize=30)
            ax.text(x[i]+0.2, y[i]+0.2, f'{i+1}',fontsize=40)
        
        for i in range(len(route_indexes)-1):
            idx1 = route_indexes[i]
            idx2 = route_indexes[i+1]
            ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], 'k-',linewidth = 8,color = 'red')
        idx1 = route_indexes[0]
        idx2 = route_indexes[-1]    
        ax.plot([x[idx1] , x[idx2] ], [y[idx1],  y[idx2] ],'k-',linewidth =8,color = 'red')
        plt.title(f"The best ant's path cost = {path_cost}",fontsize=40)
        ax.spines['bottom'].set_visible(False) 
        ax.spines['left'].set_visible(False)   
        
    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        plt.legend()

    
    
    
    
    def draw_only_best(self,x,y):
        import matplotlib.pyplot as plt
        
        path_cost = self.get_best_ant()[1]
        fig, ax = plt.subplots(figsize = (25,14))
        
        route_indexes = self.get_best_ant()[0]

        for i in range(len(x)):
            ax.plot(x[i], y[i], marker='x', markersize=30)
            ax.text(x[i]+0.2, y[i]+0.2, f'{i+1}',fontsize=40)
        
        for i in range(len(route_indexes)-1):
            idx1 = route_indexes[i]
            idx2 = route_indexes[i+1]
            ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], 'k-',linewidth = 8,color = 'red')
        idx1 = route_indexes[0]
        idx2 = route_indexes[-1]    
        ax.plot([x[idx1] , x[idx2] ], [y[idx1],  y[idx2] ],'k-',linewidth =8,color = 'red')
        plt.title(f"The best ant's path cost = {path_cost}",fontsize=40)
        ax.spines['bottom'].set_visible(False) 
        ax.spines['left'].set_visible(False)   
        
    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        plt.legend()

    
    
    def draw_route_all(self,x,y):
        import matplotlib.pyplot as plt
        
        path_cost = self.get_best_ant()[1]
        fig, ax = plt.subplots(figsize = (25,14))
        
        
        for ant in self.population.ants:
            for route_indexes in ant.routes:
                for i in range(len(route_indexes)-1):
                    idx1 = route_indexes[i]
                    idx2 = route_indexes[i+1]
                    ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], 'k-',linewidth =0.01,color = 'blue')
                idx1 = route_indexes[0]
                idx2 = route_indexes[-1]    
                ax.plot([x[idx1] , x[idx2] ], [y[idx1],  y[idx2] ],'k-',linewidth =0.01,color = 'blue')
        
        
        
        
        route_indexes = self.get_best_ant()[0]

        for i in range(len(x)):
            ax.plot(x[i], y[i], marker='x', markersize=30)
            ax.text(x[i]+0.2, y[i]+0.2, f'{i+1}',fontsize=40)
        
        for i in range(len(route_indexes)-1):
            idx1 = route_indexes[i]
            idx2 = route_indexes[i+1]
            ax.plot([x[idx1], x[idx2]], [y[idx1], y[idx2]], 'k-',linewidth =0.01,color = 'blue')
        idx1 = route_indexes[0]
        idx2 = route_indexes[-1]    
        ax.plot([x[idx1] , x[idx2] ], [y[idx1],  y[idx2] ],'k-',linewidth =0.01,color = 'blue')
        plt.title(f"Ants' routes",fontsize=40)
        ax.spines['bottom'].set_visible(False) 
        ax.spines['left'].set_visible(False)   
        
    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        plt.legend()

    
    
    
    
if __name__ == "__main__":

    
    FILE_PATH = 'cities_4.txt'
    DIST_MATRIX = prepare_matrix(FILE_PATH)
    PHEROMONES = init_pheromones(DIST_MATRIX)

    system = System(population_size=10,
                    cities_count=10,
                    n_generations=200)
    
    system.run_system()
    
    
    
    system.visualize_path_all()
    system.visualize_path_best()
    system.visualize_path_all_no_best()
    
    
    
    
    
    
    