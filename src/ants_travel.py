"""
Ant Colony System for Solving the Traveling Salesman Problem (TSP)

Authors: Adam Soja
Date: 2023

This module contains the implementation of the Ant Colony System (ACS) algorithm
to solve the Traveling Salesman Problem (TSP). The TSP is a classic optimization problem
where the goal is to find the shortest possible route that visits a given set of cities
and returns to the starting city, visiting each city exactly once.

The algorithm is implemented using Python classes to represent the ants, the population,
and the system.

Usage:
1. Define a distance matrix (2D array) representing the distances between cities.
2. Create an instance of the System class, providing the population size, number of cities,
   and the number of generations (iterations).
3. Call the run_system() method to execute the ACS algorithm.
4. Use the visualization methods to visualize the routes taken by ants and the best solution found.

Note: The algorithm assumes a symmetric distance matrix, where distance_matrix[i][j] equals the
distance from city i to city j, and distance_matrix[j][i] equals the distance from city j to city i.

Classes:
--------
Ant:
    A class representing an ant, used in the Ant Colony System algorithm.

Population:
    A class representing the population of ants.

System:
    A class representing the Ant Colony System for solving the Traveling Salesman Problem.

Functions:
----------
prepare_matrix(file_path: str) -> np.ndarray:
    Reads a distance matrix from a file and returns it as a NumPy array.

init_pheromones(distance_matrix: np.ndarray) -> dict:
    Initializes the pheromone matrix based on the provided distance matrix.

load_file(file_path: str) -> Tuple[List[float], List[float]]:
    Reads a list of coordinates (x, y) from a file and returns them as separate lists.

"""
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
    """
    A class to represent an ant in an ant colony optimization algorithm.

    ...

    Attributes
    ----------
    beta : float
        Controls the influence of distance on pheromone trails.

    alpha : float
        Controls the influence of pheromone trails on ant movement.

    cities_count : int
        Total number of cities in the problem.

    start_city : int
        Index of the starting city for the ant's tour.

    current_city : int or None
        Index of the current city the ant is located in, initially set to None.

    routes : list of list of int
        List of routes taken by the ant during multiple tours.

    visited_cities : list of int
        List of city indices visited by the ant during a single tour.

    cities_to_visit : list of int
        List of city indices yet to be visited by the ant.

    ...

    Methods
    -------
    __init__(cities_count: int)
        Initializes an Ant object with random starting city and initial state.
    visit(city: int)
        Marks a city as visited and updates the ant's current city.

    total_cost() -> float
        Calculates the total cost of the ant's tour based on the visited cities.

    get_probabilities() -> list of float
        Calculates the probabilities of choosing each unvisited city for the next move.

    decision() -> int
        Makes a decision on the next city to visit based on probabilities.

    move()
        Performs a move by visiting the next city.

    create_pher_matrix() -> dict
        Creates an empty pheromone matrix for all city pairs.

    create_temporary_pheromones() -> dict
        Calculates the temporary pheromone updates for the ant's visited tour.

    reset_to_default()
        Resets the ant's state to its initial values after completing a tour.

    get_current_stats()
        Prints the current city and cities yet to be visited.

    """
    def __init__(self, cities_count: int):
        """
        Initialize an Ant object.

        Parameters
        ----------
        cities_count : int
            The total number of cities in the problem.

        Attributes
        ----------
        beta : int
            The beta parameter used in calculating pheromone trails .
        alpha : int
            The alpha parameter used in calculating pheromone trails .
        cities_count : int
            The total number of cities in the problem.
        start_city : int
            The index of the city where the ant starts its tour.
        current_city : int or None
            The index of the current city the ant is visiting during its tour. Initialized as None.
        routes : list
            A list to store all the visited cities by the ant during its tours.
        visited_cities : list
            A list to store the indexes of cities the ant has already visited during its tour.
        cities_to_visit : list
            A list of indexes of cities the ant still needs to visit during its tour.
        """

        self.beta = 5
        self.alpha = 1
        
        self.cities_count = cities_count
        self.start_city = np.random.randint(0,cities_count-1)

        self.current_city = None

        self.routes = []
        self.visited_cities = []
        self.cities_to_visit = [i for i in range(cities_count)]

        self.visit(self.start_city)

    def visit(self, city: int):
        """Method for setting ant's visited city."""

        self.visited_cities.append(city)
        self.cities_to_visit.remove(city)
        self.current_city = city
        
    def total_cost(self):
        """Calculation of total cost of visited cities."""

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
        """Calculation of propabilities for remaining cities_to_visit."""
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
        """
        Decision of an ant based on propabilities
        ...

        Roulette choice.
        """
        probs = self.get_propabilities()
        random_number = np.random.uniform(0,1)
        number = 0
        for i in range(len(probs)):
            number += probs[i]
            if number >= random_number:
                return self.cities_to_visit[i]   
    
    def move(self):
        """Moving ant to a city based on decision."""
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
        """Creates temporary pheromones matrix dictionary. """
        ant_phero = self.create_pher_matrix()
        for i in range(len(self.visited_cities) - 1):
            
            ant_phero[self.visited_cities[i]][self.visited_cities[i+1]] += (1/self.total_cost())
            ant_phero[self.visited_cities[i+1]][self.visited_cities[i]] += (1/self.total_cost())
        ant_phero[self.visited_cities[-1]][self.visited_cities[0]] += (1/self.total_cost())
        ant_phero[self.visited_cities[0]][self.visited_cities[-1]] += (1/self.total_cost())
        cities = self.visited_cities
        return ant_phero


    def reset_to_default(self):
        """Resetting ant to default. """
        self.routes.append(self.visited_cities)
        self.current_city = None
        self.visited_cities = []
        self.cities_to_visit = [i for i in range(10)]
        self.visit(self.start_city)

    def get_current_stats(self):
        """This method prints starting city and cities to visit."""
        print(f'Start city: {self.start_city}')
        print(f'cities_to_visit: {self.cities_to_visit}')

class Population:
    """
    A class to represent a population of ants for an ant colony optimization algorithm.

    ...

    Attributes
    ----------
    evap : float
        The evaporation rate for pheromones.
    ants : list of Ant
        A list of ants in the population.
    population_size : int
        The size of the ant population.
    cities_count : int
        The total number of cities in the problem.

    Methods
    -------
    __init__(population_size: int, cities_count: int)
        Initializes a Population object with the given size and number of cities.

    get_starting_cities()
        Prints the starting cities of all ants in the population.

    move_ants()
        Moves all ants in the population to the next city in their tour.

    get_best_ant() -> tuple
        Returns the best ant in the population and its corresponding tour cost.

    reset()
        Resets the state of all ants in the population to their initial values.
        
    update_pheromone()
        Updates the pheromone levels in the environment based on ant tours.
    """

    def __init__(self, population_size: int, cities_count: int):
        """
        Initializes a Population object.

        Parameters
        ----------
        population_size : int
            The size of the ant population.
        cities_count : int
            The total number of cities in the problem.
        """
        self.evap = 0.5
        self.ants = []
        self.population_size = population_size
        self.cities_count = cities_count
        for specimen in range(population_size):
            self.ants.append(Ant(cities_count))

    def get_starting_cities(self):
        """Prints the starting cities of all ants in the population."""
        for ant in self.ants:
            print(ant.start_city)

    def move_ants(self):
        """Moves all ants in the population to the next city in their tour."""
        for ant in self.ants:
            ant.move()

    def get_best_ant(self) -> tuple:
        """
        Returns the best ant in the population and its corresponding tour cost.

        Returns
        -------
        tuple
            A tuple containing a list representing the tour of the best ant and its tour cost.
        """
        costs = [self.ants[i].total_cost() for i in range(len(self.ants))]
        return self.ants[costs.index(min(costs))].visited_cities, min(costs)

    def reset(self):
        """Resets the state of all ants in the population to their initial values."""
        for ant in self.ants:
            ant.reset_to_default()

    def update_pheromone(self):
        """Updates the pheromone levels in the environment based on ant tours."""
        ln = len(PHEROMONES)
        for i in range(ln):
            for j in range(i + 1, ln):
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
                for j in range(i + 1, ln):

                    if i not in PHEROMONES:
                        PHEROMONES[i] = {}
                    if j not in PHEROMONES:
                        PHEROMONES[j] = {}

                    PHEROMONES[i][j] += temp[i][j]
                    PHEROMONES[j][i] += temp[j][i]



class System:
    """
    A class to represent an Ant Colony System for solving the Traveling Salesman Problem.

    ...

    Attributes
    ----------
    population : Population
        The population of ants.
    n_generations : int
        The number of generations (iterations) for the algorithm.
    best_ant : list
        A list of tuples containing the best ant tour and its corresponding tour cost.

    Methods
    -------
    __init__(population_size: int, cities_count: int, n_generations: int)
        Initializes a System object with a given population size, number of cities, and generations.

    go()
        Executes a single generation of the Ant Colony System.

    run_system()
        Runs the Ant Colony System for the specified number of generations.

    get_results() -> list
        Returns the list of best ants and their corresponding tour costs.

    get_best_ant() -> tuple
        Returns the best ant tour and its corresponding tour cost.

    visualize_path_all()
        Visualizes all ants' routes.

    visualize_path_best()
        Visualizes the best ant's route.

    visualize_path_all_no_best()
        Visualizes all ants' routes without highlighting the best ant's route.

    draw_route_all_best(x, y)
        Draws the routes of all ants, highlighting the best ant's route.

    draw_only_best(x, y)
        Draws only the route of the best ant.

    draw_route_all(x, y)
        Draws the routes of all ants without highlighting the best ant's route.
    """
    def __init__(self, population_size: int, 
                cities_count: int, n_generations: int)->None:
        """
        Initializes a System object.

        Parameters
        ----------
        population_size : int
            The size of the ant population.
        cities_count : int
            The total number of cities in the problem.
        n_generations : int
            The number of generations (iterations) for the algorithm.
        """
        self.population = Population(population_size,cities_count)
        self.n_generations = n_generations
        
        self.best_ant = []
    
    def go(self):
        """Executes a single generation of the Ant Colony System."""
        self.population.move_ants()
        self.population.update_pheromone()
        self.best_ant.append(self.population.get_best_ant())
        self.population.reset()
        
    def run_system(self):
        """Runs the Ant Colony System for the specified number of generations."""

        for i in range(self.n_generations):
            self.go()

        
    def get_results(self):
        """
        Returns the list of best ants and their corresponding tour costs.

        Returns
        -------
        list
            A list of tuples, where each tuple contains a list representing an ant's tour and its tour cost.
        """
        return self.best_ant
    
    def get_best_ant(self):
        """
        Returns the best ant tour and its corresponding tour cost.

        Returns
        -------
        tuple
            A tuple containing a list representing the tour of the best ant and its tour cost.
        """
        results = self.get_results()
        ants = []
        path_cost = []
        for ant,cost in results:
            ants.append(ant)
            path_cost.append(cost)
        
        best_ant = path_cost.index(min(path_cost))
        return ants[best_ant],min(path_cost)
        
    def visualize_path_all(self):
        """Visualizes all ants' routes."""
        x,y=load_file(FILE_PATH)
        self.draw_route_all_best(x,y)
    
    
    def visualize_path_best(self):
        """Visualizes the best ant's route."""
        x,y=load_file(FILE_PATH)
        self.draw_only_best(x,y)
    
    
    def visualize_path_all_no_best(self):
        """Visualizes all ants' routes without highlighting the best ant's route."""
        x,y=load_file(FILE_PATH)
        self.draw_route_all(x,y)
    

    def draw_route_all_best(self,x,y):
        """
        Draws the routes of all ants, highlighting the best ant's route.

        Parameters
        ----------
        x : list
            The x-coordinates of the cities.
        y : list
            The y-coordinates of the cities.
        """
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
        """
        Draws only the route of the best ant.

        Parameters
        ----------
        x : list
            The x-coordinates of the cities.
        y : list
            The y-coordinates of the cities.
        """
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
        plt.savefig('best_route.png')
            
    
    def draw_route_all(self,x,y):
        """
        Draws the routes of all ants without highlighting the best ant's route.

        Parameters
        ----------
        x : list
            The x-coordinates of the cities.
        y : list
            The y-coordinates of the cities.
        """

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

    
    FILE_PATH = 'src/cities_4.txt'
    DIST_MATRIX = prepare_matrix(FILE_PATH)
    PHEROMONES = init_pheromones(DIST_MATRIX)

    system = System(population_size=10,
                    cities_count=10,
                    n_generations=200)
    
    system.run_system()
    
    
    
    system.visualize_path_all()
    system.visualize_path_best()
    system.visualize_path_all_no_best()
    
    
    
    
    
    
    