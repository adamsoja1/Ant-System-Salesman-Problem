import numpy as np
import random



def load_file(path:str)->tuple:
    
    
    with open(path, "r") as file:
        file_contents = file.read()
    
    # Extract x and y substrings
    x_start = file_contents.find("[") + 1
    x_end = file_contents.find("]")
    y_start = file_contents.find("y = [") + len("y = [")
    y_end = file_contents.find("]", y_start)
    
    x_substring = file_contents[x_start:x_end]
    y_substring = file_contents[y_start:y_end]
    
    # Split substrings and convert to lists of floats
    x_list = [float(s) for s in x_substring.split()]
    y_list = [float(s) for s in y_substring.split()]
    
    return x_list,y_list
    

"Function that transforms cords into tuples"
def getTupleCities(x,y):
    cities = []
    for i in range(len(x)):
        city = (x[i],y[i])
        cities.append(city)
    return cities


"Calculate distance between cities according to x,y cords"
def distance(city1,city2):
    "X1-X2, Y1-Y2"
    x_cord = city1[0] - city2[0]
    y_cord = city1[1] - city2[1]
    
    return np.sqrt(x_cord**2+y_cord**2)



def distance_matrix(cities):
    ln = len(cities)
    dist_matrix = {}
    for i in range(ln):
        for j in range(i+1, ln):
            dist = distance(cities[i], cities[j])
            
            if i not in dist_matrix:
                dist_matrix[i] = {}
            if j not in dist_matrix:
                dist_matrix[j] = {}
            
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
            
    return dist_matrix


def prepare_matrix(path:str):
    x,y = load_file(path)
    cities = getTupleCities(x, y)
    matrix = distance_matrix(cities)
    return matrix




def init_pheromones(DIST_MATRIX):
    """Creates matrix of pheromones level between cities (indexes)"""
    pheromone_matrix = {}
    ln = len(DIST_MATRIX)
    for i in range(ln):
        for j in range(i+1, ln):
            dist = 0.3
            
            if i not in pheromone_matrix:
                pheromone_matrix[i] = {}
            if j not in pheromone_matrix:
                pheromone_matrix[j] = {}
            
            pheromone_matrix[i][j] = dist
            pheromone_matrix[j][i] = dist
    return pheromone_matrix
