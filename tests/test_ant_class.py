import pytest
import numpy as np

import sys
sys.path.append("src")

from src.ants_travel import Ant
from src.distance_matrix import  prepare_matrix,init_pheromones


def test_ant_total_cost():
    DIST_MATRIX = prepare_matrix('src/cities_4.txt')
    
    PHEROMONES = init_pheromones(DIST_MATRIX)
    
    expected_cost =  43.629476033
    ant = Ant(10)
    ant.visited_cities = [0,5,1,9,8,4,7,6,2,3]
    actual = ant.total_cost()
    assert expected_cost == pytest.approx(actual), f"Wrong cost calculation, actual {actual}, expected {expected_cost}"    
    
    
def test_sum_of_propabilities():
    DIST_MATRIX = prepare_matrix('src/cities_4.txt')
    
    PHEROMONES = init_pheromones(DIST_MATRIX)
    
    ant = Ant(10)
    expected = np.sum(ant.get_propabilities())
    actual = 1
    assert expected == pytest.approx(actual), "Wrong propabilities calculation, actual {0}, expected {0}".format(actual,expected)  
    
    
def test_reset_to_default():
    DIST_MATRIX = prepare_matrix('src/cities_4.txt')
    
    PHEROMONES = init_pheromones(DIST_MATRIX)
    
    ant = Ant(10)
    ant.move()
    ant.reset_to_default()
    
    len_cities_to_visit = 9
    current_city = ant.start_city
    expected_visited = 1
    cities_to_visit_len = 9
    
    assert len_cities_to_visit == len(ant.cities_to_visit), "Wrong lengths calculation of cities to visit, actual {0}, expected {0}".format(len_cities_to_visit,len(ant.cities_to_visit))  
    assert current_city == ant.current_city
    assert expected_visited == len(ant.visited_cities), "Wrong number of visited cities!"
    
    
    
    
    
    
def test_start_city():
    DIST_MATRIX = prepare_matrix('src/cities_4.txt')
    
    PHEROMONES = init_pheromones(DIST_MATRIX)
    
    ant = Ant(10)
    expected  = ant.start_city  
    ant.move()
    ant.reset_to_default()
    
    assert expected == ant.start_city,"Cities do not match after resseting an ant"
    
    
def test_move_ant():
    DIST_MATRIX = prepare_matrix('src/cities_4.txt')
    
    PHEROMONES = init_pheromones(DIST_MATRIX)
    ant = Ant(10)

    cities_to_visit_before = len(ant.cities_to_visit)

    ant.move()

    cities_to_visit_after = len(ant.cities_to_visit)

    assert cities_to_visit_before != cities_to_visit_after, "Ant doesnt move corectly!"
    
    
    
    
    
    
    
    
    
    
    
