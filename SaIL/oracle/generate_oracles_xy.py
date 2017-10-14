#!/usr/bin/env python
"""Given as input a list of database folders , we calculate the clairvoyant oracle i.e number of expansions to the goal for each node.
The oracle is then stored in a json file 

Author: Mohak Bhardwaj
Date: October 6, 2017
"""
import argparse
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, "../../../planning_python")
import time
from planning_python.environment_interface.env_2d import Env2D
from planning_python.state_lattices.common_lattice.xy_analytic_lattice import XYAnalyticLattice
from planning_python.cost_functions.cost_function import UnitCost
from planning_python.heuristic_functions.heuristic_function import EuclideanHeuristicNoAng, ManhattanHeuristicNoAng
from planning_python.data_structures.planning_problem import PlanningProblem
from planning_python.planners.value_iteration import ValueIteration

x_lims = [0, 200]
y_lims = [0, 200]

env_params = {'x_lims': x_lims, 'y_lims': y_lims}
lattice_params = {'x_lims': x_lims, 'y_lims': y_lims, 'resolution': [1, 1], 'origin': (0, 0), 'rotation': 0, 'connectivity': 'eight_connected', 'path_resolution': 1}
cost_fn = UnitCost() #We want to calculate number of expansions only
heuristic_fn = EuclideanHeuristicNoAng()
lattice = XYAnalyticLattice(lattice_params)
planner = ValueIteration()
start_n = lattice.state_to_node((0,0))
goal_n = lattice.state_to_node((200, 200))
prob_params = {'heuristic_weight': 0.0} #We want to run Djikstra
prob = PlanningProblem(prob_params)

def generate_oracles(database_folders=[], num_envs=1, file_start_num=0):
  global env_params, lattice_params, cost_fn, heuristic_fn, lattice, planner, start_n, goal_n, prob
     
  e = Env2D()
  print('Generating Oracles')

  for folder in database_folders:
    env_name = os.path.split(os.path.split(os.path.abspath(folder))[0])[1]
    env_folder = os.path.split(os.path.abspath(folder))[1]
    for i in range(num_envs):
      curr_env_file = os.path.join(os.path.abspath(folder), str(file_start_num + i)+'.png')
      print(curr_env_file)
      e.initialize(curr_env_file, env_params)
      prob.initialize(e, lattice, cost_fn, heuristic_fn, start_n, goal_n, visualize=False)
      planner.initialize(prob) 
      start_time = time.time()
      path, path_cost, num_expansions, came_from, cost_so_far, c_obs = planner.plan()
      plan_time = time.time() - start_time
      print('Time taken', plan_time)
      output_file = "oracle_" + str(file_start_num + i) + ".p"
      pickle.dump(cost_so_far, open(os.path.join(os.path.abspath("./saved_oracles/xy/"+env_name+"/"+env_folder), output_file), 'wb')) 


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--database_folders', nargs='+', required=True)
  parser.add_argument('--num_envs', type=int)
  parser.add_argument('--file_start_num', type=int)
  args = parser.parse_args()
  #Run the benchmark and save results
  generate_oracles(args.database_folders, args.num_envs, args.file_start_num)
  # print(results)
