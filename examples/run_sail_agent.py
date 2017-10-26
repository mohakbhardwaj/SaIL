#!/usr/bin/env python
"""Given a database of training, and validation environments, this script runs 
SaIL training procedure and reports the results on validation""" 


import argparse
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, "../../planning_python")
sys.path.insert(0, "..")
import numpy as np
import time
from planning_python.environment_interface.env_2d import Env2D
from planning_python.state_lattices.common_lattice.xy_analytic_lattice import XYAnalyticLattice
from planning_python.cost_functions.cost_function import PathLengthNoAng
from planning_python.heuristic_functions.heuristic_function import EuclideanHeuristicNoAng, ManhattanHeuristicNoAng
from planning_python.data_structures.planning_problem import PlanningProblem
from planning_python.planners.astar import Astar
from SaIL.agents.sail_agent import SaILAgent
from SaIL.utils.plot_utils import plot_results
from SaIL.envs.state_lattice_planner_env import StateLatticePlannerEnv

#Right at the top, we define all the parameter we will need
x_lims = [0, 200]
y_lims = [0, 200]
start  = np.array([0, 0])
goal   = np.array([199, 199])
visualize_train = False
visualize_validation = True
seed_val = 0
lattice_type = "XY"

env_params = {'x_lims': x_lims, 'y_lims': y_lims}
lattice_params = dict()
lattice_params['x_lims']          = x_lims   # Useful to calculate number of cells in lattice 
lattice_params['y_lims']          = y_lims   # Useful to calculate number of cells in lattice
lattice_params['resolution']      = [1, 1]   # Useful to calculate number of cells in lattice + conversion from discrete to continuous space and vice-versa
lattice_params['origin']          = start    # Used for conversion from discrete to contin uous and vice-versa. 
lattice_params['rotation']        = 0        # Can rotate lattice with respect to world
lattice_params['connectivity']    = 'eight_connected' #Lattice connectivity (can be four or eight connected for xylattice)
lattice_params['path_resolution'] = 1         #Resolution for defining edges and doing collision checking (in meters) 


learner_params = dict()
learner_params['output_size'] = 1            #Since we are doing regression
learner_params['learning_rate'] = 0.001
learner_params['batch_size'] = 64
learner_params['training_epochs'] = 70
learner_params['graph_type'] = lattice_type
learner_params['seed_val'] = seed_val
learner_params['mode'] = "cpu"

sail_params = dict()
sail_params['beta0'] = 0.7      #Initial beta (after iter 0)
sail_params['k']     = 150      #Number of datapoitns to collect per environment
sail_params['N']     = 15       #number of SaIL iterations
sail_params['t']     = 1100     #max episode length for training
sail_params['tv']    = 20000    #episode length for validation/testing
sail_params['m']     = 250      #Number of training envs
sail_params['mv']    = 50       #Number of validation envs

cost_fn = PathLengthNoAng()    #We will evaluate paths based on their length. (Note: A policy is evaluated on number of expansions. This is for emperical reasons only)


def run_sail_training(train_folder, train_oracle_folder, train_start_num, valid_folder, valid_oracle_folder, valid_start_num, model_folder):
  global env_params, lattice_type, lattice_params, learner_params, sail_params, start, goal, cost_fn, visualize_train, visualize_validation
  print('SaIL training begin')
  num_train_envs = sail_params['m']
  num_valid_envs = sail_params['mv']

  train_env = StateLatticePlannerEnv(env_params, lattice_type, lattice_params, cost_fn, learner_params, start, goal)
  train_env.initialize(train_folder, train_oracle_folder, num_train_envs, train_start_num, valid_folder, valid_oracle_folder, num_valid_envs, valid_start_num, model_folder, visualize_train, visualize_validation)
  agent = SaILAgent(sail_params)
  agent.initialize(train_env)
  
  train_results = agent.run_training()  
  return train_results

def run_sail_testing(test_folder, test_oracle_folder, test_start_num, model_folder):
  global env_params, lattice_type, lattice_params, learner_params, sail_params, start, goal, cost_fn, visualize_train, visualize_validation
  print('SaIL testing begin')
  num_test_envs = sail_params['mv']
  
  test_env = StateLatticePlannerEnv(env_params, lattice_type, lattice_params, cost_fn, learner_params, start, goal)
  test_env.initialize(validation_env_folder = test_folder, validation_oracle_folder = test_oracle_folder, num_validation_envs = num_test_envs, valid_file_start_num = test_start_num, model_folder = model_folder, validation_visualize = visualize_validation)
  agent = SaILAgent(sail_params)
  agent.initialize(test_env)
  test_results = agent.run_test()
  return test_results



parser = argparse.ArgumentParser()
parser.add_argument('--train_folder', type=str)
parser.add_argument('--valid_folder', type=str)
parser.add_argument('--train_oracle_folder', type=str)
parser.add_argument('--valid_oracle_folder', type=str)
parser.add_argument('--train_file_start_num', type=int, default=0)
parser.add_argument('--valid_file_start_num', type=int, default=800)
parser.add_argument('--model_folder', type=str)
parser.add_argument('--results_folder', type=str)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#Run the benchmark and save results

if not args.test:
  train_results = run_sail_training(args.train_folder, args.train_oracle_folder, args.train_file_start_num, args.valid_folder, args.valid_oracle_folder, args.valid_file_start_num, args.model_folder)
  print train_results  
  results_file = os.path.join(os.path.abspath(args.results_folder), 'train_results_' + str(sail_params['m']) + '_' + str(sail_params['mv']) + '.json')
  #Write results to file
  with open(results_file, 'w') as fp:
    json.dump(train_results, fp)
else:
  test_results = run_sail_testing(args.valid_folder, args.valid_oracle_folder, args.valid_file_start_num, args.model_folder)
  print test_results
  results_file = os.path.join(os.path.abspath(args.results_folder), 'test_results_' + str(sail_params['mv']) + '.json')
  #Write results to file
  print('Saving test results in file')
  with open(results_file, 'w') as fp:
    json.dump(test_results, fp)