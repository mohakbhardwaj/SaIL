#!/usr/bin/env python
"""Loads a model from a file and runs it on a validation/test set of files

Author: Mohak Bhardwaj
Date: October 6, 2017
"""
from collections import defaultdict
import json
import os
import pprint
import sys
sys.path.insert(0, "../../planning_python")
sys.path.insert(0, "..")
import matplotlib.pyplot as plt
import numpy as np
from SaIL.agents import SaILAgent
from planning_python.cost_functions import PathLengthNoAng
from planning_python.state_lattices.common_lattice.xy_analytic_lattice import XYAnalyticLattice


x_lims = [0, 201]
y_lims = [0, 201]
start = (0, 0)
goal = (200, 200)

env_params = {'x_lims': x_lims, 'y_lims': y_lims}

lattice_params = dict()
lattice_params['x_lims']          = x_lims   # Useful to calculate number of cells in lattice 
lattice_params['y_lims']          = y_lims   # Useful to calculate number of cells in lattice
lattice_params['resolution']      = [1, 1]   # Useful to calculate number of cells in lattice + conversion from discrete to continuous space and vice-versa
lattice_params['origin']          = start    # Used for conversion from discrete to continuous and vice-versa. 
lattice_params['rotation']        = 0        # Can rotate lattice with respect to world
lattice_params['connectivity']    = 'eight_connected' #Lattice connectivity (can be four or eight connected for xylattice)
lattice_params['path_resolution'] = 1         #Resolution for defining edges and doing collision checking (in meters)
lattice = XYAnalyticLattice(lattice_params)
cost_fn = PathLengthNoAng()

learner_params = dict()
learner_params['output_size'] = 1 #regression
learner_params['input_size'] = 17 #number of features
learner_params['learning_rate'] = 0.001 #not used during testing
learner_params['batch_size'] = 64 #Not used during test
learner_params['training_epochs'] = 20 #Not used during testing
learner_params['seed_val'] = 1234
learner_params['mode'] = "gpu"
learner_params['display_step'] = 5


sail_params = dict()
sail_params['beta0'] = 0        #Initial beta (after iter 0)
sail_params['k']     = 60       #Number of datapoitns to collect per environment
sail_params['N']     = 5        #number of SaIL iterations
sail_params['T']     = 2000     #max episode length for training
sail_params['Tv']    = 20000    #episode length for validation/testing
sail_params['m']     = 200      #Number of training envs
sail_params['mv']    = 70       #Number of validation envs


def run_benchmark(test_folders, test_oracle_folders, model_files, result_folders, num_envs, test_file_start_num, visualize=False, oracle_file_type="json"):
  global sail_params, env_params, learner_params, lattice, cost_fn, start, goal
  pp = pprint.PrettyPrinter()
  for (i,folder) in enumerate(test_folders):
    agent = SaILAgent(sail_params, env_params, learner_params, lattice, cost_fn, start, goal)
    env_results = agent.run_test(folder, test_oracle_folders[i], test_file_start_num, model_files[i], visualize, oracle_file_type)
    output_file = "test_" + "iter_" + str(sail_params['N']) + "_features_" + str(learner_params['input_size']) + "_num_test_envs_" + str(sail_params['mv'])
    pp.pprint(env_results)
    json.dump(env_results, open(os.path.join(os.path.abspath(result_folders[i]), output_file), 'w'), sort_keys=True)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--test_folders', nargs='+', required=True)
  parser.add_argument('--test_oracle_folders', nargs='+', required=True)
  parser.add_argument('--model_files', nargs='+', required=True)
  parser.add_argument('--result_folders', nargs='+', required=True)
  parser.add_argument('--num_envs', type=int)
  parser.add_argument('--test_file_start_num', type=int)
  parser.add_argument('--visualize', action='store_true')
  parser.add_argument('--oracle_file_type', type=str)
  args = parser.parse_args()
  run_benchmark(args.test_folders, args.test_oracle_folders, args.model_files, args.result_folders, args.num_envs, args.test_file_start_num, args.visualize, args.oracle_file_type)
