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
visualize_train = False
visualize_validation = False

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
learner_params['batch_size'] = 64 #Not used during testing
learner_params['training_epochs'] = 20 #Not used during testing
learner_params['seed_val'] = 1234
learner_params['mode'] = "gpu"
learner_params['display_step'] = 1

sail_params = dict()
sail_params['beta0'] = 0         #Initial beta (after iter 0)
sail_params['k']     = 60        #Number of datapoints to collect per environment
sail_params['N']     = 5       #number of SaIL iterations
sail_params['T']     = 1100      #max episode length for training
sail_params['Tv']    = 20000    #episode length for validation/testing
sail_params['m']     = 200      #Number of training envs
sail_params['mv']    = 70       #Number of validation envs


def run_training(train_folder, train_oracle_folder, validation_folder, validation_oracle_folder, model_folder, file_start_num_train, file_start_num_valid, pretrained_model, oracle_file_type):
  global sail_params, env_params, learner_params, lattice, cost_fn, start, goal, visualize_train, visualize_validation
  # results = defaultdict(list)
  env_name = os.path.split(os.path.split(os.path.abspath(train_folder))[0])[1]
  output_file_str = "iter_" + str(sail_params['N']) + "_features_" + str(learner_params['input_size']) + "_num_train_envs_" + str(sail_params['m'])+ "_num_valid_envs_" + str(sail_params['mv'])
  model_folder = os.path.join(os.path.abspath(model_folder), output_file_str)
  if pretrained_model:
    pretrained_model = os.path.abspath(pretrained_model)
  agent = SaILAgent(sail_params, env_params, learner_params, lattice, cost_fn, start, goal)
  env_results = agent.run_training(train_folder, train_oracle_folder, validation_folder, validation_oracle_folder, model_folder, file_start_num_train, file_start_num_valid, pretrained_model, visualize_train, visualize_validation, oracle_file_type)
  json.dump(env_results, open(os.path.join(os.path.abspath("../SaIL/results/xy/"+env_name), output_file_str), 'w'), sort_keys=True)
  print(env_results)
  

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--train_folder', required=True)
  parser.add_argument('--train_oracle_folder', required=True)
  parser.add_argument('--validation_folder', required=True)
  parser.add_argument('--validation_oracle_folder', required=True)
  parser.add_argument('--model_folder', required=True)
  parser.add_argument('--train_file_start_num', type=int)
  parser.add_argument('--validation_file_start_num', type=int)
  parser.add_argument('--pretrained_model', type=str)
  parser.add_argument('--oracle_file_type', type=str)
  
  args = parser.parse_args()
  #Run the benchmark and save results
  run_training(args.train_folder, args.train_oracle_folder, args.validation_folder, args.validation_oracle_folder, args.model_folder, args.train_file_start_num, args.validation_file_start_num, args.pretrained_model, args.oracle_file_type)
  # print(results)