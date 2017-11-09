

"""An agent that implements the SaIL algorithm as described in the CoRL,2017 paper, Learning Heuristic Search via Imitation. 
The agent takes as arguments an environment database, and parameters as described in the paper and runs the SaIL algorithm for a
particular number of iterations and returns the results. 

The classic SaIL agent uses a state lattice based planner as the search function. It can be replaced by RGGs or any other 
graph plan algorithm

Author: Mohak Bhardwaj
Date: 16 October, 2017
"""
from __future__ import division
import numpy as np
import os
from planning_python.data_structures import PlanningProblem
from planning_python.environment_interface import Env2D
from SaIL.planners import TrainPlanner, TestPlanner
from SaIL.learners import SupervisedRegressionNetwork
from SaIL.oracle import Oracle


class SaILAgent():
  def __init__(self, sail_params, env_params, learner_params, lattice, cost_fn, start, goal):
    self.beta0 = sail_params['beta0'] #Initial beta (after iter 0)
    self.k     = sail_params['k']     #Number of datapoitns to collect per environment
    self.N     = sail_params['N']     #number of SaIL iterations
    self.T     = sail_params['T']     #episode length for training
    self.Tv    = sail_params['Tv']    #sail_params['Tv']    #episode length for validation/testing
    self.m     = sail_params['m']     #Number of training envs
    self.mv    = sail_params['mv']    #Number of validation envs
    self.env_params = env_params

    
    #planning related definitions
    self.e = Env2D()
    self.oracle = Oracle()
    self.cost_fn = cost_fn
    self.heuristic_fn = SupervisedRegressionNetwork(learner_params)
    self.lattice = lattice
    self.lattice.precalc_costs(self.cost_fn)
    self.train_planner = TrainPlanner()
    self.test_planner = TestPlanner()
    prob_params = {'heuristic_weight':1.0} #doesn't matter as we are greedy
    self.prob = PlanningProblem(prob_params)
    self.start = start
    self.goal = goal
    self.start_n = self.lattice.state_to_node(self.start)
    self.goal_n = self.lattice.state_to_node(self.goal)

  def run_training(self, train_folder, train_oracle_folder, validation_folder, validation_oracle_folder, model_folder, file_start_num_train, file_start_num_valid,
                                                                                                        pretrained_model, visualize_train, visualize_validation, oracle_file_type="json"):
    #Runs the SaIL training loop for N
    
    results = dict()
    results['train_loss_per_iter'] = []
    results['validation_loss_per_iter'] = []
    results['avg_expansions_per_iter'] = []
    results['avg_time_per_iter'] = []
    results['avg_path_cost_per_iter'] = []
    results['num_solved_per_iter'] = []
    results['dataset_size_per_iter'] = []
    agg_dataset = []

    min_expansions_so_far = np.inf #We will save the best performing learner
    env_name = os.path.split(os.path.split(os.path.abspath(train_folder))[0])[1]
    env_folder = os.path.split(os.path.abspath(train_folder))[1]
    self.heuristic_fn.initialize()
    if pretrained_model:
      print('Found pretrained model')
      self.heuristic_fn.load_params(pretrained_model)
    for i in range(self.N):
      # curr_beta = (self.beta0)**i #curr_beta = 1 for 0th iteration (expert only), then decayed exponentially based on beta0
      curr_beta = 0
      iter_expansions = 0
    

      for j in range(self.m):
        curr_env_file = os.path.join(os.path.abspath(train_folder), str(file_start_num_train + j)+'.png')
        curr_oracle_file = os.path.join(os.path.abspath(train_oracle_folder), "oracle_" + str(file_start_num_train + j)+'.' + oracle_file_type)
        print(curr_oracle_file)
        self.e.initialize(curr_env_file, self.env_params)
        # self.e.calculate_distance_transform()
        self.oracle.initialize(curr_oracle_file) 
        self.prob.initialize(self.e, self.lattice, self.cost_fn, self.heuristic_fn, self.start_n, self.goal_n, visualize= visualize_train) 
        self.train_planner.initialize(self.prob)
        try:
          path, path_cost, curr_expansions, plan_time, _, _, _, dataset = self.train_planner.plan(self.oracle, curr_beta, self.k, self.T)
          agg_dataset += dataset #add data to meta dataset
        except ValueError:
          continue
        self.train_planner.clear_planner()
        self.e.clear()
        self.oracle.clear()
        print('[Training Iter: %d, Environment Number: %d, results]: Path Cost %f, Number of Expansions %f, Planning Time %f'%(i, j, path_cost, curr_expansions, plan_time))
      
      results['dataset_size_per_iter'].append(len(agg_dataset))
      avg_loss_train = self.heuristic_fn.train(agg_dataset) #Regression loss in this iteration of training
      avg_path_cost, iter_expansions, avg_time_taken, num_solved, avg_loss_valid = self.run_validation(validation_folder, validation_oracle_folder, file_start_num_valid, visualize_validation, oracle_file_type) #True task loss on validation set
      
      print('Iter %d. Beta = %f; Task Loss = %f; Avg Time Taken = %f'%(i, curr_beta, iter_expansions, avg_time_taken))
      if iter_expansions <= min_expansions_so_far:  
        print('Saving Current Policy. Iteration %d. Task Loss: %f. Best so far: %f'%(i, iter_expansions, min_expansions_so_far))
        self.heuristic_fn.save_params(model_folder)
        min_expansions_so_far = iter_expansions

      results['train_loss_per_iter'].append(avg_loss_train)
      results['validation_loss_per_iter'].append(avg_loss_valid)
      results['avg_path_cost_per_iter'].append(avg_path_cost)
      results['avg_expansions_per_iter'].append(iter_expansions)
      results['avg_time_per_iter'].append(avg_time_taken) 
      results['num_solved_per_iter'].append(num_solved)

    return results

  def run_validation(self, validation_folder, validation_oracle_folder, file_start_num_valid, visualize_validation, oracle_file_type="json"):
    """Runs validation on validation environments and returns average loss"""
    print('Running validation')
    avg_path_cost = 0
    avg_expansions = 0
    avg_time_taken = 0
    num_solved = 0
    avg_loss = 0
    num_unsovable = 0
    for j in range(self.mv):
      curr_env_file = os.path.join(os.path.abspath(validation_folder), str(file_start_num_valid+ j)+'.png')
      curr_oracle_file = os.path.join(os.path.abspath(validation_oracle_folder), "oracle_" + str(file_start_num_valid + j)+ '.' + oracle_file_type)
      self.e.initialize(curr_env_file, self.env_params)
      # self.e.calculate_distance_transform()
      self.oracle.initialize(curr_oracle_file)
      self.prob.initialize(self.e, self.lattice, self.cost_fn, self.heuristic_fn, self.start_n, self.goal_n, visualize= visualize_validation)
      self.test_planner.initialize(self.prob)
      try:
        path, path_cost, curr_expansions, plan_time, came_from, cost_so_far, c_obs, avg_episode_loss= self.test_planner.plan(self.oracle, self.Tv)
        print('[Validation Environment Number: %d, results]: Path Cost %f, Number of Expansions %f, Planning Time %f'%(j, path_cost, curr_expansions, plan_time))
        if path_cost < np.inf:
          num_solved += 1
        avg_path_cost += path_cost
        avg_expansions += curr_expansions
        avg_time_taken += plan_time
        avg_loss += avg_episode_loss
      except ValueError:
        num_unsovable += 1
        continue          
      self.test_planner.clear_planner()
      self.e.clear()
      self.oracle.clear()      
    print('Done validation')
    legit_envs = self.mv - num_unsovable
    avg_expansions /= self.mv
    avg_path_cost /= self.mv
    avg_time_taken /= self.mv
    avg_loss /= self.mv
    
    return avg_path_cost, avg_expansions, avg_time_taken, num_solved, avg_loss

  def run_test(self, test_folder, test_oracle_folder, file_start_num_test, model_file, visualize_test, oracle_file_type='json'):
    test_results = dict()
    self.heuristic_fn.initialize()
    self.heuristic_fn.load_params(model_file)
    avg_path_cost, avg_expansions, avg_time_taken, num_solved, avg_loss = self.run_validation(test_folder, test_oracle_folder, file_start_num_test, visualize_test, oracle_file_type)
    test_results['avg_path_cost'] = avg_path_cost
    test_results['avg_expansions'] = avg_expansions
    test_results['avg_time_taken'] = avg_time_taken
    test_results['num_solved'] = num_solved
    test_results['avg_loss'] = avg_loss
    return test_results
