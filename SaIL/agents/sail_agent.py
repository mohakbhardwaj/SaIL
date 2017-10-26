#!/usr/bin/env python

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
from planning_python.data_structures.planning_problem import PlanningProblem


class SaILAgent():
  def __init__(self, sail_params):
    self.beta0 = sail_params['beta0'] #Initial beta (after iter 0)
    self.k     = sail_params['k']     #Number of datapoitns to collect per environment
    self.N     = sail_params['N']     #number of SaIL iterations
    self.T     = sail_params['t']    #episode length for training
    self.Tv    = sail_params['tv']#sail_params['Tv']    #episode length for validation/testing
    self.m     = sail_params['m']     #Number of training envs
    self.mv    = sail_params['mv']    #Number of validation envs

  def initialize(self, env):
    # assert env.initialize, "The underlying environment has not been initialized!!"
    self.env = env #This env takes care of loading and storing all (train, validation, test) environments + learner
    print('Initialized Agent')

  def run_training(self):
    #Runs the SaIL training loop for N
    
    results = dict()
    results['loss_per_iter'] = []
    results['expansions_per_iter'] = []
    results['avg_time_per_iter'] = []
    results['avg_path_cost_per_iter'] = []
    results['num_solved_per_iter'] = []

    min_expansions_so_far = np.inf #We will save the best performing learner

    for i in range(self.N):

      curr_beta = (self.beta0)**i #curr_beta = 1 for 0th iteration (expert only), then decayed exponentially based on beta0
      iter_expansions = 0
      self.env.set_mixing_param(curr_beta) #Set the mixing parameter for creating mixture policy
      
      for j in range(self.m):
        
        self.env.sample_world(phase='train', mode='cycle')                  #sample a world from P(world) and sample start,goal from P(s,g) mode='cycle implies we are cycling through a database'
        self.env.compute_oracle(phase='train', mode='cycle')                #computes oracle for corresponding world
        
        k_tsteps =  np.random.randint(0, self.T, size=self.k)               #Sample k timesteps where data will be collected
        
        self.env.clear_planner()
    
        path, path_cost, num_expansions, time_taken, _, _ = self.env.run_episode(phase='train', k_tsteps=k_tsteps, max_expansions= self.T) #runs an episode and aggregates data
              
      iter_loss = self.env.train_learner() #Regression loss in this iteration of training
      avg_path_cost, iter_expansions, avg_time_taken, num_solved = self.run_validation() #True task loss on validation set
      
      print('Iter %d. Beta = %f; Task Loss = %f; Avg Time Taken = %f'%(i, curr_beta, iter_expansions, avg_time_taken))
      if iter_expansions <= min_expansions_so_far:  
        print('Saving Learner. Task Loss: %f. Best so far: %f'%(iter_expansions, min_expansions_so_far))
        self.env.save_learner()
        min_expansions_so_far = iter_expansions

      results['loss_per_iter'].append(iter_loss)
      results['avg_path_cost_per_iter'].append(avg_path_cost)
      results['expansions_per_iter'].append(iter_expansions)
      results['avg_time_per_iter'].append(avg_time_taken) 
      results['num_solved_per_iter'].append(num_solved)

    return results

  def run_validation(self):
    """Runs validation on validation environments and returns average loss"""
    # self.valid_env.set_learner_weights(learner_weights)
    print('Running validation')
    avg_path_cost = 0
    avg_expansions = 0
    avg_time_taken = 0
    num_solved = 0
    for j in range(self.mv):
      print('Validation Env %d'%(j))
      self.env.sample_world(phase='validation', mode='cycle')
      self.env.compute_oracle(phase='validation', mode='cycle')
      self.env.clear_planner()
      path, path_cost, num_expansions, time_taken, _, _ = self.env.run_episode(phase='validation', max_expansions=self.Tv) 
      
      # print num_expansions, time_taken
      if path_cost < np.inf:
        num_solved += 1
      avg_path_cost += path_cost
      avg_expansions += num_expansions
      avg_time_taken += time_taken
    
    print('Done validation')
    avg_expansions /= self.mv
    avg_path_cost /= self.mv
    avg_time_taken /= self.mv
    
    return avg_path_cost, avg_expansions, avg_time_taken, num_solved

  def run_test(self):
    test_results = dict()
    self.env.load_learner()
    avg_path_cost, avg_expansions, avg_time_taken, num_solved = self.run_validation()
    test_results['avg_path_cost'] = avg_path_cost
    test_results['avg_expansions'] = avg_expansions
    test_results['avg_time_taken'] = avg_time_taken
    test_results['num_solved'] = num_solved
    return test_results

     
