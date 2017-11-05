#!/usr/bin/env python

"""An agent that implements the SaIL algorithm as described in the CoRL,2017 paper, Learning Heuristic Search via Imitation. 
The agent takes as arguments an environment database, and parameters as described in the paper and runs the SaIL algorithm for a
particular number of iterations and returns the results. 

The classic SaIL agent uses a state lattice based planner as the search function. It can be replaced by RGGs or any other 
graph plan algorithm

Author: Mohak Bhardwaj
Date: 16 October, 2017
"""
import numpy as np
from planning_python.data_structures.planning_problem import PlanningProblem


class SaILAgent():
  def __init__(self, env):
    self.env = env #This env takes care of loading and storing all (train, validation, test) environments + planners
    #unpack learner params and create learner object

  def initialize(sail_params, learner_params):
    """Unpack params, create and initialize planner and learner"""
    #Unpack SaIL params
    self.beta0 = sail_params['beta0']
    self.k     = sail_parmas['k']
    self.N     = sail_params['N']
    self.T     = sail_params['T']
    self.m     = sail_params['m']     #Number of training envs
    #Unpack learner params

  def run_training():
    #Runs the SaIL training loop for N
    loss_per_iter = []
    expansions_per_iter = []
    D = []
    for i in range(self.N):
      curr_beta = (self.beta0)**i #curr_beta = 1 for 0th iteration (expert only), then decayed exponentially based on beta0
      iter_expansions = 0
      self.env.set_learner_policy(self.network) #Set the learner policy for the current iteration
      self.env.set_mixing_param(curr_beta) #Set the mixing parameter for creating mixture policy
      
      for j in range(self.m):
        self.env.sample_world(mode='cycle')      #sample a world from P(world) and sample start,goal from P(s,g) mode='cycle implies we are cycling through a database'
        self.env.compute_oracle()                #computes oracle for corresponding world
        k_tsteps =  np.random.randint(0, self.T) #Sample k timesteps where data will be collected
        num_expansions, time_taken, data = self.env.run_episode_train(k_tsteps, self.T)
        D.append(data) #Aggregate data                                        
        iter_expansions += num_expansions
      
      iter_loss = self.learner.train(D)
      loss_per_iter.append(iter_loss)
      expansions_per_iter.append(iter_expansions) 
    
    return loss_per_iter, expansions_per_iter

  def run_validation(num_envs):
    """Runs validation on validation environments and returns average loss"""
    avg_valid_loss = 0
    avg_expansions = 0
    for i in range(self.num_envs): 
    return None

  def run_test(num_envs):
    """ Runs agent on test environments and returns average loss"""

    return 0
  
  def run_episode_train(self):
    return None
  def run_episode_test(self):
    return None