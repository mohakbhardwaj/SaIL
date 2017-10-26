#!/usr/bin/env python
"""An environment that takes as input databases of environments and runs episodes, 
where each episode is a search based planner. It then returns the average number of expansions,
and features (if training)
Author: Mohak Bhardwaj
"""
from __future__ import division #Force float division
from collections import defaultdict
from math import fabs
import matplotlib.pyplot as plt
import numpy as np
import os
import cPickle as pickle
import time
from SaIL.learners.supervised_regression_network import SupervisedRegressionNetwork
from planning_python.data_structures.priority_queue import PriorityQueue
from planning_python.planners.search_based_planner import SearchBasedPlanner
from planning_python.environment_interface.env_2d import Env2D
from planning_python.state_lattices.common_lattice.xy_analytic_lattice import XYAnalyticLattice
from planning_python.state_lattices.common_lattice.xyh_analytic_lattice import XYHAnalyticLattice
from planning_python.cost_functions.cost_function import PathLengthNoAng, DubinsPathLength
from planning_python.heuristic_functions.heuristic_function import EuclideanHeuristicNoAng, ManhattanHeuristicNoAng, DubinsHeuristic
from planning_python.data_structures.planning_problem import PlanningProblem


class StateLatticePlannerEnv(SearchBasedPlanner):
  def __init__(self, env_params, lattice_type, lattice_params, cost_fn, learner_params, start, goal):
    
    self.env_params = env_params
    self.cost = cost_fn
    self.lattice_type = lattice_type
  
    normalizn_terms = dict()
    self.hs = [EuclideanHeuristicNoAng(), ManhattanHeuristicNoAng()]
    
    min_st = np.array((lattice_params['x_lims'][0], lattice_params['y_lims'][0]))
    max_st = np.array((lattice_params['x_lims'][1], lattice_params['y_lims'][1]))
    normalizn_terms['euc_dist_norm'] = self.hs[0].get_heuristic(min_st, max_st)
    normalizn_terms['man_dist_norm'] = self.hs[1].get_heuristic(min_st, max_st)
    normalizn_terms['g_norm'] = (lattice_params['x_lims'][1] - lattice_params['x_lims'][0])*(lattice_params['y_lims'][1] - lattice_params['y_lims'][0])
    
    if lattice_type == "XY":
      self.lattice = XYAnalyticLattice(lattice_params)
      normalizn_terms['max_children'] = self.lattice.children.shape[0]
      normalizn_terms['coord_norm'] = np.array([self.lattice.num_cells[0], self.lattice.num_cells[1]], dtype=np.float)
    elif lattice_type == "XYH":
      self.lattice = XYHAnalyticLattice(lattice_params)
      normalizn_terms['dubins_dist_norm'] = (lattice_params['x_lims'][1] - lattice_params['x_lims'][0])*(lattice_params['y_lims'][1]-lattice_params['y_lims'][0])
      normalizn_terms['max_children'] = len(self.lattice.children[0])
      normalizn_terms['coord_norm'] = np.array([self.lattice.num_cells[0], self.lattice.num_cells[1], self.lattice.num_headings], dtype=np.float)
      self.hs.append(DubinsHeuristic(lattice_params['radius']))

    self.lattice.precalc_costs(self.cost) #Enumerate and cache successors and edge costs(for speed-ups) 
    self.start_state = start
    self.goal_state = goal
    self.start_node = self.lattice.state_to_node(start)
    self.goal_node = self.lattice.state_to_node(goal)
       
    #Data structures for planning
    # self.frontier = [] #Frontier is un-sorted as it is sorted on demand (using heuristic)
    self.oracle_frontier = PriorityQueue() #Frontier sorted according to oracle(for mixing)
    self.visited = {} #Keep track of visited cells
    # self.c_obs = []  #Keep track of collision checks done so far
    self.cost_so_far = defaultdict(lambda: np.inf) #Keep track of cost of path to the node
    self.came_from = {} #Keep track of parent during search
    
    self.learner_params = learner_params
    self.learner_params['start_n'] = self.start_node
    self.learner_params['goal_n'] = self.goal_node
    self.learner_params['normalizn_terms'] = normalizn_terms
    self.learner = SupervisedRegressionNetwork(self.learner_params)


  def initialize(self, train_env_folder=[], train_oracle_folder=[], num_train_envs=0, train_file_start_num=0, validation_env_folder=[], validation_oracle_folder=[], num_validation_envs=0, valid_file_start_num=0, model_folder=[],train_visualize=False, validation_visualize=False):
    """Initialize everything"""
    self.train_env_folder = train_env_folder
    self.train_oracle_folder = train_oracle_folder 
    self.num_train_envs = num_train_envs 
    self.train_visualize = train_visualize
    self.validation_env_folder = validation_env_folder
    self.validation_oracle_folder = validation_oracle_folder
    self.num_validation_envs = num_validation_envs
    self.validation_visualize = validation_visualize
    self.model_folder = os.path.abspath(model_folder)

    self.train_start_file = train_file_start_num
    self.validation_start_file = valid_file_start_num
    self.curr_train_num = - 1
    self.curr_valid_num = - 1
    
    self.curr_oracle = dict()
    self.env = Env2D()
    # #We need to initialize a dummy env before learner (if visualization is needed) as Tensorflow hogs entire GPU otherwise
    if self.train_visualize:
      import matplotlib.pyplot as plt
      file_path = os.path.join(os.path.abspath(self.train_env_folder), str(self.train_start_file)+'.png')
      self.initialize_env_from_file(file_path, 'True')
    if self.validation_visualize:
      import matplotlib.pyplot as plt
      file_path = os.path.join(os.path.abspath(self.validation_env_folder), str(self.validation_start_file)+'.png')
      self.initialize_env_from_file(file_path, "True")
    
    self.learner.initialize()
    self.initialized = True
    print('Initialized planner env')

  def set_mixing_param(self, beta):
    self.beta = beta
  
  def run_episode(self, phase="train", k_tsteps=[], max_expansions=100000):
    assert self.initialized == True, "Planner has not been initialized properly. Please call initialize or reset_problem function before plan function"
    #Clear the features in the learner
    # print k_tsteps
    self.learner.clear_features()
    start_t = time.time()
    
    self.came_from[self.start_node]= (None, None)
    self.cost_so_far[self.start_node] = 0.  #For each node, this is the cost of the shortest path to the start

    self.learner.cost_so_far[self.start_node] = 0
    self.learner.num_invalid_predecessors[self.start_node] = 0
    self.learner.num_invalid_siblings[self.start_node] = 0
    self.learner.depth_so_far[self.start_node] = 1
    self.learner.euc_dist[self.start_node] = self.hs[0].get_heuristic(self.start_state, self.goal_state)
    self.learner.man_dist[self.start_node] = self.hs[1].get_heuristic(self.start_state, self.goal_state)
    
    if self.lattice_type == "XYH":
      self.learner.dubins_dist[self.start_node] = self.hs[2].get_heuristic(self.start_state, self.goal_state)
   

    if phase == "train":
      start_h_val = self.curr_oracle[self.start_node]
      self.oracle_frontier.put(self.start_node, start_h_val)
    
    self.learner.append_to_frontier(self.start_node) #This frontier is just a list

    curr_expansions = 0         #Number of expansions done
    num_rexpansions = 0
    found_goal = False
    path =[]
    path_cost = np.inf

    while len(self.learner.frontier) > 0: 
      #Check 1: Stop search if frontier gets too large
      if curr_expansions >= max_expansions:
        print("Max Expansions Done.")
        break
      # #Check 2: Stop search if open list gets too large
      # if len(self.frontier) > 500000:
      #   print("Timeout.")
      #   break

      #################################################################################################
      #Step 1: With probability beta, we select the oracle and (1-beta) we select the learner, also we collect data if 
      # curr_expansions is in one of the k timesteps
      if phase == "train":
        if curr_expansions in k_tsteps:
          rand_idx = np.random.randint(len(self.learner.frontier))
          n = self.learner.get_node_from_frontier(rand_idx)   #Choose a random action
          self.learner.update_database([self.learner.get_feature_vec([n])], [self.curr_oracle[n]])
        if np.random.random() <= self.beta:
          h, _, curr_node = self.oracle_frontier.get()
        else:
          best_idx = self.learner.get_best_node()
          curr_node = self.learner.get_node_from_frontier(best_idx)
          self.learner.delete_from_frontier(best_idx)
      else:
        best_idx = self.learner.get_best_node()
        curr_node = self.learner.get_node_from_frontier(best_idx)
        self.learner.delete_from_frontier(best_idx)
      #################################################################################################
      if curr_node in self.visited:
        continue
      #Step 3: Add to visited
      self.visited[curr_node] = 1

      #Check 3: Stop search if goal found
      if curr_node == self.goal_node:  
        print "Found goal"
        found_goal = True
        break
      
      #Step 4: If search has not ended, add neighbors of current node to frontier
      neighbors, edge_costs, valid_edges, invalid_edges = self.get_successors(curr_node)
      
      #Update the features of the parent and current node
      n_invalid_edges = len(invalid_edges)
      self.learner.num_invalid_grand_children[self.came_from[curr_node][0]] += n_invalid_edges
      self.learner.num_invalid_children[curr_node] = n_invalid_edges

      if n_invalid_edges > 0:
        nodes = []
        for i_edge in invalid_edges:
          nodes.append(self.lattice.state_to_node(i_edge[1]))
        self.learner.update_cobs(nodes) 
      

      # self.c_obs.append(invalid_edges)
      g = self.cost_so_far[curr_node]
      
      for i, neighbor in enumerate(neighbors):
        new_g = g + edge_costs[i]
        
        if neighbor not in self.visited:
          #Add neighbor to open only if it wasn't in open already (don't need duplicates) [Note: Only do this if ordering in the frontier doesn't matter]
          if neighbor not in self.cost_so_far:
            #Update the oracle frontier only during training (for mixing)
            if phase == "train":
              h_val = self.curr_oracle[neighbor]
              self.oracle_frontier.put(neighbor, h_val) 
            self.learner.append_to_frontier(neighbor)    
          #Keep track of cost of shortest path to neighbor and parent it came from (for features and reconstruct path)  
          if new_g < self.cost_so_far[neighbor]:
            self.came_from[neighbor] = (curr_node, valid_edges[i])
            self.cost_so_far[neighbor] = new_g

            #Update feature dicts
            self.learner.cost_so_far[neighbor] = new_g
            self.learner.num_invalid_predecessors[neighbor] = self.learner.num_invalid_predecessors[curr_node] + n_invalid_edges
            self.learner.num_invalid_siblings[neighbor] = n_invalid_edges
            self.learner.depth_so_far[neighbor] = self.learner.depth_so_far[curr_node] + 1
            self.learner.euc_dist[neighbor] = self.hs[0].get_heuristic(self.lattice.node_to_state(neighbor), self.goal_state)
            self.learner.man_dist[neighbor] = self.hs[1].get_heuristic(self.lattice.node_to_state(neighbor), self.goal_state)
            if self.lattice_type == "XYH":
              self.learner.dubins_dist[self.start_node] = self.hs[2].get_heuristic(self.lattice.node_to_state(neighbor), self.goal_state)
      #Step 6:increment number of expansions
      curr_expansions += 1

    if found_goal:
      path, path_cost = self.reconstruct_path(self.came_from, self.start_node, self.goal_node, self.cost_so_far)
    else:
      print ('Found no solution')
    time_taken = time.time()- start_t
    return path, path_cost, curr_expansions, time_taken, self.came_from, self.cost_so_far#, self.c_obs    #Run planner on current env and return data seetn. Also, update current env to next env

  
  def train_learner(self):
    avg_cost = self.learner.train()
    return avg_cost
  
  def save_learner(self):
    self.learner.save_params(self.model_folder)

  def load_learner(self):
    self.learner.load_params(self.model_folder) 

  def sample_world(self, phase= "train", mode='cycle'):
    if phase == "train":
      self.curr_train_num = (self.curr_train_num+1)%self.num_train_envs
      file_path = os.path.join(os.path.abspath(self.train_env_folder), str(self.curr_train_num + self.train_start_file)+'.png')
      # self.env.reset_plot(self.lattice.node_to_state(self.start_node), self.lattice.node_to_state(self.goal_node))
      self.initialize_env_from_file(file_path, self.train_visualize)
      self.visualize = self.train_visualize
    else:
      self.curr_valid_num = (self.curr_valid_num+1)%self.num_validation_envs
      file_path = os.path.join(os.path.abspath(self.validation_env_folder), str(self.curr_valid_num + self.validation_start_file)+'.png')
      self.initialize_env_from_file(file_path, self.validation_visualize)
      self.visualize = self.validation_visualize

  def compute_oracle(self, phase='train', mode='cycle'):
    if phase == "train":
      file_path = os.path.join(os.path.abspath(self.train_oracle_folder), "oracle_"+str(self.curr_train_num + self.train_start_file)+'.p')
    else:
      file_path = os.path.join(os.path.abspath(self.validation_oracle_folder), "oracle_"+str(self.curr_valid_num + self.validation_start_file)+'.p')
    
    with open(file_path, 'rb') as fh:
      self.curr_oracle = pickle.load(fh) 
   
  def initialize_env_from_file(self, file_path, visualize):
    # env = Env2D()
    self.env.initialize(file_path, self.env_params)
    if visualize:
      if not self.env.plot_initialized:
        self.env.initialize_plot(self.lattice.node_to_state(self.start_node), self.lattice.node_to_state(self.goal_node))
      else:
        self.env.reset_plot(self.lattice.node_to_state(self.start_node), self.lattice.node_to_state(self.goal_node))

    self.initialized = True
    # return env

  def clear_planner(self):
    # self.frontier = []
    self.learner.clear_frontier()
    self.visited = {}
    # self.c_obs = []
    self.cost_so_far = defaultdict(lambda: np.inf)
    self.came_from = {}
    self.oracle_frontier.clear()
    # if self.visualize: plt.close() 

