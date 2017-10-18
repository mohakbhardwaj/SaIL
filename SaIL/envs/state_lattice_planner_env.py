#!/usr/bin/env python
"""An environment that takes as input databases of environments and runs episodes, 
where each episode is a search based planner. It then returns the average number of expansions,
and features (if training)
Author: Mohak Bhardwaj
"""
from collections import defaultdict
import numpy as np
import os
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
  def __init__(self, env_params, lattice_type, lattice_params, cost_fn, learner_params):
    
    self.env_params = env_params
    self.cost_fn = cost_fn
    self.lattice_type = lattice_type
    if lattice_type == "XY":
      self.lattice = XYAnalyticLattice(lattice_params)
      self.start_n = self.lattice.state_to_node((lattice_params['x_lims'][0], lattice_params['y_lims'][0]))
      self.goal_n = self.lattice.state_to_node((lattice_params['x_lims'][1]-1, lattice_params['y_lims'][0]-1))

    
    elif lattice_type == "XYH":
      self.lattice = XYHAnalyticLattice(lattice_params)
      self.start_n = self.lattice.state_to_node((lattice_params['x_lims'][0], lattice_params['y_lims'][0], 0))
      self.goal_n = self.lattice.state_to_node((lattice_params['x_lims'][1]-1, lattice_params['y_lims'][0]-1, 0))  
    
    self.lattice.precalc_costs(self.cost_fn) #Enumerate and cache successors and edge costs 
    
    self.learner_policy = None #This will be set prior to running a polciy using set_learner_policy      
        
    #Data structures for planning
    self.frontier = [] #Frontier is un-sorted as it is sorted on demand (using heuristic)
    self.oracle_frontier = PriorityQueue() #Frontier sorted according to oracle(for mixing)
    self.visited = {} #Keep track of visited cells
    self.c_obs = []  #Keep track of collision checks done so far
    self.cost_so_far = defaultdict(lambda: np.inf) #Keep track of cost of path to the node
    self.came_from = {} #Keep track of parent during search

    self.learner = SupervisedRegressionNetwork(learner_params) #learner is a part of the environment

  def initialize(self, env_folder, oracle_folder, num_envs, file_start_num, phase='train', visualize=False):
    """Initialize everything"""
    self.env_folder = env_folder 
    self.oracle_folder = oracle_folder 
    self.num_envs = num_envs  
    self.phase = phase
    self.visualize = visualize
    self.curr_env_num = file_start_num - 1

      
  def set_mixing_param(self, beta):
    self.beta = beta
  
  def run_episode(k_tsteps=None, max_expansions=1000000):
    assert self.initialized == True, "Planner has not been initialized properly. Please call initialize or reset_problem function before plan function"
    
    start_t = time.time()
    data = [] #Dataset that will be filled during training   
    
    self.came_from[self.start_n]= (None, None)
    self.cost_so_far[self.start_n] = 0.  #For each node, this is the cost of the shortest path to the start

    
    self.num_invalid_predecessors[start] = 0
    self.num_invalid_siblings[start] = 0
    self.depth_so_far[start] = 0
    
    if self.phase == "train":
      start_h_val = self.oracle[self.start_n]
      self.oracle_frontier.put(self.start_n, start_h_val)
    
    self.frontier.append(self.start_n) #This frontier is just a list

    curr_expansions = 0         #Number of expansions done
    num_rexpansions = 0
    found_goal = False
    path =[]
    path_cost = np.inf

    while len(self.frontier) > 0: 
      #Check 1: Stop search if frontier gets too large
      if curr_expansions >= max_expansions:
        print("Max Expansions Done.")
        break
      #Check 2: Stop search if open list gets too large
      if len(self.frontier) > 500000:
        print("Timeout.")
        break

      #################################################################################################
      #Step 1: With probability beta, we select the oracle and (1-beta) we select the learner, also we collect data if 
      # curr_expansions is in one of the k timesteps
      if phase == "train":
        if curr_expansions in k_tsteps:
          rand_idx = np.random.randint(len(self.frontier))
          n = self.frontier[rand_idx]   #Choose a random action
          data.append(self.get_feature_vec[n], self.curr_oracle[n]) #Query oracle for Q-value of that action and append to dataset
        if np.random.random() <= self.beta:
          h, curr_node = self.oracle_frontier.get()
        else
          curr_node = self.get_best_node()
      else:
        curr_node = self.get_best_node()
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
      self.num_invalid_grand_children[self.came_from[curr_node][0]] += n_invalid_edges
      self.num_invalid_children[curr_node] = n_invalid_edges

      #Step 5: Update c_obs with collision checks performed
      self.c_obs.append(invalid_edges)
      g = self.cost_so_far[curr_node]
      
      for i, neighbor in enumerate(neighbors):
        new_g = g + edge_costs[i]
        
        if neighbor not in self.visited
          #Add neighbor to open only if it wasn't in open already (don't need duplicates) [Note: Only do this if ordering in the frontier doesn't matter]
          if neighbor not in self.cost_so_far:
            #Update the oracle frontier only during training (for mixing)
            if self.phase == "train":
              h_val = self.curr_oracle[neighbor]
              self.oracle_frontier.put(neighbor, h_val)
            self.frontier.append(neighbor)    
          #Keep track of cost of shortest path to neighbor and parent it came from (for features and reconstruct path)  
          if new_g < self.cost_so_far[neighbor]:
            self.came_from[neighbor] = (curr_node, valid_edges[i])
            self.cost_so_far[neighbor] = new_g

            #Update feature dicts
            self.learner.cost_so_far[neighbor] = new_g
            self.learner.num_invalid_predecessors[neighbor] = self.num_invalid_predecessors[curr_node] + n_invalid_edges
            self.learner.num_invalid_siblings[neighbor] = n_invalid_edges
            self.learner.depth_so_far[neighbor] = self.depth_so_far[curr_node] + 1
      #Step 6:increment number of expansions
      curr_expansions += 1

    if found_goal:
      path, path_cost = self.reconstruct_path(self.came_from, self.start_node, self.goal_node, self.cost_so_far)
    else:
      print ('Found no solution, priority queue empty')
    time_taken = time.time()- start_t
    return path, path_cost, curr_expansions, time_taken, self.came_from, self.cost_so_far, self.c_obs    #Run planner on current env and return data seetn. Also, update current env to next env

  def get_heuristic(self, node, goal):
    """Given a node and goal, calculate features and get heuristic value"""    
    return 0
  
  def get_best_node(self):
    """Evaluates all the nodes in the frontier and returns the best node"""
    return None
  
  def sample_world(self, mode='cycle'):
    self.curr_env_num = (self.curr_env_num+1)%self.num_envs
    file_path = os.path.join(os.path.abspath(self.env_folder), str(self.curr_env_num)+'.png')
    self.curr_env = initialize_env_from_file(file_path)

  def compute_oracle(self, mode='cycle'):
    file_path = os.path.join(os.path.abspath(self.oracle_folder), "oracle_"+str(self.curr_env_num)+'.p')
    self.curr_oracle = pickle.load(cost_so_far, open(file_path, 'rb')) 

  def initialize_env_from_file(self, file_path):
    env = Env2D()
    env.initialize(file_path, self.env_params)
    if self.visualize:
      self.env.initialize_plot(self.lattice.node_to_state(self.start_node), self.lattice.node_to_state(self.goal_node))
    self.initialized = True
    return env
  
  def clear_planner(self):
    self.frontier.clear()
    self.visited = {}
    self.c_obs = []
    self.cost_so_far = {}
    self.came_from = {}
