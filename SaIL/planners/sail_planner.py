#!/usr/bin/env python
"""Planner class that derives from search based planner. It defines festure calculation, normalization and 
heuristic calculation.
The train and test planner derive from this planner"""


from collections import defaultdict
from math import atan2, pi
import numpy as np
import time
from planning_python.data_structures.priority_queue import PriorityQueue
from planning_python.planners.search_based_planner import SearchBasedPlanner
from planning_python.heuristic_functions.heuristic_function import EuclideanHeuristicNoAng, ManhattanHeuristicNoAng, DubinsHeuristic

class SaILPlanner(SearchBasedPlanner):
  def __init__(self):
    """Planner takes as input a planning problem object and returns
      the path and generated states
    @param problem   - planning problem object with following fields
    """
    self.frontier = PriorityQueue() #Open list
    self.f_o = PriorityQueue() #Frontier sorted according to oracle
    self.visited = {} #Keep track of visited cells
    self.c_obs = set()  #Keep track of collision checks done so far
    self.cost_so_far = {} #Keep track of cost of path to the node
    self.came_from = {} #Keep track of parent during search
    self.depth_so_far = {} #Keep track of depth in tree
    
    super(SaILPlanner, self).__init__()
  
  def initialize(self, problem):
    super(SaILPlanner, self).initialize(problem)
    
    self.euch = EuclideanHeuristicNoAng()
    self.manh = ManhattanHeuristicNoAng()
    if self.lattice.ndims == 3: self.dubh = DubinsHeuristic(self.lattice.radius)  
    self.norm_terms = dict()
    min_st = np.array((self.lattice.x_lims[0], self.lattice.y_lims[0]))
    max_st = np.array((self.lattice.x_lims[1], self.lattice.y_lims[1]))
    self.norm_terms['euc'] = self.euch.get_heuristic(min_st, max_st)
    self.norm_terms['manhattan'] = self.manh.get_heuristic(min_st, max_st)
    print('Planner Initialized')

  def get_features(self, node):
    """Calculates features for the node give the current state of the search"""  
  
    s = self.lattice.node_to_state(node)
    goal_s = self.lattice.node_to_state(self.goal_node)
    start_s = self.lattice.node_to_state(self.start_node) 
    #calculate search based features
    features = list(s)
    features.append(self.cost_so_far[node])
    # features.append(self.euch.get_heuristic(s, start_s)/self.norm_terms['euc'])       #normalized euclidean distance to start
    features.append(self.euch.get_heuristic(s, goal_s))#/self.norm_terms['euc'] )       #normalized euclidean distance to goal
    # features.append(self.manh.get_heuristic(s, start_s)/self.norm_terms['manhattan']) #normalized manhattan distance to start
    features.append(self.manh.get_heuristic(s, goal_s))#/self.norm_terms['manhattan'])  #normalized manhattan distance to goal
  
    if self.lattice.ndims == 3:
      features.append(self.dubh.get_heuristic(s, start_s)) #normalized dubins distance to start
      features.append(self.dubh.get_heuristic(s, goal_s)) #normalized dubins distance to goal
      features.append(s[-1])                #heading of the state

    features.append(self.depth_so_far[node])
    features += list(goal_s)

    #calculate environment based features 
    #if distance transform available, use that
    if self.env.distance_transform_available:
      d_obs, obs_dx, obs_dy = self.env.get_obstacle_distance(s)
      features += [d_obs, atan2(obs_dy, obs_dx)/pi]
      feature_arr = np.asarray(features, dtype=np.float32)
      feature_arr = (feature_arr - min(feature_arr))/(max(feature_arr) - min(feature_arr)) #normalize
    
    else: #work off of c_obs only
      if len(self.c_obs):
        closest_obs = []
        closest_obs_x = []
        closest_obs_y = []
        d_cobs_min = np.inf
        d_x_min = np.inf
        d_y_min = np.inf

        for obs_config in self.c_obs:
          d = self.euch.get_heuristic(s, obs_config)
          d_x, d_y = list(np.abs(s[0:2] - obs_config[0:2]))
          
          if d < d_cobs_min:
            d_cobs_min = d
            closest_obs = obs_config
          if d_x < d_x_min:
            d_x_min = d_x
            closest_obs_x = obs_config
          if d_y < d_y_min:
            d_y_min = d_y
            closest_obs_y = obs_config
        
        features += [d_cobs_min] + list(closest_obs) + [d_x_min] + list(closest_obs_x)  + [d_y_min] + list(closest_obs_y)
        #Normalize features here
        feature_arr = np.asarray(features, dtype=np.float32)
        feature_arr = (feature_arr - min(feature_arr))/(max(feature_arr) - min(feature_arr)) #normalize
      else:
        feature_arr  = np.array(features, dtype = np.float32)
        feature_arr  = (feature_arr - min(feature_arr))/(max(feature_arr) - min(feature_arr)) #normalize
        env_features =   np.array([-1]*(3 + (self.lattice.ndims*3)), dtype=np.float32) #-1 for all obstacles
        feature_arr = np.concatenate((feature_arr, env_features))

    return feature_arr

  def get_heuristic(self, node1, node2):
    """Requires a heuristic function that goes works off of feature"""
    if self.heuristic == None:
      return 0
    ftrs = self.get_features(node1)
    # s_2 = self.lattice.node_to_state(node2)
    h_val = self.heuristic.get_heuristic(ftrs)
    return h_val

  def clear_planner(self):
    self.frontier.clear()
    self.f_o.clear()
    self.visited.clear()
    self.c_obs.clear()
    self.cost_so_far.clear()
    self.came_from.clear()
  
  def collect_data(self, rand_idx, oracle):
    _, _, rand_node = self.frontier.get_idx(rand_idx) 
    #we get features for that action
    rand_f = self.get_features(rand_node)
    #we query oracle for label
    y = oracle.get_optimal_q(rand_node)
    return rand_f, y

  def policy(self):
    h, _, curr_node = self.frontier.get()
    return h, curr_node
  
  def policy_mix(self, beta):
    """Implements the mixture policy"""
    if np.random.sample(1) < beta:
      h, _ , curr_node = self.f_o.get()
      _, _, _ = self.frontier.pop_task(curr_node)
    else:
      h, _, curr_node = self.frontier.get()
      _, _, _ = self.f_o.pop_task(curr_node)
    
    return h, curr_node   



