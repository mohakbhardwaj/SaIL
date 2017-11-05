#!/usr/bin/env python
from collections import defaultdict
import math
import numpy as np
import time
from planning_python.data_structures.priority_queue import PriorityQueue
from planning_python.planners import SearchBasedPlanner
from SaIL.planners import SaILPlanner

class TrainPlanner(SaILPlanner):
  def __init__(self):
    """Planner takes as input a planning problem object and returns
      the path and generated states
    @param problem   - planning problem object with following fields
    """
    
    super(TrainPlanner, self).__init__()

  def plan(self, oracle, beta=1, num_datapoints=0, max_expansions = 100000):
    assert self.initialized == True, "Planner has not been initialized properly. Please call initialize or reset_problem function before plan function"
    start_t = time.time()
    curr_expansions = 0         #Number of expansions done
    num_rexpansions = 0
    found_goal = False
    path =[]
    path_cost = np.inf
    dataset = []
    #Uniformly sample timesteps to collect data    
    if beta == 1:
      oracle_max = oracle.get_optimal_q(self.start_node)
      data_steps = np.random.choice(int(oracle_max), num_datapoints)#, replace=False)
    else: data_steps = np.random.choice(max_expansions, num_datapoints)#, replace=False)
    
    self.came_from[self.start_node]= (None, None)
    self.cost_so_far[self.start_node] = 0.
    self.depth_so_far[self.start_node] = 0
    start_h_val = self.get_heuristic(self.start_node, self.goal_node)
    self.frontier.put(self.start_node, start_h_val)
    self.f_o.put(self.start_node, oracle.get_optimal_q(self.start_node))

    while not self.frontier.empty():
      #Check 1: Stop search if frontier gets too large
      if curr_expansions >= max_expansions:
        print("Max Expansions Done.")
        break
      #Check 2: Stop search if open list gets too large
      if self.frontier.size() > 500000:
        print("Timeout.")
        break
      #Step 1: Collect data if curr expansions is in sampled timesteps
      if curr_expansions in data_steps:
        #we choose a random feasible action
        rand_idx = np.random.randint(self.frontier.size())
        #collect features, label for action
        rand_f, y = self.collect_data(rand_idx, oracle)
        #append to dataset
        dataset.append((rand_f, y))
      #Step 2: Pop the best node (according to the mixture policy)
      h, curr_node = self.policy_mix(beta)

      if curr_node in self.visited:
        continue

      #Step 3: Add to visited and increment expansions
      self.visited[curr_node] = 1
      curr_expansions += 1

      #Step 4: Get successors for current node
      neighbors, edge_costs, valid_edges, invalid_edges = self.get_successors(curr_node)

      #Step 5: Update c_obs with collision checks performed
      for edge,coll_state in invalid_edges:
        self.c_obs.add(coll_state)
      #Step 6: Expand neighbors and update lists
      g = self.cost_so_far[curr_node]
      for i, neighbor in enumerate(neighbors):
        new_g = g + edge_costs[i]
        if neighbor not in self.visited:
          #update statistics if shorter path found
          if neighbor not in self.cost_so_far or new_g <= self.cost_so_far[neighbor]:
            self.came_from[neighbor] = (curr_node, valid_edges[i])
            self.cost_so_far[neighbor] = new_g
            self.depth_so_far[neighbor] = self.depth_so_far[curr_node] + 1
          #If neighbor is goal, then end search
          if neighbor == self.goal_node: 
            print "Found goal"
            found_goal = True
            break
          #append to frontiers
          h_val = self.get_heuristic(neighbor, self.goal_node)
          self.frontier.put(neighbor, h_val)
          self.f_o.put(neighbor, oracle.get_optimal_q(neighbor))
      
      if found_goal: break

    if found_goal:
      path, path_cost = self.reconstruct_path(self.came_from, self.start_node, self.goal_node, self.cost_so_far)
    else:
      print ('Found no solution, priority queue empty')
    plan_time = time.time() - start_t

    return path, path_cost, curr_expansions, plan_time, self.came_from, self.cost_so_far, self.c_obs, dataset


class TestPlanner(SaILPlanner):
  def __init__(self):
    """Planner takes as input a planning problem object and returns
      the path and generated states
    @param problem   - planning problem object with following fields
    """
    
    super(TestPlanner, self).__init__()

  def plan(self, oracle, max_expansions = 100000):
    assert self.initialized == True, "Planner has not been initialized properly. Please call initialize or reset_problem function before plan function"
    start_t = time.time()
    
    self.came_from[self.start_node]= (None, None)
    self.cost_so_far[self.start_node] = 0.
    self.depth_so_far[self.start_node] = 0
    start_h_val = self.get_heuristic(self.start_node, self.goal_node)
    self.frontier.put(self.start_node, start_h_val)

    curr_expansions = 0         #Number of expansions done
    num_rexpansions = 0
    avg_loss = math.pow(start_h_val-oracle.get_optimal_q(self.start_node), 2)
    found_goal = False
    path =[]
    path_cost = np.inf
    dataset = []

    while not self.frontier.empty():
      #Check 1: Stop search if frontier gets too large
      if curr_expansions >= max_expansions:
        print("Max Expansions Done.")
        break
      #Check 2: Stop search if open list gets too large
      if self.frontier.size() > 500000:
        print("Timeout.")
        break
      
      #Step 1: Pop the best node (according to the learner's policy)
      h, curr_node = self.policy()
      avg_loss += math.pow(h - oracle.get_optimal_q(curr_node), 2)
      if curr_node in self.visited:
        continue

      #Step 3: Add to visited and increment expansions
      self.visited[curr_node] = 1
      curr_expansions += 1

      #Step 4: Get successors for current node
      neighbors, edge_costs, valid_edges, invalid_edges = self.get_successors(curr_node)

      #Step 5: Update c_obs with collision checks performed
      for edge,coll_state in invalid_edges:
        self.c_obs.add(coll_state)

      #Step 6: Expand neighbors and update lists
      g = self.cost_so_far[curr_node]
      for i, neighbor in enumerate(neighbors):
        new_g = g + edge_costs[i]
        if neighbor not in self.visited:
          #update statistics if shorter path found      
          if neighbor not in self.cost_so_far or new_g <= self.cost_so_far[neighbor]:
            self.came_from[neighbor] = (curr_node, valid_edges[i])
            self.cost_so_far[neighbor] = new_g
            self.depth_so_far[neighbor] = self.depth_so_far[curr_node] + 1
          #If neighbor is goal, then end search
          if neighbor == self.goal_node: 
            print "Found goal"
            found_goal = True
            break
          #append to frontier
          h_val = self.get_heuristic(neighbor, self.goal_node)
          self.frontier.put(neighbor, h_val)

      if found_goal: break

    if found_goal:
      path, path_cost = self.reconstruct_path(self.came_from, self.start_node, self.goal_node, self.cost_so_far)
    else:
      print ('Found no solution, priority queue empty')
    plan_time = time.time() - start_t
    avg_loss/= (2*curr_expansions)
    avg_loss = np.sqrt(avg_loss)

    return path, path_cost, curr_expansions, plan_time, self.came_from, self.cost_so_far, self.c_obs, avg_loss


