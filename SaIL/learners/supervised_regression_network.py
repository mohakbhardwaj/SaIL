#!/usr/bin/env python
"""Generic network class for supervised regression
Created on: March 25, 2017
Author: Mohak Bhardwaj"""
from __future__ import division #force float division
from collections import defaultdict
import numpy as np
from sets import Set
np.set_printoptions(threshold='nan')
import os
import random


class SupervisedRegressionNetwork():
  def __init__(self, params):
    
    self.output_size = params['output_size']
    self.learning_rate = params['learning_rate']
    self.batch_size =  params['batch_size']
    self.training_epochs = params['training_epochs']
    self.display_step = 1

    self.seed_val = params['seed_val']
    self.graph_type = params['graph_type']
    self.start_n = params['start_n']
    self.goal_n = params['goal_n']
    
    if params['mode'] == "gpu":
      self.device = '/gpu:0'
    else:
      self.device = '/cpu:0'
  
    #Unpack normalization terms
    self.euc_dist_norm = params['normalizn_terms']['euc_dist_norm']
    self.man_dist_norm = params['normalizn_terms']['man_dist_norm']
    self.max_children = params['normalizn_terms']['max_children']
    self.coord_norm = params['normalizn_terms']['coord_norm']
    self.g_norm = params['normalizn_terms']['g_norm']
    
    if self.graph_type == "XY":
      self.ndims = 2
      self.input_shape = [14]

    elif self.graph_type == "XYH": 
      self.ndims = 3
      self.input_shape = [17]
      self.dubins_dist_norm = params['normalizn_terms']['dubins_dist_norm']
    
    self.norm_start_n = np.array(self.start_n,dtype=np.float)/self.coord_norm
    self.norm_goal_n = np.array(self.goal_n,dtype=np.float)/self.coord_norm
    
    #Dictionaries that keep track of important values for feature calculation(These are updated by the search)
    self.cost_so_far = defaultdict(lambda: np.inf)            #For each node, this is the cost of the shortest path to the start
    self.num_invalid_predecessors = defaultdict(lambda: 0)   #For each node, this is the number of invalid predecessor edges (including siblings of parent)
    self.num_invalid_children =  defaultdict(lambda: 0)      #For each node, this is the number of invalid children edges
    self.num_invalid_siblings = defaultdict(lambda: 0)       #For each node, this is the number of invalid siblings edges (from best parent so far)
    self.num_invalid_grand_children = defaultdict(lambda: 0) #For each node, this is the number of invalid grandchildren edges (seen so far)
    self.depth_so_far = defaultdict(lambda: 0)               #For each node, this is the depth of the node along the tree(along shortest path)
    self.euc_dist = defaultdict(lambda: 0)
    self.man_dist = defaultdict(lambda: 0)
    # self.distance_transform = None
    self.c_obs = Set()
    
    if self.graph_type == "XYH":
      self.dubins_dist = defaultdict(lambda: 0)

    self.data = [] #Maintains database 
    self.frontier = [] #Maintain frontier
    
  def initialize(self):
    global tf
    global tflearn
    # import matplotlib.pyplot as plt
    import tensorflow as tf 
    import tflearn
    config = tf.ConfigProto()
    config.allow_soft_placement=True
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)#, log_device_placement=True))
    with tf.device(self.device):
      self.graph_ops = self.init_graph()
      self.init_op = tf.global_variables_initializer()

    self.sess.run(self.init_op)
    print('network created and initialized')

  def create_network(self):
    """Constructs and initializes core network architecture"""
    open_list_input = tf.placeholder(tf.float32, [None, self.ndims])   
    feature_op = tf.py_func(self.get_feature_vec, [open_list_input], tf.float32, stateful=True, name="feature_calc")
    
    state_input = tf.placeholder(tf.float32, [None] + self.input_shape)
    net = tflearn.fully_connected(state_input, 100, activation='relu')
    net = tflearn.fully_connected(net, 50, activation ='relu')
    output = tflearn.fully_connected(net, self.output_size, activation = 'linear')
    return state_input, output, feature_op, open_list_input


  def init_graph(self):
    """Overall architecture including target network,
    gradient ops etc"""
    state_input, output, feature_op, open_list_input = self.create_network()
    network_params = tf.trainable_variables()
    target = tf.placeholder(tf.float32, [None] + [self.output_size])
    cost = tf.reduce_sum(tf.pow(output - target, 2))/(2*self.batch_size)
    optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)
    train_net = optimizer.minimize(cost, var_list = network_params)
    saver = tf.train.Saver()
    graph_operations = {"s": state_input,\
                        "output": output,\
                        "target": target,\
                        "cost": cost,\
                        "train_net": train_net,\
                        "open_list_input": open_list_input,\
                        "feature_calc": feature_op,\
                        "network_params": network_params,\
                        "saver": saver}
    return graph_operations
  
  def get_feature_vec(self, open_list):
    """Given a node, calculate the features for that node"""
    features = np.zeros(shape=[len(open_list)]+ self.input_shape, dtype=np.float32)
    
    for i, node in enumerate(open_list):
      if not type(node) is tuple:
        node = tuple(node)
            
      feature_vec = list(node/self.coord_norm)   
      feature_vec = list(self.norm_goal_n)
      feature_vec.append(self.cost_so_far[node]/self.g_norm) 
      feature_vec.append(self.euc_dist[node]/self.euc_dist_norm)
      feature_vec.append(self.man_dist[node]/self.man_dist_norm)
      if self.graph_type == "XYH": feature_vec.append(self.dubins_dist[node]/self.dubins_dist_norm)
      feature_vec.append(self.num_invalid_predecessors[node]/(self.depth_so_far[node]*self.max_children)) #Normalized invalid predecessors
      feature_vec.append(self.num_invalid_siblings[node]/self.max_children)
      feature_vec.append(self.num_invalid_children[node]/self.max_children)
      feature_vec.append(self.num_invalid_grand_children[node]/(2*self.max_children))
      if len(self.c_obs) > 0:
        feature_vec.append(self.closest_obs(node)/self.euc_dist_norm)
      else:
        feature_vec.append(-1.0)
      # print feature_vec[-1]
      features[i] = np.array(feature_vec, dtype=np.float32)
    # print features[-1,:]
    return features

  def train(self):
    print(len(self.data))
    #Shuffle the database
    random.shuffle(self.data)
    for epoch in xrange(self.training_epochs):
      # random.shuffle(database)
      avg_cost = 0.
      total_batch = int(len(self.data)/self.batch_size)
      for i in xrange(total_batch):
        batch_x, batch_y = self.get_next_batch(self.data, i)
        #Run optimization op(backprop) and cost op(to get loss value)
        _, c = self.sess.run([self.graph_ops['train_net'], self.graph_ops['cost']],\
                             feed_dict = {self.graph_ops['s']:batch_x,\
                                          self.graph_ops['target']:batch_y})
        #Compute Average Loss
        avg_cost+= c/total_batch
      
      #Display logs per epoch
      if epoch%self.display_step == 0:
        print "epoch:", '%04d' % (epoch+1), "cost=", \
              "{:.9f}".format(np.sqrt(avg_cost))
    
    print('optimization finished!')
    return np.sqrt(avg_cost)

  def get_best_node(self):
    """takes as input an open list and returns the best node to be expanded"""
    features = self.graph_ops['feature_calc'].eval(session = self.sess, feed_dict = {self.graph_ops['open_list_input']: self.frontier})
    features = features.reshape([len(self.frontier)] + self.input_shape)
    output = self.graph_ops['output'].eval(session=self.sess, feed_dict={self.graph_ops['s']: features})
    return np.argmin(output)

  def get_q_value(self, obs):
    obs = obs.reshape(self.input_shape)
    output = self.graph_ops['output'].eval(session=self.sess, feed_dict={self.graph_ops['s']:[obs]})
    return np.argmax(output)

  def update_database(self, features, labels):
    for f,l in zip(features, labels):
      self.data.append((f,l))

  def save_params(self, folder_path):
    file_name = os.path.join(folder_path, self.graph_type + "_" + str(self.ndims))
    print(file_name)
    save_path = self.graph_ops['saver'].save(self.sess, file_name)
    print("Model saved in file: %s" % file_name)
    return

  def load_params(self, folder_path):
    file_name = file_name = os.path.join(folder_path, self.graph_type + "_" + str(self.ndims))
    self.graph_ops['saver'].restore(self.sess, file_name)
    print('Weights loaded from file %s'%file_name)
  
  def get_params(self):
    return self.sess.run(self.graph_ops['network_params'])

  def set_params(self, input_params):
    [self.graph_ops['network_params'].assign(input_params[i]) for i in range(len(input_params))]
    

  def get_next_batch(self, database, i):
    batch = database[i*self.batch_size: (i+1)*self.batch_size]
    batch_x = np.array([_[0] for _ in batch])
    batch_y = np.array([_[1] for _ in batch])
    new_shape_ip = [self.batch_size] + self.input_shape
    new_shape_op = [self.batch_size] + [self.output_size]
    batch_x = batch_x.reshape(new_shape_ip)  
    batch_y = batch_y.reshape(new_shape_op)
    return batch_x, batch_y
  
  def reset(self):
    self.sess.run(self.init_op)

  def clear_features(self):
    #Dictionaries that keep track of important values for feature calculation(These are updated by the search)
    self.cost_so_far = defaultdict(lambda: np.inf)  
    self.num_invalid_predecessors = defaultdict(lambda: 0)   
    self.num_invalid_children =  defaultdict(lambda: 0)      
    self.num_invalid_siblings = defaultdict(lambda: 0)       
    self.num_invalid_grand_children = defaultdict(lambda: 0)
    self.depth_so_far = defaultdict(lambda: 0) 
    self.c_obs.clear()

  def clear_frontier(self):
    self.frontier = []

  def append_to_frontier(self, node):
    self.frontier.append(node)

  def get_node_from_frontier(self, idx):
    return self.frontier[idx]

  def delete_from_frontier(self, idx):
    del self.frontier[idx]    

  def update_cobs(self, nodes):
    for node in nodes:
      self.c_obs.add(node)

  def closest_obs(self, node):
    nodes = np.asarray(list(self.c_obs))
    node = np.asarray(node)
    deltas = nodes - node
    dist_2 = np.sqrt(np.einsum('ij,ij->i', deltas, deltas))
    return np.min(dist_2)