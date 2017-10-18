#!/usr/bin/env python
"""Generic network class for supervised regression
Created on: March 25, 2017
Author: Mohak Bhardwaj"""

from collections import defaultdict
from planning_python.heuristic_functions.heuristic_function import EuclideanHeuristicNoAng, ManhattanHeuristicNoAng, DubinsHeuristic
import numpy as np
import random


class SupervisedRegressionNetwork():
  def __init__(self                       ,\
               output_size                ,\
               input_size                 ,\
               learning_rate = 0.001      ,\
               batch_size = 32            ,\
               training_epochs = 15       ,\
               summary_file = "learner_1" ,\
               mode = "cpu"               ,\
               seed_val = 1234            ,\
               graph_type = "XY"):
    
    
    self.output_size = output_size
    self.input_size = input_size
    self.learning_rate = learning_rate
    self.batch_size =  batch_size
    self.training_epochs = training_epochs
    self.display_step = 1
    self.summary_dir_train = os.path.join(os.path.abspath('saved_data/summaries'), summary_file+'_train') 
    self.summary_dir_test = os.path.join(os.path.abspath('saved_data/summaries'), summary_file+'_test') 
    print self.summary_dir_test
    print self.summary_dir_train
    self.seed_val = seed_val
    self.input_shape = [self.input_size]
    
    if mode == "gpu":
      self.device = '/gpu:0'
    else:
      self.device = '/cpu:0'
    
    global tf
    global tflearn
    import tensorflow as tf 
    import tflearn

    # config = tf.ConfigProto()
    # # config.device_count = {'GPU': 0}
    # # config.log_device_placement=True
    # config.allow_soft_placement=True
    # config.gpu_options.allow_growth = True  
    self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))#, log_device_placement=True))
    
    #Heuristic Functions that will be helpful in calculating features
    self.hs = [EuclideanHeuristicNoAng(), ManhattanHeuristicNoAng()]

    #Some values that will help with normalization of features
    if self.graph_type == "XY":
      self.dist_norm = (lattice_params['x_lims'][1]-lattice_params['x_lims'][0])*(lattice_params['y_lims'][1]-lattice_params['y_lims'][0])#Max possible distance between two nodes
      self.max_children = self.lattice.children.shape[0]
      self.coord_norm = np.array([self.lattice.num_cells[0], self.lattice.num_cells[1]], dtype=np.float)

    elif self.graph_type == "XYH": 
      self.dist_norm = None #Max possible dubin's distance between two nodes
      self.max_children = len(self.lattice.children[0])
      self.coord_norm = np.array([self.lattice.num_cells[0], self.lattice.num_cells[1], self.lattice.num_headings], dtype=np.float)
      self.hs.append(DubinsHeuristic(lattice_params['radius']))
    self.norm_start_n = np.array(self.start_n,dtype=np.float)/self.coord_norm
    self.norm_goal_n = np.array(self.goal_n,dtype=np.float)/self.coord_norm
    
    #Dictionaries that keep track of important values for feature calculation
    self.cost_so_far = defaultdict(lambda: np.inf)  #For each node, this is the cost of the shortest path to the start
    self.num_invalid_predecessors = defaultdict(lambda: -1)   #For each node, this is the number of invalid predecessor edges (including siblings of parent)
    self.num_invalid_children =  defaultdict(lambda: -1)       #For each node, this is the number of invalid children edges
    self.num_invalid_siblings = defaultdict(lambda: -1)       #For each node, this is the number of invalid siblings edges (from best parent so far)
    self.num_invalid_grand_children = defaultdict(lambda: -1) #For each node, this is the number of invalid grandchildren edges (seen so far)
    self.depth_so_far = defaultdict(lambda: -1) #For each node, this is the depth of the node along the tree(along shortest path)

    with tf.device(self.device):
      self.graph_ops = self.init_graph()
      self.init_op = tf.global_variables_initializer()
      
      # _, self.episode_stats_vars = self.build_summaries()
      
      # self.summary_ops = tf.summary.merge_all()
      # print('Here')  
      # self.test_writer = tf.summary.FileWriter(self.summary_dir_test, self.sess.graph) 
      # print('Here2')  
      # self.train_writer = tf.summary.FileWriter(self.summary_dir_train, self.sess.graph)
      # print('Her3')

    self.sess.run(self.init_op)
    print('network created and initialized')

  def create_network(self):
    """Constructs and initializes core network architecture"""  
    state_input = tf.placeholder(tf.float32, [None] + self.input_shape)
    net = tf.py_func(self.get_feature_vec, state_input, tf.float32, stateful=True, name="feature_calc")
    net = tflearn.fully_connected(state_input, 100, activation='relu')
    net = tflearn.fully_connected(net, 50, activation ='relu')
    output = tflearn.fully_connected(net, self.output_size, activation = 'linear')
    return state_input, output


  def init_graph(self):
    """Overall architecture including target network,
    gradient ops etc"""
    state_input, output = self.create_network()
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
                        "network_params": network_params,\
                        "saver": saver}
    return graph_operations
  
  def get_feature_vec(self, node):
    """Given a node, calculate the features for that node"""
    feature_vec = [self.norm_start_n, self.norm_goal_n]
    feature_vec.append(node,/self.coord_norm)    
    for h in self.heuristic_functions: feature_vec.append(h(node, self.goal_n)/self.dist_norm) #Normalize the distances
    feature_vec.append(self.num_invalid_predecessors[node]/self.depth_so_far[node]*self.max_children) #Normalized invalid predecessors
    feature_vec.append(self.num_invalid_siblings/self.max_children)
    feature_vec.append(self.num_invalid_children/self.max_children)
    feature_vec.append(self.num_invalid_grand_children/2*self.max_children)
    return feature_vec

  def train(self, database):
    #Shuffle the database
    random.shuffle(database)
    for epoch in xrange(self.training_epochs):
      # random.shuffle(database)
      avg_cost = 0.
      total_batch = int(len(database)/self.batch_size)
      for i in xrange(total_batch):
        batch_x, batch_y = self.get_next_batch(database, i)
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


  def get_best_node(self, obs):
    """takes as input an open list and returns the best node to be expanded"""
    return None
    
  def get_q_value(self, obs):
    obs = obs.reshape(self.input_shape)
    output = self.graph_ops['output'].eval(session=self.sess, feed_dict={self.graph_ops['s']:[obs]})
    return output

  def save_params(self, file_name):
    #file_path = os.path.join(os.path.abspath('saved_data/saved_models'), file_name +'.ckpt')
    print(file_name)
    save_path = self.graph_ops['saver'].save(self.sess, file_name)
    print("Model saved in file: %s" % file_name)
    return

  def load_params(self, file_name):
    #file_path = os.path.join(os.path.abspath('saved_data/saved_models'), file_name +'.ckpt')
    self.graph_ops['saver'].restore(self.sess, file_name)
    print('Weights loaded from file %s'%file_name)
  
  def get_params(self):
    return self.graph_ops['network_params']

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
