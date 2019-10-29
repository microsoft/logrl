'''
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
'''

# This file is partially derived from Dopamine with the following original copyright note:
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compact implementation of a LogDQN agent.

Details in "Using a Logarithmic Mapping to Enable Lower Discount Factors 
in Reinforcement Learning" by van Seijen, Fatemi, Tavakoli (2019).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random



from dopamine.agents.dqn import dqn_agent
from log_dqn import circular_replay_buffer_64
import numpy as np
import tensorflow as tf

import gin.tf

slim = tf.contrib.slim


@gin.configurable
class LogDQNAgent(dqn_agent.DQNAgent):
  """An implementation of the LogDQN agent."""

  def __init__(self,
               sess,
               num_actions,
               gamma=0.96,
               c=0.5,
               k=100,
               pos_q_init=1.0,
               neg_q_init=0.0,
               net_init_method='asym',
               alpha=0.00025,
               clip_qt_max=True,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               tf_device='/cpu:*',
               use_staging=True,
               max_tf_checkpoints_to_keep=3,
               loss_type='Huber',
               optimizer=tf.train.RMSPropOptimizer(
                   learning_rate=0.0025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=0.00001,
                   centered=True),
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      gamma: float, discount factor with the usual RL meaning.
      c: float, a hyperparameter of the logarithmic mapping approach.
      k: int, a hyperparameter of the logarithmic mapping approach. 
      pos_q_init: float, used to evaluate 'd' in logarithmic mapping.
      neg_q_init: float, used to evaluate 'd' in logarithmic mapping.
      net_init_method: str, determines how to initialize the weights of the 
        LogDQN network heads. 
      alpha: float, effective step-size: alpha = beta_reg * beta_log.
      clip_qt_max: bool, when True clip the maximum target value.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between LogDQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      max_tf_checkpoints_to_keep: int, the number of TensorFlow checkpoints to
        keep.
      optimizer: `tf.train.Optimizer`, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """

    tf.logging.info('Creating %s agent with the following parameters:',
                    self.__class__.__name__)
    tf.logging.info('\t gamma: %f', gamma)
    tf.logging.info('\t c: %f', c)
    tf.logging.info('\t k: %d', k)
    tf.logging.info('\t pos_q_init: %s', np.amax([gamma**k, pos_q_init]))
    tf.logging.info('\t neg_q_init: %s', np.amax([gamma**k, neg_q_init]))
    tf.logging.info('\t pos_Delta: %s', -c * np.log(np.amax([gamma**k, pos_q_init])))
    tf.logging.info('\t neg_Delta: %s', -c * np.log(np.amax([gamma**k, neg_q_init])))
    tf.logging.info('\t net_init_method: %s', net_init_method)
    tf.logging.info('\t clip_qt_max: %s', clip_qt_max)
    tf.logging.info('\t update_horizon: %d', update_horizon)
    tf.logging.info('\t min_replay_history: %d', min_replay_history)
    tf.logging.info('\t update_period: %d', update_period)
    tf.logging.info('\t target_update_period: %d', target_update_period)
    tf.logging.info('\t epsilon_train: %f', epsilon_train)
    tf.logging.info('\t epsilon_eval: %f', epsilon_eval)
    tf.logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
    tf.logging.info('\t tf_device: %s', tf_device)
    tf.logging.info('\t use_staging: %s', use_staging)
    tf.logging.info('\t loss_type: %s', loss_type)
    tf.logging.info('\t optimizer: %s', optimizer)
    tf.logging.info('\t beta_log: %f', optimizer._learning_rate)
    tf.logging.info('\t beta_reg: %f', alpha / optimizer._learning_rate)
    tf.logging.info('\t alpha: %f', alpha)

    self.tf_float = tf.float64
    self.np_float = np.float64

    self.num_actions = num_actions
    self.gamma = self.np_float(gamma)
    self.c = self.np_float(c) 
    self.k = self.np_float(k)
    self.pos_q_init = np.amax([self.gamma**self.k, self.np_float(pos_q_init)])
    self.neg_q_init = np.amax([self.gamma**self.k, self.np_float(neg_q_init)])
    self.pos_Delta = -self.c * np.log(self.pos_q_init)
    self.neg_Delta = -self.c * np.log(self.neg_q_init)
    self.clip_qt_max = clip_qt_max
    self.net_init_method = net_init_method
    self.alpha = alpha
    self.beta_reg = alpha / optimizer._learning_rate
    self.update_horizon = update_horizon
    self.cumulative_gamma = self.gamma**self.np_float(update_horizon)
    self.min_replay_history = min_replay_history
    self.target_update_period = target_update_period
    self.epsilon_fn = epsilon_fn
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self.update_period = update_period
    self.eval_mode = False
    self.training_steps = 0
    self.optimizer = optimizer
    self.loss_type = loss_type
    self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency

    with tf.device(tf_device):
      # Create a placeholder for the state input to the LogDQN network.
      # The last axis indicates the number of consecutive frames stacked.
      state_shape = [1, 
          dqn_agent.OBSERVATION_SHAPE, 
          dqn_agent.OBSERVATION_SHAPE, 
          dqn_agent.STACK_SIZE]
      self.state = np.zeros(state_shape)
      self.state_ph = tf.placeholder(tf.uint8, state_shape, name='state_ph')
      self._replay = self._build_replay_buffer(use_staging)

      self._build_networks()

      self._train_op = self._build_train_op()
      self._sync_qt_ops = self._build_sync_op()

    if self.summary_writer is not None:
      # All tf.summaries should have been defined prior to running this.
      self._merged_summaries = tf.summary.merge_all()
    self._sess = sess
    self._saver = tf.train.Saver(max_to_keep=max_tf_checkpoints_to_keep)

    # Variables to be initialized by the agent once it interacts with the
    # environment.
    self._observation = None
    self._last_observation = None

  def _get_network_type(self):
    """Returns the type of the outputs of a Q-value network.

    Returns:
      net_type: _network_type object defining the outputs of the network.
    """
    return collections.namedtuple('LogDQN_network', ['q_values',
        'pos_q_tilde_values', 'neg_q_tilde_values', 
        'pos_q_values', 'neg_q_values'])

  def _network_template(self, state):
    """Builds the convolutional network used to compute the agent's Q-values.

    Args:
      state: `tf.Tensor`, contains the agent's current state.

    Returns:
      net: _network_type object containing the tensors output by the network.
    """
    net = tf.cast(state, tf.float32)
    net = tf.div(net, 255.)
    net = slim.conv2d(net, 32, [8, 8], stride=4)
    net = slim.conv2d(net, 64, [4, 4], stride=2)
    net = slim.conv2d(net, 64, [3, 3], stride=1)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 512)
    net = tf.cast(net, self.tf_float)

    # Create two network heads with the specified initialization scheme.
    pos_q_tilde_values = slim.fully_connected(net, self.num_actions, 
        activation_fn=None)
    if self.net_init_method=='standard':  
      neg_q_tilde_values = slim.fully_connected(net, self.num_actions, 
          activation_fn=None)
    elif self.net_init_method=='asym':
      neg_q_tilde_values = slim.fully_connected(net, self.num_actions, 
          activation_fn=None,
          weights_initializer=tf.zeros_initializer())

    # Inverse mapping of Q-tilde values.
    pos_q_values = tf.exp((pos_q_tilde_values - self.pos_Delta) / self.c)
    neg_q_values = tf.exp((neg_q_tilde_values - self.neg_Delta) / self.c)

    # Aggregate positive and negative heads' Q-values. 
    q_values = pos_q_values - neg_q_values
    
    return self._get_network_type()(q_values, pos_q_tilde_values, 
        neg_q_tilde_values, pos_q_values, neg_q_values)

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """
    # Calling online_convnet will generate a new graph as defined in
    # self._get_network_template using whatever input is passed, but will always
    # share the same weights.
    self.online_convnet = tf.make_template('Online', self._network_template)
    self.target_convnet = tf.make_template('Target', self._network_template)
    self._net_outputs = self.online_convnet(self.state_ph)
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]

    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)

    # Gets greedy actions over the aggregated target-network's Q-values for the 
    # replay's next states, used for retrieving the target Q-values for both heads. 
    self._replay_next_target_net_q_argmax = tf.argmax(
        self._replay_next_target_net_outputs.q_values, axis=1)   

  def _build_replay_buffer(self, use_staging):
    """Creates a float64-compatible replay buffer used by the agent.

    Args:
      use_staging: bool, if True, uses a staging area to prefetch data for
        faster training.

    Returns:
      A WrapperReplayBuffer64 object.
    """
    return circular_replay_buffer_64.WrappedReplayBuffer64(
        observation_shape=dqn_agent.OBSERVATION_SHAPE,
        stack_size=dqn_agent.STACK_SIZE,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma)

  def _build_target_q_op(self):
    """Build an op used as a target for the logarithmic Q-value.

    Returns:
      target_q_op: An op calculating the logarithmic Q-value.
    """
    one = tf.constant(1, dtype=self.tf_float)
    zero = tf.constant(0, dtype=self.tf_float)
    # One-hot encode the greedy actions over the target-network's aggregated 
    # Q-values for the replay's next states. 
    replay_next_target_net_q_argmax_one_hot = tf.one_hot(
        self._replay_next_target_net_q_argmax, self.num_actions, one, zero, 
        name='replay_next_target_net_q_argmax_one_hot')
    # Calculate each head's target Q-value (in standard space) with the 
    # action that maximizes the target-network's aggregated Q-values for 
    # the replay's next states.
    pos_replay_next_qt_max_unclipped = tf.reduce_sum(
        self._replay_next_target_net_outputs.pos_q_values * \
          replay_next_target_net_q_argmax_one_hot,
        reduction_indices=1,
        name='pos_replay_next_qt_max_unclipped')
    neg_replay_next_qt_max_unclipped = tf.reduce_sum(
        self._replay_next_target_net_outputs.neg_q_values * \
          replay_next_target_net_q_argmax_one_hot,
        reduction_indices=1,
        name='neg_replay_next_qt_max_unclipped')

    # Clips the maximum target-network's positive and negative Q-values 
    # for the replay's next states.
    if self.clip_qt_max:
      min_return = zero   
      max_return = one / (one - self.cumulative_gamma)

      pos_replay_next_qt_max_clipped_min = tf.maximum(min_return, 
          pos_replay_next_qt_max_unclipped)
      pos_replay_next_qt_max = tf.minimum(max_return, 
          pos_replay_next_qt_max_clipped_min)
      
      neg_replay_next_qt_max_clipped_min = tf.maximum(min_return, 
          neg_replay_next_qt_max_unclipped)
      neg_replay_next_qt_max = tf.minimum(max_return, 
          neg_replay_next_qt_max_clipped_min)
    else:
      pos_replay_next_qt_max = pos_replay_next_qt_max_unclipped
      neg_replay_next_qt_max = neg_replay_next_qt_max_unclipped

    # Terminal state masking.
    pos_replay_next_qt_max_masked = pos_replay_next_qt_max * \
        (1. - tf.cast(self._replay.terminals, self.tf_float))
    neg_replay_next_qt_max_masked = neg_replay_next_qt_max * \
        (1. - tf.cast(self._replay.terminals, self.tf_float))

    # Creates the positive and negative head's separate reward signals
    # and bootstraps from the appropriate target for each head.
    # Positive head's reward signal is r if r > 0 and 0 otherwise.
    pos_standard_td_target_unclipped = self._replay.rewards * \
        tf.cast(tf.greater(self._replay.rewards, zero), self.tf_float) + \
          self.cumulative_gamma * pos_replay_next_qt_max_masked
    # Negative head's reward signal is -r if r < 0 and 0 otherwise.
    neg_standard_td_target_unclipped = -1 * self._replay.rewards * \
        tf.cast(tf.less(self._replay.rewards, zero), self.tf_float) + \
          self.cumulative_gamma * neg_replay_next_qt_max_masked
          
    # Clips the minimum TD-targets in the standard space for both positive 
    # and negative heads so as to avoid log(x <= 0).
    pos_standard_td_target = tf.maximum(self.cumulative_gamma**self.k,
        pos_standard_td_target_unclipped)
    neg_standard_td_target = tf.maximum(self.cumulative_gamma**self.k,
        neg_standard_td_target_unclipped)
    
    # Gets the current-network's positive and negative Q-values (in standard 
    # space) for the replay's chosen actions.
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, one, zero, 
        name='replay_action_one_hot')
    pos_replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.pos_q_values * replay_action_one_hot,
        reduction_indices=1, name='pos_replay_chosen_q')
    neg_replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.neg_q_values * replay_action_one_hot,
        reduction_indices=1, name='neg_replay_chosen_q') 

    # Averaging samples in the standard space.
    pos_UT_new = pos_replay_chosen_q + \
        self.beta_reg * (pos_standard_td_target - pos_replay_chosen_q)
    neg_UT_new = neg_replay_chosen_q + \
        self.beta_reg * (neg_standard_td_target - neg_replay_chosen_q)

    # Forward mapping.
    pos_log_td_target = self.c * tf.log(pos_UT_new) + self.pos_Delta
    neg_log_td_target = self.c * tf.log(neg_UT_new) + self.neg_Delta
 
    pos_log_td_target = tf.cast(pos_log_td_target, tf.float32)
    neg_log_td_target = tf.cast(neg_log_td_target, tf.float32)
    return pos_log_td_target, neg_log_td_target

  def _build_train_op(self):
    """Builds a training op.

    Returns:
      train_op: An op performing one step of training from replay data.
    """
    one = tf.constant(1, dtype=self.tf_float)
    zero = tf.constant(0, dtype=self.tf_float)
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, one, zero, name='action_one_hot')
    # For the replay's chosen actions, these are the current-network's positive 
    # and negative Q-tilde values, which will be updated for each head separately.
    pos_replay_chosen_q_tilde = tf.reduce_sum(
        self._replay_net_outputs.pos_q_tilde_values * replay_action_one_hot,
        reduction_indices=1,
        name='pos_replay_chosen_q_tilde')
    neg_replay_chosen_q_tilde = tf.reduce_sum(
        self._replay_net_outputs.neg_q_tilde_values * replay_action_one_hot,
        reduction_indices=1,
        name='neg_replay_chosen_q_tilde')

    pos_replay_chosen_q_tilde = tf.cast(pos_replay_chosen_q_tilde, tf.float32)
    neg_replay_chosen_q_tilde = tf.cast(neg_replay_chosen_q_tilde, tf.float32)

    # Gets the target for both positive and negative heads.
    pos_log_td_target, neg_log_td_target = self._build_target_q_op()
    pos_log_target = tf.stop_gradient(pos_log_td_target)
    neg_log_target = tf.stop_gradient(neg_log_td_target)

    if self.loss_type == 'Huber':
      pos_loss = tf.losses.huber_loss(pos_log_target,
          pos_replay_chosen_q_tilde, reduction=tf.losses.Reduction.NONE)
      neg_loss = tf.losses.huber_loss(neg_log_target,
          neg_replay_chosen_q_tilde, reduction=tf.losses.Reduction.NONE)
    elif self.loss_type == 'MSE':
      pos_loss = tf.losses.mean_squared_error(pos_log_target,
          pos_replay_chosen_q_tilde, reduction=tf.losses.Reduction.NONE)
      neg_loss = tf.losses.mean_squared_error(neg_log_target,
          neg_replay_chosen_q_tilde, reduction=tf.losses.Reduction.NONE)  

    loss = pos_loss + neg_loss
    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.summary.scalar(self.loss_type+'Loss', tf.reduce_mean(loss))
    return self.optimizer.minimize(tf.reduce_mean(loss))