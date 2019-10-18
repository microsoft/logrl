"""The standard DQN replay memory modified to support float64 rewards.

This script modifies the Dopamine's implementation of an out-of-graph 
replay memory + in-graph wrapper to support float64 formatted rewards.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import math
import os
import pickle

from dopamine.replay_memory import circular_replay_buffer
import numpy as np
import tensorflow as tf

import gin.tf


class OutOfGraphReplayBuffer64(circular_replay_buffer.OutOfGraphReplayBuffer):

  def __init__(self,
               observation_shape,
               stack_size,
               replay_capacity,
               batch_size,
               update_horizon=1,
               gamma=0.99,
               max_sample_attempts=circular_replay_buffer.MAX_SAMPLE_ATTEMPTS,
               extra_storage_types=None,
               observation_dtype=np.uint8):
    super(OutOfGraphReplayBuffer64, self).__init__(
        observation_shape=observation_shape,
        stack_size=stack_size,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        update_horizon=update_horizon,
        gamma=gamma,
        max_sample_attempts=max_sample_attempts,
        extra_storage_types=extra_storage_types,
        observation_dtype=observation_dtype)
    self._cumulative_discount_vector = np.array(
        [self._gamma**np.float64(n) for n in range(update_horizon)],
        dtype=np.float64)

  def get_storage_signature(self):
    storage_elements = [
        circular_replay_buffer.ReplayElement('observation', 
            self._observation_shape, self._observation_dtype),
        circular_replay_buffer.ReplayElement('action', (), np.int32),
        circular_replay_buffer.ReplayElement('reward', (), np.float64),
        circular_replay_buffer.ReplayElement('terminal', (), np.uint8)
    ]

    for extra_replay_element in self._extra_storage_types:
      storage_elements.append(extra_replay_element)
    return storage_elements

  def get_transition_elements(self, batch_size=None):
    batch_size = self._batch_size if batch_size is None else batch_size

    transition_elements = [
        circular_replay_buffer.ReplayElement('state', 
            (batch_size,) + self._state_shape, self._observation_dtype),
        circular_replay_buffer.ReplayElement('action', (batch_size,), np.int32),
        circular_replay_buffer.ReplayElement('reward', (batch_size,), np.float64),
        circular_replay_buffer.ReplayElement('next_state', 
            (batch_size,) + self._state_shape, self._observation_dtype),
        circular_replay_buffer.ReplayElement('terminal', (batch_size,), np.uint8),
        circular_replay_buffer.ReplayElement('indices', (batch_size,), np.int32)
    ]
    for element in self._extra_storage_types:
      transition_elements.append(
          circular_replay_buffer.ReplayElement(element.name, 
              (batch_size,) + tuple(element.shape), element.type))
    return transition_elements


@gin.configurable(blacklist=['observation_shape', 'stack_size',
                             'update_horizon', 'gamma'])
class WrappedReplayBuffer64(circular_replay_buffer.WrappedReplayBuffer):

  def __init__(self,
               observation_shape,
               stack_size,
               use_staging=True,
               replay_capacity=1000000,
               batch_size=32,
               update_horizon=1,
               gamma=0.99,
               wrapped_memory=None,
               max_sample_attempts=circular_replay_buffer.MAX_SAMPLE_ATTEMPTS,
               extra_storage_types=None,
               observation_dtype=np.uint8):
    if replay_capacity < update_horizon + 1:
      raise ValueError(
          'Update horizon ({}) should be significantly smaller '
          'than replay capacity ({}).'.format(update_horizon, replay_capacity))
    if not update_horizon >= 1:
      raise ValueError('Update horizon must be positive.')
    if not 0.0 <= gamma <= 1.0:
      raise ValueError('Discount factor (gamma) must be in [0, 1].')

    self.batch_size = batch_size

    if wrapped_memory is not None:
      self.memory = wrapped_memory
    else:
      self.memory = OutOfGraphReplayBuffer64(
          observation_shape, stack_size, replay_capacity, batch_size,
          update_horizon, gamma, max_sample_attempts,
          observation_dtype=observation_dtype,
          extra_storage_types=extra_storage_types)

    self.create_sampling_ops(use_staging)