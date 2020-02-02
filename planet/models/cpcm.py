# Copyright 2019 The PlaNet Authors. All rights reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from planet import tools
from planet.models import base


class CPCM(base.Base):
  """CPC Model

  The stochastic latent is computed from the hidden state at the same time
  step. If an observation is present, the posterior latent is compute from both
  the hidden state and the observation.

  Prior:    Posterior:

  (a)       (a)
     \         \
      v         v
  [h]->[h]  [h]->[h]
      ^ |       ^ :
     /  v      /  v
  (s)  (s)  (s)  (s)
                  ^
                  :
                 (o)
  """

  def __init__(
      self, state_size, belief_size, embed_size,
      future_rnn=True, activation=tf.nn.elu,
      num_layers=1, n_neg=128):
    self._state_size = state_size
    self._belief_size = belief_size
    self._embed_size = embed_size
    self._future_rnn = future_rnn
    self._cell = tf.contrib.rnn.GRUBlockCell(self._belief_size)
    self._kwargs = dict(units=self._embed_size, activation=activation)
    self._num_layers = num_layers
    self._n_neg = n_neg
    super(CPCM, self).__init__(
        tf.make_template('transition', self._transition),
        tf.make_template('posterior', self._posterior))

  @property
  def state_size(self):
    return {
        'state': self._state_size,
        'belief': self._belief_size,
        'rnn_state': self._belief_size,
    }

  def features_from_state(self, state):
    """Extract features for the decoder network from a prior or posterior."""
    return tf.concat([state['state'], state['belief']], -1)

  def cpc_from_states(self, posterior, prior):
    # posterior B x T x H, prior B x T x H
    b, t, h = tools.shape(posterior['state'])
    z_pos, z_next = posterior['state'], prior['state']
    z_pos = tf.reshape(z_pos, [b * t, h]) # B * T x H
    z_next = tf.reshape(z_next, [b * t, h]) # B * T x H
    pos_log_density = -tf.reduce_sum(tf.square(z_pos - z_next), axis=-1, keepdims=True) # B * T x 1
    dot_product = tf.matmul(z_next, z_pos, transpose_b=True) # B * T x B * T
    z_next_sqnorm = tf.reduce_sum(z_next ** 2, axis=-1, keepdims=True) # B * T x 1
    z_pos_sqnorm = tf.expand_dims(tf.reduce_sum(z_pos ** 2, axis=-1), axis=0) # 1 x B * T
    neg_log_density = -(z_next_sqnorm - 2 * dot_product + z_pos_sqnorm) # B * T x B * T
    neg_log_density = neg_log_density - 1e10 * tf.eye(b * t)

    log_density = tf.concat((neg_log_density, pos_log_density), axis=1) # B * T x B * T + 1
    log_density = tf.nn.log_softmax(log_density, axis=1)
    loss = -tf.reduce_mean(log_density[:, -1])
    return loss

  def _transition(self, prev_state, prev_action, zero_obs):
    """Compute prior next state by applying the transition dynamics."""
    hidden = tf.concat([prev_state['state'], prev_action], -1)
    for _ in range(self._num_layers):
      hidden = tf.layers.dense(hidden, **self._kwargs)
    belief, rnn_state = self._cell(hidden, prev_state['rnn_state'])
    if self._future_rnn:
      hidden = belief
    for _ in range(self._num_layers):
      hidden = tf.layers.dense(hidden, **self._kwargs)
    mean = tf.layers.dense(hidden, self._state_size, None)
    return {
        'state': mean,
        'belief': belief,
        'rnn_state': rnn_state,
    }

  def _posterior(self, prev_state, prev_action, obs):
    """Compute posterior state from previous state and current observation."""
    prior = self._transition_tpl(prev_state, prev_action, tf.zeros_like(obs))
    hidden = tf.concat([prior['belief'], obs], -1)
    for _ in range(self._num_layers):
      hidden = tf.layers.dense(hidden, **self._kwargs)
    mean = tf.layers.dense(hidden, self._state_size, None)
    return {
        'state': mean,
        'belief': prior['belief'],
        'rnn_state': prior['rnn_state'],
    }
