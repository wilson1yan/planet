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

from planet import tools
from planet.models import base


class SSM(base.Base):
  """Gaussian state space model.

  Implements the transition function and encoder using feed forward networks.

  Prior:    Posterior:

  (a)       (a)
     \         \
      v         v
  (s)->(s)  (s)->(s)
                  ^
                  :
                 (o)
  """

  def __init__(
      self, state_size, embed_size, activation=tf.nn.elu):
    self._state_size = state_size
    self._embed_size = embed_size
    super(SSM, self).__init__(
        tf.make_template('transition', self._transition),
        tf.make_template('posterior', self._posterior))
    self._kwargs = dict(units=self._embed_size, activation=activation)

  @property
  def state_size(self):
    return {
        'state': self._state_size,
    }

  def features_from_state(self, state):
    """Extract features for the decoder network from a prior or posterior."""
    return state['state']

  def cpc_from_states(self, posterior, prior):
    # posterior B x T x H, prior B x T x H
    z_pos, z_next = posterior['state'][:, :, 1:], prior['state'][:, :, 1:]
    b, t, h = tools.shape(z_pos)
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
    inputs = tf.concat([prev_state['state'], prev_action], -1)
    hidden = tf.layers.dense(inputs, **self._kwargs)
    mean = tf.layers.dense(hidden, self._state_size, None)
    return {
        'state': mean,
    }

  def _posterior(self, prev_state, prev_action, obs):
    """Compute posterior state from previous state and current observation."""
    inputs = obs
    hidden = tf.layers.dense(inputs, **self._kwargs)
    mean = tf.layers.dense(hidden, self._state_size, None)
    return {
        'state': mean,
    }
