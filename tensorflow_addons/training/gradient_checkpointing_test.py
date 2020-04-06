# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Tests for methods implementing gradient checkpointing."""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow_addons.training import recompute_sequential
from tensorflow.keras import layers

def _get_simple_cnn_model(img_dim, n_channels):
    model = tf.keras.Sequential([
    layers.Reshape(
        target_shape=[img_dim, img_dim, n_channels],
        input_shape=(img_dim, img_dim, n_channels)),
    layers.Conv2D(3, 3, padding='same', activation=tf.nn.relu),
    layers.MaxPooling2D((1, 1), padding='same'),
    layers.Flatten(),
    layers.Dense(4, activation=tf.nn.relu),
    layers.Dense(2)])
    return model

def _compute_loss(logits, labels):
  return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels))

def _model_fn(model, x):
    return model(x)

def test_recompute_sequential_forward_pass():
    img_dim = 2
    n_channels = 1
    bs = 1 
    x = tf.random.uniform([bs,img_dim,img_dim,n_channels])
    model = _get_simple_cnn_model(img_dim, n_channels)
    logits_no_recompute = _model_fn(model, x)
    recompute_model_fn = recompute_sequential(_model_fn)
    logits_with_recompute = recompute_model_fn(model, x, _watch_vars=model.trainable_variables)
    np.testing.assert_allclose(logits_no_recompute, logits_with_recompute)

def test_recompute_sequential_gradients():
    img_dim = 2
    n_channels = 1
    bs = 1 
    x = tf.random.uniform([bs,img_dim,img_dim,n_channels])
    y = tf.ones([bs], dtype=tf.int64)
    model = _get_simple_cnn_model(img_dim, n_channels)
    recompute_model_fn = recompute_sequential(_model_fn)
    with tf.GradientTape() as tape: 
            logits_no_recompute = _model_fn(model, x)
            loss_no_recompute  = _compute_loss(logits_no_recompute, y)
    grads_no_recompute = tape.gradient(loss_no_recompute, model.trainable_variables)
    del tape
    with tf.GradientTape() as tape: 
            logits_with_recompute = recompute_model_fn(model, x, _watch_vars=model.trainable_variables)
            loss_with_recompute = _compute_loss(logits_with_recompute, y)
    grads_with_recompute = tape.gradient(loss_with_recompute, model.trainable_variables)
    del tape 
    assert(len(grads_no_recompute) == len(grads_with_recompute))
    for i in range(len(grads_no_recompute)):
        np.testing.assert_allclose(grads_no_recompute[i], grads_with_recompute[i])