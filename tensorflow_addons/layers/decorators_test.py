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
"""Tests for Decorators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.layers import decorators
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class RecomputeGradTest(tf.test.TestCase):

  def testRecomputeGrad(self):

    m = tf.keras.Sequential()
    for _ in range(3):
      m.add(tf.keras.layers.BatchNormalization())
      m.add(tf.keras.layers.Conv1D(10, 1, use_bias=False))
      m.add(tf.keras.layers.ReLU())

    def fn(x):
      return m(x)

    @decorators.recompute_grad
    def fn_recompute(x):
      return m(x)

    @decorators.recompute_grad(use_data_dep=True)
    def fn_use_data_dep(x):
      return m(x)

    @decorators.recompute_grad(tupleize_grads=True)
    def fn_tupleize(x):
      return m(x)

    @decorators.recompute_grad(use_data_dep=True, tupleize_grads=True)
    def fn_both(x):
      return m(x)

    x = tf.random.uniform((3, 1, 3))

    names_and_fns = [
        ("recompute", fn_recompute),
        ("regular", fn),
        ("use_data_dep", fn_use_data_dep),
        ("tupleize", fn_tupleize),
        ("tuple_and_data_dep", fn_both),
    ]
    outputs_and_vars = []
    all_grads = []
    for name, wrapped_fn in names_and_fns:
      with tf.GradientTape() as tape:
        out = tf.math.reduce_sum(wrapped_fn(x))
      outputs_and_vars.append((out, m.trainable_variables))
      all_grads.append(tape.gradient(out, m.trainable_variables))

    outputs = list(zip(*outputs_and_vars))[0]

    # All outputs are the same
    current = outputs[0]
    for out in outputs[1:]:
      self.assertAllClose(current, out)
      current = out

    # All gradients are the same
    for grads in zip(all_grads):
      current = grads[0]
      for g in grads[1:]:
        self.assertAllClose(current, g)
        current = g

  def testDoubleCall(self):

    dense = tf.keras.layers.Dense(2)

    @decorators.recompute_grad
    def layer_with_recompute(inputs):
      return dense(inputs)

    with tf.GradientTape() as tape:
      inputs = tf.ones((2, 4), tf.float32)
      out1 = layer_with_recompute(inputs)
      out2 = layer_with_recompute(inputs) + out1
      out = tf.math.reduce_sum(out2)

    tvars = dense.trainable_variables
    assert len(tvars) == 2
    grads = tape.gradient(out, tvars)
    for grad in grads:
      self.assertTrue(grad is not None)

  def testWithIsRecomputeKwarg(self):

    kwarg_values = []

    dense = tf.keras.layers.Dense(2)
    batch_normalization = tf.keras.layers.BatchNormalization()

    @decorators.recompute_grad
    def layer_with_recompute(inputs, is_recomputing=False):
      kwarg_values.append(is_recomputing)

      out = dense(inputs)
      out = batch_normalization(out, training=True)
      # TODO: What to do with this? How do we keep the BatchNormalization
      # layer from updating the statistics but still using only batch statistics?
      #if is_recomputing:
      #  # Ensure that the updates are not duplicated by popping off the latest
      #  # 2 additions.
      #  update_ops = ops.get_collection_ref(ops.GraphKeys.UPDATE_OPS)
      #  update_ops.pop()
      #  update_ops.pop()
      return out

    with tf.GradientTape() as tape:
      x = tf.ones((2, 4), tf.float32)
      y = layer_with_recompute(x)
      loss = tf.math.reduce_sum(y)
    tvars = dense.trainable_variables + \
            batch_normalization.trainable_variables
    tape.gradient(loss, tvars)

    # TODO: What to do with this?
    #update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
    #self.assertEqual(2, len(update_ops))
    self.assertEqual([False, True], kwarg_values)

  def testWithoutVariables(self):

    def concat_n(layer_list, num_inputs):
      return tf.reduce_sum(
          tf.concat([x for x in layer_list[-num_inputs:]], axis=-1),
          axis=1, keepdims=True)

    @decorators.recompute_grad
    def concat_n_wrap(*args):
      return concat_n(args, 3)

    with tf.GradientTape() as tape:
      # DenseNet-style layers
      layer_list = [tf.random.uniform((4, 8))]
      for _ in range(5):
        layer_list.append(tf.sqrt(concat_n_wrap(*layer_list)))
    grads = tape.gradient(layer_list[-1], layer_list[0])

  def testErrorOnClosedOverTensor(self):
    x = tf.random.uniform((4, 8))
    y = tf.random.uniform((4, 8))
    z = x * y

    with self.assertRaisesWithPredicateMatch(ValueError, "closes over"):
      @decorators.recompute_grad
      def fn_with_capture(a):  # pylint: disable=unused-variable
        return a * z


if __name__ == "__main__":
  tf.test.main()
