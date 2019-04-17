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
"""Tests for RNN cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.utils import test_utils
from tensorflow_addons.rnn import cell as rnn_cell


@test_utils.run_all_in_graph_and_eager_modes
class RNNCellTest(tf.test.TestCase):

    def testNASCell(self):
        units = 6
        batch_size = 3
        expected_output = np.array(
            [[0.576751, 0.576751, 0.576751, 0.576751, 0.576751, 0.576751],
             [0.618936, 0.618936, 0.618936, 0.618936, 0.618936, 0.618936],
             [0.627393, 0.627393, 0.627393, 0.627393, 0.627393, 0.627393]])
        expected_state = np.array([[
            0.7157977, 0.7157977, 0.7157977, 0.7157977, 0.7157977, 0.7157977,
            0.5767508, 0.5767508, 0.5767508, 0.5767508, 0.5767508, 0.5767508
        ], [
            0.7804162, 0.7804162, 0.7804162, 0.7804162, 0.7804162, 0.7804162,
            0.6189357, 0.6189357, 0.6189357, 0.6189357, 0.6189357, 0.6189357
        ], [
            0.7945764, 0.7945764, 0.7945764, 0.7945764, 0.7945765, 0.7945765,
            0.6273934, 0.6273934, 0.6273934, 0.6273934, 0.6273934, 0.6273934
        ]])
        with self.cached_session():
            cell = rnn_cell.NASCell(units=units)
            inputs = tf.constant(
                np.array(
                    [[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]],
                    dtype=np.float32),
                dtype=tf.float32)
            state_value = tf.constant(
                0.1 * np.ones((batch_size, units), dtype=np.float32),
                dtype=tf.float32)
            init_state = [state_value, state_value]
            output, state = cell(inputs, init_state)
            # self.evaluate([tf.variables.global_variables_initializer()])
            res = self.evaluate([output, state])

            # This is a smoke test: Only making sure expected values not change.
            self.assertLen(res, 2)
            self.assertAllClose(res[0], expected_output)
            # There should be 2 states in the list.
            self.assertLen(res[1], 2)
            # Checking the shape of each state to be batch_size * num_units
            new_c, new_h = res[1]
            self.assertEqual(new_c.shape[0], batch_size)
            self.assertEqual(new_c.shape[1], units)
            self.assertEqual(new_h.shape[0], batch_size)
            self.assertEqual(new_h.shape[1], units)
            self.assertAllClose(np.concatenate(res[1], axis=1), expected_state)

# def testNASCellProj(self):
#   num_units = 6
#   batch_size = 3
#   num_proj = 5
#   expected_output = np.array(
#       [[1.697418, 1.697418, 1.697418, 1.697418,
#         1.697418], [1.840037, 1.840037, 1.840037, 1.840037, 1.840037],
#        [1.873985, 1.873985, 1.873985, 1.873985, 1.873985]])
#   expected_state = np.array([[
#       0.69855207, 0.69855207, 0.69855207, 0.69855207, 0.69855207, 0.69855207,
#       1.69741797, 1.69741797, 1.69741797, 1.69741797, 1.69741797
#   ], [
#       0.77073824, 0.77073824, 0.77073824, 0.77073824, 0.77073824, 0.77073824,
#       1.84003687, 1.84003687, 1.84003687, 1.84003687, 1.84003687
#   ], [
#       0.78973997, 0.78973997, 0.78973997, 0.78973997, 0.78973997, 0.78973997,
#       1.87398517, 1.87398517, 1.87398517, 1.87398517, 1.87398517
#   ]])
#   with self.cached_session() as sess:
#     with variable_scope.variable_scope(
#         "nas_proj_test", initializer=init_ops.constant_initializer(0.5)):
#       cell = contrib_rnn_cell.NASCell(num_units=num_units, num_proj=num_proj)
#       inputs = constant_op.constant(
#           np.array(
#               [[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]],
#               dtype=np.float32),
#           dtype=dtypes.float32)
#       state_value_c = constant_op.constant(
#           0.1 * np.ones((batch_size, num_units), dtype=np.float32),
#           dtype=dtypes.float32)
#       state_value_h = constant_op.constant(
#           0.1 * np.ones((batch_size, num_proj), dtype=np.float32),
#           dtype=dtypes.float32)
#       init_state = rnn_cell.LSTMStateTuple(state_value_c, state_value_h)
#       output, state = cell(inputs, init_state)
#       sess.run([variables.global_variables_initializer()])
#       res = sess.run([output, state])
#
#       # This is a smoke test: Only making sure expected values not change.
#       self.assertEqual(len(res), 2)
#       self.assertAllClose(res[0], expected_output)
#       # There should be 2 states in the tuple.
#       self.assertEqual(len(res[1]), 2)
#       # Checking the shape of each state to be batch_size * num_units
#       new_c, new_h = res[1]
#       self.assertEqual(new_c.shape[0], batch_size)
#       self.assertEqual(new_c.shape[1], num_units)
#       self.assertEqual(new_h.shape[0], batch_size)
#       self.assertEqual(new_h.shape[1], num_proj)
#       self.assertAllClose(np.concatenate(res[1], axis=1), expected_state)
#
# @test_util.run_in_graph_and_eager_modes
# def testNASCellKerasRNN(self):
#   """Tests that NASCell works with keras RNN layer."""
#   cell = contrib_rnn_cell.NASCell(10)
#   seq_input = ops.convert_to_tensor(
#       np.random.rand(2, 3, 5), name="seq_input", dtype=dtypes.float32)
#   rnn_layer = keras_layers.RNN(cell=cell)
#   rnn_outputs = rnn_layer(seq_input)
#   self.evaluate([variables.global_variables_initializer()])
#   self.assertEqual(self.evaluate(rnn_outputs).shape, (2, 10))
#
# def testUGRNNCell(self):
#   num_units = 2
#   batch_size = 3
#   expected_state_and_output = np.array(
#       [[0.13752282, 0.13752282], [0.10545051, 0.10545051],
#        [0.10074195, 0.10074195]],
#       dtype=np.float32)
#   with self.cached_session() as sess:
#     with variable_scope.variable_scope(
#         "ugrnn_cell_test", initializer=init_ops.constant_initializer(0.5)):
#       cell = contrib_rnn_cell.UGRNNCell(num_units=num_units)
#       inputs = constant_op.constant(
#           np.array(
#               [[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]],
#               dtype=np.float32),
#           dtype=dtypes.float32)
#       init_state = constant_op.constant(
#           0.1 * np.ones((batch_size, num_units), dtype=np.float32),
#           dtype=dtypes.float32)
#       output, state = cell(inputs, init_state)
#       sess.run([variables.global_variables_initializer()])
#       res = sess.run([output, state])
#       # This is a smoke test: Only making sure expected values didn't change.
#       self.assertEqual(len(res), 2)
#       self.assertAllClose(res[0], expected_state_and_output)
#       self.assertAllClose(res[1], expected_state_and_output)