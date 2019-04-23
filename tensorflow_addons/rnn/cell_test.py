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
import tensorflow.keras as keras

from tensorflow_addons.utils import test_utils
from tensorflow_addons.rnn import cell as rnn_cell


@test_utils.run_all_in_graph_and_eager_modes
class NASCellTest(tf.test.TestCase):

    def test_base(self):
        units = 6
        batch_size = 3
        expected_output = np.array(
            [[0.576751, 0.576751, 0.576751, 0.576751, 0.576751, 0.576751],
             [0.618936, 0.618936, 0.618936, 0.618936, 0.618936, 0.618936],
             [0.627393, 0.627393, 0.627393, 0.627393, 0.627393, 0.627393]])
        expected_state = np.array(
            [[
                0.7157977, 0.7157977, 0.7157977, 0.7157977, 0.7157977,
                0.7157977, 0.5767508, 0.5767508, 0.5767508, 0.5767508,
                0.5767508, 0.5767508
            ],
             [
                 0.7804162, 0.7804162, 0.7804162, 0.7804162, 0.7804162,
                 0.7804162, 0.6189357, 0.6189357, 0.6189357, 0.6189357,
                 0.6189357, 0.6189357
             ],
             [
                 0.7945764, 0.7945764, 0.7945764, 0.7945764, 0.7945765,
                 0.7945765, 0.6273934, 0.6273934, 0.6273934, 0.6273934,
                 0.6273934, 0.6273934
             ]])
        const_initializer = tf.constant_initializer(0.5)
        cell = rnn_cell.NASCell(
            units=units,
            kernel_initializer=const_initializer,
            recurrent_initializer=const_initializer)

        inputs = tf.constant(
            np.array([[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]],
                     dtype=np.float32),
            dtype=tf.float32)
        state_value = tf.constant(
            0.1 * np.ones((batch_size, units), dtype=np.float32),
            dtype=tf.float32)
        init_state = [state_value, state_value]
        output, state = cell(inputs, init_state)
        self.evaluate([tf.compat.v1.global_variables_initializer()])
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

    def test_projection(self):
        units = 6
        batch_size = 3
        projection = 5
        expected_output = np.array(
            [[1.697418, 1.697418, 1.697418, 1.697418, 1.697418],
             [1.840037, 1.840037, 1.840037, 1.840037, 1.840037],
             [1.873985, 1.873985, 1.873985, 1.873985, 1.873985]])

        expected_state = np.array(
            [[
                0.69855207, 0.69855207, 0.69855207, 0.69855207, 0.69855207,
                0.69855207, 1.69741797, 1.69741797, 1.69741797, 1.69741797,
                1.69741797
            ],
             [
                 0.77073824, 0.77073824, 0.77073824, 0.77073824, 0.77073824,
                 0.77073824, 1.84003687, 1.84003687, 1.84003687, 1.84003687,
                 1.84003687
             ],
             [
                 0.78973997, 0.78973997, 0.78973997, 0.78973997, 0.78973997,
                 0.78973997, 1.87398517, 1.87398517, 1.87398517, 1.87398517,
                 1.87398517
             ]])
        const_initializer = tf.constant_initializer(0.5)
        cell = rnn_cell.NASCell(
            units=units,
            projection=projection,
            kernel_initializer=const_initializer,
            recurrent_initializer=const_initializer,
            projection_initializer=const_initializer)
        inputs = tf.constant(
            np.array([[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]],
                     dtype=np.float32),
            dtype=tf.float32)
        state_value_c = tf.constant(
            0.1 * np.ones((batch_size, units), dtype=np.float32),
            dtype=tf.float32)
        state_value_h = tf.constant(
            0.1 * np.ones((batch_size, projection), dtype=np.float32),
            dtype=tf.float32)
        init_state = [state_value_c, state_value_h]
        output, state = cell(inputs, init_state)
        self.evaluate([tf.compat.v1.global_variables_initializer()])
        res = self.evaluate([output, state])

        # This is a smoke test: Only making sure expected values not change.
        self.assertLen(res, 2)
        self.assertAllClose(res[0], expected_output)
        # There should be 2 states in the tuple.
        self.assertLen(res[1], 2)
        # Checking the shape of each state to be batch_size * num_units
        new_c, new_h = res[1]
        self.assertEqual(new_c.shape[0], batch_size)
        self.assertEqual(new_c.shape[1], units)
        self.assertEqual(new_h.shape[0], batch_size)
        self.assertEqual(new_h.shape[1], projection)
        self.assertAllClose(np.concatenate(res[1], axis=1), expected_state)

    def test_keras_RNN(self):
        """Tests that NASCell works with keras RNN layer."""
        cell = rnn_cell.NASCell(10)
        seq_input = tf.convert_to_tensor(
            np.random.rand(2, 3, 5), name="seq_input", dtype=tf.float32)
        rnn_layer = keras.layers.RNN(cell=cell)
        rnn_outputs = rnn_layer(seq_input)
        self.evaluate([tf.compat.v1.global_variables_initializer()])
        self.assertEqual(self.evaluate(rnn_outputs).shape, (2, 10))

    def test_config(self):
        cell = rnn_cell.NASCell(10, projection=5, use_bias=True)

        expected_config = {
            "dtype": None,
            "name": "nas_cell",
            "trainable": True,
            "units": 10,
            "projection": 5,
            "use_bias": True,
            "kernel_initializer": "glorot_uniform",
            "recurrent_initializer": "glorot_uniform",
            "bias_initializer": "zeros",
            "projection_initializer": "glorot_uniform",
        }
        config = cell.get_config()
        self.assertEqual(config, expected_config)

        restored_cell = rnn_cell.NASCell.from_config(config)
        restored_config = restored_cell.get_config()
        self.assertEqual(config, restored_config)


@test_utils.run_all_in_graph_and_eager_modes
class LayerNormBasicLSTMCellTest(tf.test.TestCase):

    # NOTE: all the values in the current test case have been calculated.
    def testBasicLSTMCell(self):
        x = tf.ones([1, 2])
        c0 = tf.constant(0.1 * np.asarray([[0, 1]]), dtype=tf.float32)
        h0 = tf.constant(0.1 * np.asarray([[2, 3]]), dtype=tf.float32)
        state0 = [h0, c0]
        c1 = tf.constant(0.1 * np.asarray([[4, 5]]), dtype=tf.float32)
        h1 = tf.constant(0.1 * np.asarray([[6, 7]]), dtype=tf.float32)
        state1 = [h1, c1]
        state = (state0, state1)
        const_initializer = tf.constant_initializer(0.5)
        single_cell = lambda: rnn_cell.LayerNormLSTMCell(
            2,
            kernel_initializer=const_initializer,
            recurrent_initializer=const_initializer)
        cell = keras.layers.StackedRNNCell([single_cell() for _ in range(2)])
        output, output_states = cell(x, state)
        self.evaluate([tf.compat.v1.global_variables_initializer()])
        output_v, output_states_v = self.evaluate([output, output_states])

        expected_output = np.array([[-0.38079708, 0.38079708]])
        expected_state0_h = np.array([[-0.38079708, 0.38079708]])
        expected_state0_c = np.array([[-1.0, 1.0]])
        expected_state1_h = np.array([[-0.38079708, 0.38079708]])
        expected_state1_c = np.array([[-1.0, 1.0]])

        actual_state0_h = output_states_v[0][0]
        actual_state0_c = output_states_v[0][1]
        actual_state1_h = output_states_v[1][0]
        actual_state1_c = output_states_v[1][1]

        self.assertAllClose(output_v, expected_output, 1e-5)
        self.assertAllClose(expected_state0_c, actual_state0_c, 1e-5)
        self.assertAllClose(expected_state0_h, actual_state0_h, 1e-5)
        self.assertAllClose(expected_state1_c, actual_state1_c, 1e-5)
        self.assertAllClose(expected_state1_h, actual_state1_h, 1e-5)

            # with variable_scope.variable_scope(
            #     "other", initializer=init_ops.constant_initializer(0.5)):
            #     x = array_ops.zeros(
            #         [1, 3])  # Test BasicLSTMCell with input_size != num_units.
            #     c = array_ops.zeros([1, 2])
            #     h = array_ops.zeros([1, 2])
            #     state = rnn_cell.LSTMStateTuple(c, h)
            #     cell = contrib_rnn_cell.LayerNormBasicLSTMCell(2)
            #     g, out_m = cell(x, state)
            #     sess.run([variables.global_variables_initializer()])
            #     res = sess.run(
            #         [g, out_m], {
            #             x.name: np.array([[1., 1., 1.]]),
            #             c.name: 0.1 * np.asarray([[0, 1]]),
            #             h.name: 0.1 * np.asarray([[2, 3]]),
            #         })
            #
            #     expected_h = np.array([[-0.38079708, 0.38079708]])
            #     expected_c = np.array([[-1.0, 1.0]])
            #     self.assertEqual(len(res), 2)
            #     self.assertAllClose(res[0], expected_h, 1e-5)
            #     self.assertAllClose(res[1].c, expected_c, 1e-5)
            #     self.assertAllClose(res[1].h, expected_h, 1e-5)

    # def testBasicLSTMCellWithoutNorm(self):
    #     """Tests that BasicLSTMCell with layer_norm=False."""
    #     with self.cached_session() as sess:
    #         with variable_scope.variable_scope(
    #             "root", initializer=init_ops.constant_initializer(0.5)):
    #             x = array_ops.zeros([1, 2])
    #             c0 = array_ops.zeros([1, 2])
    #             h0 = array_ops.zeros([1, 2])
    #             state0 = rnn_cell.LSTMStateTuple(c0, h0)
    #             c1 = array_ops.zeros([1, 2])
    #             h1 = array_ops.zeros([1, 2])
    #             state1 = rnn_cell.LSTMStateTuple(c1, h1)
    #             state = (state0, state1)
    #             single_cell = lambda: contrib_rnn_cell.LayerNormBasicLSTMCell(2, layer_norm=False)  # pylint: disable=line-too-long
    #             cell = rnn_cell.MultiRNNCell([single_cell() for _ in range(2)])
    #             g, out_m = cell(x, state)
    #             sess.run([variables.global_variables_initializer()])
    #             res = sess.run(
    #                 [g, out_m], {
    #                     x.name: np.array([[1., 1.]]),
    #                     c0.name: 0.1 * np.asarray([[0, 1]]),
    #                     h0.name: 0.1 * np.asarray([[2, 3]]),
    #                     c1.name: 0.1 * np.asarray([[4, 5]]),
    #                     h1.name: 0.1 * np.asarray([[6, 7]]),
    #                 })
    #
    #             expected_h = np.array([[0.70230919, 0.72581059]])
    #             expected_state0_c = np.array([[0.8020075, 0.89599884]])
    #             expected_state0_h = np.array([[0.56668288, 0.60858738]])
    #             expected_state1_c = np.array([[1.17500675, 1.26892781]])
    #             expected_state1_h = np.array([[0.70230919, 0.72581059]])
    #
    #             actual_h = res[0]
    #             actual_state0_c = res[1][0].c
    #             actual_state0_h = res[1][0].h
    #             actual_state1_c = res[1][1].c
    #             actual_state1_h = res[1][1].h
    #
    #             self.assertAllClose(actual_h, expected_h, 1e-5)
    #             self.assertAllClose(expected_state0_c, actual_state0_c, 1e-5)
    #             self.assertAllClose(expected_state0_h, actual_state0_h, 1e-5)
    #             self.assertAllClose(expected_state1_c, actual_state1_c, 1e-5)
    #             self.assertAllClose(expected_state1_h, actual_state1_h, 1e-5)
    #
    #         with variable_scope.variable_scope(
    #             "other", initializer=init_ops.constant_initializer(0.5)):
    #             x = array_ops.zeros(
    #                 [1, 3])  # Test BasicLSTMCell with input_size != num_units.
    #             c = array_ops.zeros([1, 2])
    #             h = array_ops.zeros([1, 2])
    #             state = rnn_cell.LSTMStateTuple(c, h)
    #             cell = contrib_rnn_cell.LayerNormBasicLSTMCell(2, layer_norm=False)
    #             g, out_m = cell(x, state)
    #             sess.run([variables.global_variables_initializer()])
    #             res = sess.run(
    #                 [g, out_m], {
    #                     x.name: np.array([[1., 1., 1.]]),
    #                     c.name: 0.1 * np.asarray([[0, 1]]),
    #                     h.name: 0.1 * np.asarray([[2, 3]]),
    #                 })
    #
    #             expected_h = np.array([[0.64121795, 0.68166804]])
    #             expected_c = np.array([[0.88477188, 0.98103917]])
    #             self.assertEqual(len(res), 2)
    #             self.assertAllClose(res[0], expected_h, 1e-5)
    #             self.assertAllClose(res[1].c, expected_c, 1e-5)
    #             self.assertAllClose(res[1].h, expected_h, 1e-5)
    #
    # def testBasicLSTMCellWithStateTuple(self):
    #     with self.cached_session() as sess:
    #         with variable_scope.variable_scope(
    #             "root", initializer=init_ops.constant_initializer(0.5)):
    #             x = array_ops.zeros([1, 2])
    #             c0 = array_ops.zeros([1, 2])
    #             h0 = array_ops.zeros([1, 2])
    #             state0 = rnn_cell.LSTMStateTuple(c0, h0)
    #             c1 = array_ops.zeros([1, 2])
    #             h1 = array_ops.zeros([1, 2])
    #             state1 = rnn_cell.LSTMStateTuple(c1, h1)
    #             cell = rnn_cell.MultiRNNCell(
    #                 [contrib_rnn_cell.LayerNormBasicLSTMCell(2) for _ in range(2)])
    #             h, (s0, s1) = cell(x, (state0, state1))
    #             sess.run([variables.global_variables_initializer()])
    #             res = sess.run(
    #                 [h, s0, s1], {
    #                     x.name: np.array([[1., 1.]]),
    #                     c0.name: 0.1 * np.asarray([[0, 1]]),
    #                     h0.name: 0.1 * np.asarray([[2, 3]]),
    #                     c1.name: 0.1 * np.asarray([[4, 5]]),
    #                     h1.name: 0.1 * np.asarray([[6, 7]]),
    #                 })
    #
    #             expected_h = np.array([[-0.38079708, 0.38079708]])
    #             expected_h0 = np.array([[-0.38079708, 0.38079708]])
    #             expected_c0 = np.array([[-1.0, 1.0]])
    #             expected_h1 = np.array([[-0.38079708, 0.38079708]])
    #             expected_c1 = np.array([[-1.0, 1.0]])
    #
    #             self.assertEqual(len(res), 3)
    #             self.assertAllClose(res[0], expected_h, 1e-5)
    #             self.assertAllClose(res[1].c, expected_c0, 1e-5)
    #             self.assertAllClose(res[1].h, expected_h0, 1e-5)
    #             self.assertAllClose(res[2].c, expected_c1, 1e-5)
    #             self.assertAllClose(res[2].h, expected_h1, 1e-5)
    #
    # def testBasicLSTMCellWithStateTupleLayerNorm(self):
    #     """The results of LSTMCell and LayerNormBasicLSTMCell should be the same."""
    #     with self.cached_session() as sess:
    #         with variable_scope.variable_scope(
    #             "root", initializer=init_ops.constant_initializer(0.5)):
    #             x = array_ops.zeros([1, 2])
    #             c0 = array_ops.zeros([1, 2])
    #             h0 = array_ops.zeros([1, 2])
    #             state0 = rnn_cell_impl.LSTMStateTuple(c0, h0)
    #             c1 = array_ops.zeros([1, 2])
    #             h1 = array_ops.zeros([1, 2])
    #             state1 = rnn_cell_impl.LSTMStateTuple(c1, h1)
    #             cell = rnn_cell_impl.MultiRNNCell([
    #                 contrib_rnn_cell.LayerNormLSTMCell(
    #                     2, layer_norm=True, norm_gain=1.0, norm_shift=0.0)
    #                 for _ in range(2)
    #             ])
    #             h, (s0, s1) = cell(x, (state0, state1))
    #             sess.run([variables.global_variables_initializer()])
    #             res = sess.run(
    #                 [h, s0, s1], {
    #                     x.name: np.array([[1., 1.]]),
    #                     c0.name: 0.1 * np.asarray([[0, 1]]),
    #                     h0.name: 0.1 * np.asarray([[2, 3]]),
    #                     c1.name: 0.1 * np.asarray([[4, 5]]),
    #                     h1.name: 0.1 * np.asarray([[6, 7]]),
    #                 })
    #
    #             expected_h = np.array([[-0.38079708, 0.38079708]])
    #             expected_h0 = np.array([[-0.38079708, 0.38079708]])
    #             expected_c0 = np.array([[-1.0, 1.0]])
    #             expected_h1 = np.array([[-0.38079708, 0.38079708]])
    #             expected_c1 = np.array([[-1.0, 1.0]])
    #
    #             self.assertEqual(len(res), 3)
    #             self.assertAllClose(res[0], expected_h, 1e-5)
    #             self.assertAllClose(res[1].c, expected_c0, 1e-5)
    #             self.assertAllClose(res[1].h, expected_h0, 1e-5)
    #             self.assertAllClose(res[2].c, expected_c1, 1e-5)
    #             self.assertAllClose(res[2].h, expected_h1, 1e-5)
    #
    # def testBasicLSTMCellWithDropout(self):
    #
    #     def _is_close(x, y, digits=4):
    #         delta = x - y
    #         return delta < 10**(-digits)
    #
    #     def _is_close_in(x, items, digits=4):
    #         for i in items:
    #             if _is_close(x, i, digits):
    #                 return True
    #         return False
    #
    #     keep_prob = 0.5
    #     c_high = 2.9998924946
    #     c_low = 0.999983298578
    #     h_low = 0.761552567265
    #     h_high = 0.995008519604
    #     num_units = 5
    #     allowed_low = [1, 2, 3]
    #
    #     with self.cached_session() as sess:
    #         with variable_scope.variable_scope(
    #             "other", initializer=init_ops.constant_initializer(1)):
    #             x = array_ops.zeros([1, 5])
    #             c = array_ops.zeros([1, 5])
    #             h = array_ops.zeros([1, 5])
    #             state = rnn_cell.LSTMStateTuple(c, h)
    #             cell = contrib_rnn_cell.LayerNormBasicLSTMCell(
    #                 num_units, layer_norm=False, dropout_keep_prob=keep_prob)
    #
    #             g, s = cell(x, state)
    #             sess.run([variables.global_variables_initializer()])
    #             res = sess.run(
    #                 [g, s], {
    #                     x.name: np.ones([1, 5]),
    #                     c.name: np.ones([1, 5]),
    #                     h.name: np.ones([1, 5]),
    #                 })
    #
    #             # Since the returned tensors are of size [1,n]
    #             # get the first component right now.
    #             actual_h = res[0][0]
    #             actual_state_c = res[1].c[0]
    #             actual_state_h = res[1].h[0]
    #
    #             # For each item in `c` (the cell inner state) check that
    #             # it is equal to one of the allowed values `c_high` (not
    #             # dropped out) or `c_low` (dropped out) and verify that the
    #             # corresponding item in `h` (the cell activation) is coherent.
    #             # Count the dropped activations and check that their number is
    #             # coherent with the dropout probability.
    #             dropped_count = 0
    #             self.assertTrue((actual_h == actual_state_h).all())
    #             for citem, hitem in zip(actual_state_c, actual_state_h):
    #                 self.assertTrue(_is_close_in(citem, [c_low, c_high]))
    #                 if _is_close(citem, c_low):
    #                     self.assertTrue(_is_close(hitem, h_low))
    #                     dropped_count += 1
    #                 elif _is_close(citem, c_high):
    #                     self.assertTrue(_is_close(hitem, h_high))
    #             self.assertIn(dropped_count, allowed_low)


if __name__ == "__main__":
    tf.test.main()
