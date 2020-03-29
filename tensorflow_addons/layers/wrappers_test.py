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
# =============================================================================

import sys
import os
import tempfile
from absl.testing import parameterized

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.layers import wrappers
from tensorflow_addons.utils import test_utils


def test_basic():
    test_utils.layer_test(
        wrappers.WeightNormalization,
        kwargs={"layer": tf.keras.layers.Conv2D(5, (2, 2)),},
        input_shape=(2, 4, 4, 3),
    )


def test_no_bias():
    test_utils.layer_test(
        wrappers.WeightNormalization,
        kwargs={"layer": tf.keras.layers.Dense(5, use_bias=False),},
        input_shape=(2, 4),
    )


@test_utils.run_all_in_graph_and_eager_modes
class WeightNormalizationTest(tf.test.TestCase, parameterized.TestCase):
    def _check_data_init(self, data_init, input_data, expected_output):
        layer = tf.keras.layers.Dense(
            input_data.shape[-1],
            activation=None,
            kernel_initializer="identity",
            bias_initializer="zeros",
        )
        test_utils.layer_test(
            wrappers.WeightNormalization,
            kwargs={"layer": layer, "data_init": data_init,},
            input_data=input_data,
            expected_output=expected_output,
        )

    def test_with_data_init_is_false(self):
        input_data = np.array([[[-4, -4], [4, 4]]], dtype=np.float32)
        self._check_data_init(
            data_init=False, input_data=input_data, expected_output=input_data
        )

    def test_with_data_init_is_true(self):
        input_data = np.array([[[-4, -4], [4, 4]]], dtype=np.float32)
        self._check_data_init(
            data_init=True, input_data=input_data, expected_output=input_data / 4
        )

    def test_non_layer(self):
        images = tf.random.uniform((2, 4, 3))
        with self.assertRaises(AssertionError):
            wrappers.WeightNormalization(images)

    def test_non_kernel_layer(self):
        images = tf.random.uniform((2, 2, 2))
        with self.assertRaisesRegexp(ValueError, "contains a `kernel`"):
            non_kernel_layer = tf.keras.layers.MaxPooling2D(2, 2)
            wn_wrapper = wrappers.WeightNormalization(non_kernel_layer)
            wn_wrapper(images)

    def test_with_time_dist(self):
        batch_shape = (8, 8, 16, 16, 3)
        inputs = tf.keras.layers.Input(batch_shape=batch_shape)
        a = tf.keras.layers.Conv2D(3, 3)
        b = wrappers.WeightNormalization(a)
        out = tf.keras.layers.TimeDistributed(b)(inputs)
        tf.keras.Model(inputs, out)

    @parameterized.named_parameters(
        ["Dense", lambda: tf.keras.layers.Dense(1), False],
        ["SimpleRNN", lambda: tf.keras.layers.SimpleRNN(1), True],
        ["Conv2D", lambda: tf.keras.layers.Conv2D(3, 1), False],
        ["LSTM", lambda: tf.keras.layers.LSTM(1), True],
    )
    def test_serialization(self, base_layer, rnn):
        base_layer = base_layer()
        wn_layer = wrappers.WeightNormalization(base_layer, not rnn)
        new_wn_layer = tf.keras.layers.deserialize(tf.keras.layers.serialize(wn_layer))
        self.assertEqual(wn_layer.data_init, new_wn_layer.data_init)
        self.assertEqual(wn_layer.is_rnn, new_wn_layer.is_rnn)
        self.assertEqual(wn_layer.is_rnn, rnn)
        if not isinstance(base_layer, tf.keras.layers.LSTM):
            # Issue with LSTM serialization, check with TF-core
            # Before serialization: tensorflow.python.keras.layers.recurrent_v2.LSTM
            # After serialization: tensorflow.python.keras.layers.recurrent.LSTM
            self.assertTrue(isinstance(new_wn_layer.layer, base_layer.__class__))

    @parameterized.named_parameters(
        ["Dense", lambda: tf.keras.layers.Dense(1), [1]],
        ["SimpleRNN", lambda: tf.keras.layers.SimpleRNN(1), [None, 10]],
        ["Conv2D", lambda: tf.keras.layers.Conv2D(3, 1), [3, 3, 1]],
        ["LSTM", lambda: tf.keras.layers.LSTM(1), [10, 10]],
    )
    def test_model_build(self, base_layer_fn, input_shape):
        inputs = tf.keras.layers.Input(shape=input_shape)
        for data_init in [True, False]:
            base_layer = base_layer_fn()
            wt_layer = wrappers.WeightNormalization(base_layer, data_init)
            model = tf.keras.models.Sequential(layers=[inputs, wt_layer])
            model.build()

    @parameterized.named_parameters(
        ["Dense", lambda: tf.keras.layers.Dense(1), [1]],
        ["SimpleRNN", lambda: tf.keras.layers.SimpleRNN(1), [10, 10]],
        ["Conv2D", lambda: tf.keras.layers.Conv2D(3, 1), [3, 3, 1]],
        ["LSTM", lambda: tf.keras.layers.LSTM(1), [10, 10]],
    )
    def test_save_file_h5(self, base_layer, input_shape):
        base_layer = base_layer()
        wn_conv = wrappers.WeightNormalization(base_layer)
        model = tf.keras.Sequential(layers=[wn_conv])
        model.build([None] + input_shape)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_weights(os.path.join(tmp_dir, "wrapper_test_model.h5"))

    @parameterized.named_parameters(
        ["Dense", lambda: tf.keras.layers.Dense(1), [1]],
        ["SimpleRNN", lambda: tf.keras.layers.SimpleRNN(1), [10, 10]],
        ["Conv2D", lambda: tf.keras.layers.Conv2D(3, 1), [3, 3, 1]],
        ["LSTM", lambda: tf.keras.layers.LSTM(1), [10, 10]],
    )
    def test_forward_pass(self, base_layer, input_shape):
        sample_data = np.ones([1] + input_shape, dtype=np.float32)
        base_layer = base_layer()
        base_output = base_layer(sample_data)
        wn_layer = wrappers.WeightNormalization(base_layer, False)
        wn_output = wn_layer(sample_data)
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.assertAllClose(self.evaluate(base_output), self.evaluate(wn_output))


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("data_init", [True, False])
@pytest.mark.parametrize(
    "base_layer_fn, input_shape",
    [
        (lambda: tf.keras.layers.Dense(1), [1]),
        (lambda: tf.keras.layers.SimpleRNN(1), [10, 10]),
        (lambda: tf.keras.layers.Conv2D(3, 1), [3, 3, 1]),
        (lambda: tf.keras.layers.LSTM(1), [10, 10]),
    ],
)
def test_removal(base_layer_fn, input_shape, data_init):
    sample_data = np.ones([1] + input_shape, dtype=np.float32)

    base_layer = base_layer_fn()
    wn_layer = wrappers.WeightNormalization(base_layer, data_init)
    wn_output = wn_layer(sample_data)
    wn_removed_layer = wn_layer.remove()
    wn_removed_output = wn_removed_layer(sample_data)
    np.testing.assert_allclose(wn_removed_output.numpy(), wn_output.numpy())
    assert isinstance(wn_removed_layer, base_layer.__class__)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
