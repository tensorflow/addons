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
"""Tests for GeLU activation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow_addons.layers.gelu import GeLU
from tensorflow_addons.utils import test_utils
from tensorflow_addons.utils.test_utils import keras_parameterized
from absl.testing import parameterized


@parameterized.parameters([np.float16, np.float32, np.float64])
@test_utils.run_all_in_graph_and_eager_modes
class TestGeLU(tf.test.TestCase):
    def test_random(self, dtype):
        x = np.array([[0.5, 1.2, -0.3]]).astype(dtype)
        val = np.array([[0.345714, 1.0617027, -0.11462909]]).astype(dtype)
     
        test_utils.layer_test(
            GeLU,
            kwargs={'dtype': dtype},
            input_data=x,
            expected_output=val)


@keras_parameterized.run_all_keras_modes
@keras_parameterized.run_with_all_model_types
class TestGeLU_v2(keras_parameterized.TestCase):
	def test_layer_random(self):
		layer = tf.keras.layers.Dense(1, activation=GeLU())
		model = keras_parameterized.testing_utils.get_model_from_layers([layer], 
                                                              input_shape=(10,))
		model.compile(
			'sgd',
			'mse',
			run_eagerly=keras_parameterized.testing_utils.should_run_eagerly())
		model.fit(np.ones((10, 10)), np.ones((10, 1)), batch_size=2)

if __name__ == '__main__':
    tf.test.main()