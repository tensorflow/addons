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
"""Tests for AdaptivePooling2D layer."""

import sys

import pytest
import numpy as np
import tensorflow as tf
from tensorflow_addons.layers.adaptive_pooling import AdaptiveAveragePooling2D
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class AdaptiveAveragePooling2DTest(tf.test.TestCase):
    def test(self):
        valid_input = np.arange(start=0.0, stop=40.0, step=1.0).astype(np.float32)
        valid_input = np.reshape(valid_input, (1, 4, 10, 1))
        model = tf.keras.models.Sequential()
        model.add(AdaptiveAveragePooling2D(2, 2))
        outputs = model.predict(valid_input, steps=1)
        result = np.squeeze(outputs).tolist()
        self.assertEqual(result, [[7.0, 12.0], [27.0, 32.0]])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
