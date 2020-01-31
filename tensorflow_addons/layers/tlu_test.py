# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TLU activation."""

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_addons.layers.tlu import TLU
from tensorflow_addons.utils import test_utils


@parameterized.parameters([np.float16, np.float32, np.float64])
@test_utils.run_all_in_graph_and_eager_modes
class TestTLU(tf.test.TestCase):
    def test_random(self, dtype):
        x = np.array([[-2.5, 0.0, 0.3]]).astype(dtype)
        val = np.array([[0.0, 0.0, 0.3]]).astype(dtype)
        test_utils.layer_test(
            TLU, kwargs={"dtype": dtype}, input_data=x, expected_output=val
        )

    def test_affine(self, dtype):
        x = np.array([[-2.5, 0.0, 0.3]]).astype(dtype)
        val = np.array([[-1.5, 1.0, 1.3]]).astype(dtype)
        test_utils.layer_test(
            TLU,
            kwargs={
                "affine": True,
                "dtype": dtype,
                "alpha_initializer": "ones",
                "tau_initializer": "ones",
            },
            input_data=x,
            expected_output=val,
        )

    def test_serialization(self, dtype):
        tlu = TLU(
            affine=True, alpha_initializer="ones", tau_initializer="ones", dtype=dtype
        )
        serialized_tlu = tf.keras.layers.serialize(tlu)
        new_layer = tf.keras.layers.deserialize(serialized_tlu)
        self.assertEqual(tlu.get_config(), new_layer.get_config())


if __name__ == "__main__":
    tf.test.main()
