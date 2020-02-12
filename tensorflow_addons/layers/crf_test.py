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
"""Tests for Conditional Random Field layer."""

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class TestCRF(tf.test.TestCase):
    def test_unmasked_viterbi_decode(self):
        x = np.array(
            [
                [
                    # O   B-X  I-X  B-Y  I-Y
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                ],
                [
                    # O   B-X  I-X  B-Y  I-Y
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                ],
            ]
        )  # yapf: disable

        expected_y = np.array(
            [[1, 2, 2], [1, 1, 1]]  # B-X  I-X  I-X  # B-X  B-X  B-X
        )  # yapf: disable

        transitions = np.ones([5, 5])
        boundary_value = np.ones(5)

        test_utils.layer_test(
            CRF,
            kwargs={
                "units": 5,
                "use_kernel": False,  # disable kernel transform
                "chain_initializer": tf.keras.initializers.Constant(transitions),
                "use_boundary": True,
                "boundary_initializer": tf.keras.initializers.Constant(boundary_value),
            },
            input_data=x,
            expected_output=expected_y,
            expected_output_dtype=tf.int32,
            validate_training=False,
        )


if __name__ == "__main__":
    tf.test.main()
