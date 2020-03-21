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

import sys

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.crf import CRF
from tensorflow_addons.utils import test_utils


def test_unmasked_viterbi_decode():
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
    )

    expected_y = np.array([[1, 2, 2], [1, 1, 1]])  # B-X  I-X  I-X  # B-X  B-X  B-X

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


def get_test_data():
    logits = np.array(
        [
            [[0, 0, 0.5, 0.5, 0.2], [0, 0, 0.3, 0.3, 0.1], [0, 0, 0.9, 10, 1]],
            [[0, 0, 0.2, 0.5, 0.2], [0, 0, 3, 0.3, 0.1], [0, 0, 0.9, 1, 1]],
        ]
    )
    tags = np.array([[2, 3, 4], [3, 2, 2]])

    transitions = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.8, 0.3, 0.1, 0.7, 0.9],
            [-0.3, 2.1, -5.6, 3.4, 4.0],
            [0.2, 0.4, 0.6, -0.3, -0.4],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    boundary_values = np.ones((5,))
    crf_layer = CRF(
        units=5,
        use_kernel=False,  # disable kernel transform
        chain_initializer=tf.keras.initializers.Constant(transitions),
        use_boundary=True,
        boundary_initializer=tf.keras.initializers.Constant(boundary_values),
        name="crf_layer",
    )
    return logits, tags, transitions, boundary_values, crf_layer


def test_keras_model_compile():
    crf_layer = get_test_data()[-1]
    model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(3, 5)), crf_layer])

    model.compile(loss=None, optimizer="adam")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
