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

from tensorflow_addons.layers.crf import CRF, CRFLossLayer
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


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_keras_model_inference():
    logits, _, _, _, crf_layer = get_test_data()

    input_tensor = tf.keras.layers.Input(shape=(3, 5))
    decoded_sequence, _, _, _ = crf_layer(input_tensor)
    model = tf.keras.Model(input_tensor, decoded_sequence)

    model.predict(logits)
    model(logits).numpy()


def test_in_subclass_model():
    tf.config.experimental_run_functions_eagerly(True)
    train_x = np.array(
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

    train_y = np.array([[1, 2, 2], [1, 1, 1]])  # B-X  I-X  I-X  # B-X  B-X  B-X

    x_input = tf.keras.layers.Input(shape=train_x.shape[1:])
    y_input = tf.keras.layers.Input(shape=train_y.shape[1:])

    decoded_sequence, potentials, sequence_length, chain_kernel = CRF(5)(x_input)
    inference_model = tf.keras.Model(x_input, decoded_sequence)

    crf_loss = CRFLossLayer()([potentials, y_input, sequence_length, chain_kernel])

    training_model = tf.keras.Model([x_input, y_input], crf_loss)

    training_model.compile("adam", loss="mae", run_eagerly=True)
    training_model.fit((train_x, train_y), y=np.zeros((2,)))
    training_model.evaluate((train_x, train_y), y=np.zeros((2,)))

    inference_model.predict(train_x)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
