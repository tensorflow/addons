# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for CRF (Conditional Random Field) Model Wrapper."""

import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf
from tensorflow_addons.text.crf_wrapper import CRFModelWrapper


def get_test_data():
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
    y = np.array([[1, 2, 2], [1, 1, 1]])  # B-X  I-X  I-X  # B-X  B-X  B-X
    return x, y


@pytest.mark.usefixtures("run_with_mixed_precision_policy")
def test_traing():
    x_np, y_np = get_test_data()
    get_some_model(x_np, y_np)


def get_some_model(x_np, y_np, sanity_check=True):
    x_input = tf.keras.layers.Input(shape=x_np.shape[1:])
    y_outputs = tf.keras.layers.Lambda(lambda x: x)(x_input)
    base_model = tf.keras.Model(x_input, y_outputs)

    model = CRFModelWrapper(base_model, y_np.shape[-1])

    model.compile("adam")
    if sanity_check:
        model.fit(x=x_np, y=y_np)
        model.evaluate(x_np, y_np)
    model.predict(x_np)
    return model


def clone(model: CRFModelWrapper):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "model")
        model.save(file_path)
        new_model = tf.keras.models.load_model(file_path)

    return new_model


def assert_all_equal(array_list1, array_list2):
    for arr1, arr2 in zip(array_list1, array_list2):
        np.testing.assert_equal(np.array(arr1), np.array(arr2))


def test_serialization():

    x_np, y_np = get_test_data()
    model = get_some_model(x_np, y_np, sanity_check=False)

    new_model = clone(model)

    assert_all_equal(model.predict(x_np), new_model.predict(x_np))
    assert_all_equal(model.get_weights(), new_model.get_weights())

    original_loss = model.train_on_batch(x_np, y_np, return_dict=True)["crf_loss"]
    clone_loss = new_model.train_on_batch(x_np, y_np, return_dict=True)["crf_loss"]
    assert_all_equal(model.get_weights(), new_model.get_weights())
    assert original_loss == clone_loss
