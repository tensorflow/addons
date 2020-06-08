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
"""Tests for PolynomialCrossing layer."""


import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.polynomial import PolynomialCrossing


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_full_matrix():
    x0 = np.asarray([[0.1, 0.2, 0.3]]).astype(np.float32)
    x = np.asarray([[0.4, 0.5, 0.6]]).astype(np.float32)
    layer = PolynomialCrossing(projection_dim=None, kernel_initializer="ones")
    output = layer([x0, x])
    np.testing.assert_allclose([[0.55, 0.8, 1.05]], output)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_low_rank_matrix():
    x0 = np.asarray([[0.1, 0.2, 0.3]]).astype(np.float32)
    x = np.asarray([[0.4, 0.5, 0.6]]).astype(np.float32)
    layer = PolynomialCrossing(projection_dim=1, kernel_initializer="ones")
    output = layer([x0, x])
    np.testing.assert_allclose([[0.55, 0.8, 1.05]], output)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_invalid_proj_dim():
    with pytest.raises(ValueError, match="should be smaller than last_dim / 2"):
        x0 = np.random.random((12, 5))
        x = np.random.random((12, 5))
        layer = PolynomialCrossing(projection_dim=6)
        layer([x0, x])


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_invalid_inputs():
    with pytest.raises(ValueError, match="must be a tuple or list of size 2"):
        x0 = np.random.random((12, 5))
        x = np.random.random((12, 5))
        x1 = np.random.random((12, 5))
        layer = PolynomialCrossing(projection_dim=6)
        layer([x0, x, x1])


def test_serialization():
    layer = PolynomialCrossing(projection_dim=None)
    serialized_layer = tf.keras.layers.serialize(layer)
    new_layer = tf.keras.layers.deserialize(serialized_layer)
    assert layer.get_config() == new_layer.get_config()


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_diag_scale():
    x0 = np.asarray([[0.1, 0.2, 0.3]]).astype(np.float32)
    x = np.asarray([[0.4, 0.5, 0.6]]).astype(np.float32)
    layer = PolynomialCrossing(
        projection_dim=None, diag_scale=1.0, kernel_initializer="ones"
    )
    output = layer([x0, x])
    np.testing.assert_allclose([[0.59, 0.9, 1.23]], output)
