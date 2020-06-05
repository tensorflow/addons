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
"""Tests for SpatialPyramidPooling layers"""

import pytest
import numpy as np

import tensorflow as tf
from tensorflow_addons.layers.spatial_pyramid_pooling import SpatialPyramidPooling2D
from tensorflow_addons.utils import test_utils


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_spp_shape_2d():
    spp = SpatialPyramidPooling2D([1, 3, 5])
    output_shape = [256, 35, 64]
    assert spp.compute_output_shape([256, None, None, 64]).as_list() == output_shape

    spp = SpatialPyramidPooling2D([1, 3, 5], data_format="channels_first")
    output_shape = [256, 64, 35]
    assert spp.compute_output_shape([256, 64, None, None]).as_list() == output_shape


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_spp_output_2d():
    inputs = np.arange(start=0.0, stop=16.0, step=1.0).astype(np.float32)
    inputs = np.reshape(inputs, (1, 4, 4, 1))
    output = np.array([[[7.5], [2.5], [4.5], [10.5], [12.5]]]).astype(np.float32)
    test_utils.layer_test(
        SpatialPyramidPooling2D,
        kwargs={"bins": [[1, 1], [2, 2]], "data_format": "channels_last"},
        input_data=inputs,
        expected_output=output,
    )

    inputs = np.arange(start=0.0, stop=16.0, step=1.0).astype(np.float32)
    inputs = np.reshape(inputs, (1, 1, 4, 4))
    output = np.array([[[7.5, 2.5, 4.5, 10.5, 12.5]]]).astype(np.float32)
    test_utils.layer_test(
        SpatialPyramidPooling2D,
        kwargs={"bins": [[1, 1], [2, 2]], "data_format": "channels_first"},
        input_data=inputs,
        expected_output=output,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_serialization():
    layer = SpatialPyramidPooling2D([[1, 1], [3, 3]])
    serialized_layer = tf.keras.layers.serialize(layer)
    new_layer = tf.keras.layers.deserialize(serialized_layer)
    assert layer.get_config() == new_layer.get_config()


def test_keras(tmpdir):
    test_inputs = np.arange(start=0.0, stop=16.0, step=1.0).astype(np.float32)
    test_inputs = np.reshape(test_inputs, (1, 4, 4, 1))
    test_output = [[[7.5], [2.5], [4.5], [10.5], [12.5]]]

    inputs = tf.keras.layers.Input((None, None, 1))
    spp = SpatialPyramidPooling2D([1, 2])(inputs)
    model = tf.keras.Model(inputs=[inputs], outputs=[spp])

    model_path = str(tmpdir / "spp_model.h5")
    model.save(model_path)
    model = tf.keras.models.load_model(model_path)
    model_output = model.predict(test_inputs).tolist()
    assert model_output == test_output
