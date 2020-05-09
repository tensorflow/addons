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
"""Tests for octave convolutional layers."""

import numpy as np

import tensorflow as tf
from tensorflow_addons.layers.octave_convolutional import OctaveConv1D


def test_octave_conv1d_padding_output_shape():
    # verify output_shape with padding
    kwargs = {"filters": 3, "kernel_size": 3, "low_freq_ratio": 0.5, "padding": "same"}
    layer = OctaveConv1D(**kwargs)
    y = layer(np.ones((1, 10, 2)))
    assert (y[0].shape.as_list(), y[1].shape.as_list()) == ([1, 10, 2], [1, 5, 1])


def test_octave_conv1d_dilation_rate_output_shape():
    # verify output_shape with dilation rate
    kwargs = {
        "filters": 3,
        "kernel_size": 3,
        "low_freq_ratio": 0.5,
        "dilation_rate": 2,
    }
    layer = OctaveConv1D(**kwargs)
    y = layer(np.ones((1, 20, 4)))
    assert (y[0].shape.as_list(), y[1].shape.as_list()) == ([1, 20, 2], [1, 10, 1])


def test_octave_conv1d_strides_output_shape():
    # verify output_shape with strides
    kwargs = {
        "filters": 3,
        "kernel_size": 3,
        "low_freq_ratio": 0.5,
        "strides": 2,
    }
    layer = OctaveConv1D(**kwargs)
    y = layer(np.ones((1, 20, 4)))
    assert (y[0].shape.as_list(), y[1].shape.as_list()) == ([1, 10, 2], [1, 5, 1])


def test_octave_conv1d_regularizers():
    # verify regularizers
    kwargs = {
        "filters": 3,
        "kernel_size": 3,
        "padding": "same",
        "low_freq_ratio": 0.5,
        "kernel_regularizer": "l2",
        "bias_regularizer": "l2",
        "activity_regularizer": "l2",
        "strides": 1,
    }
    layer = OctaveConv1D(**kwargs)
    layer.build((None, 10, 4))
    assert len(layer.losses) == 4
    layer(np.ones((1, 10, 4)))
    assert len(layer.losses) == 8


def test_octave_conv1d_constraints():
    # verify constraints
    def identity(x):
        return x

    k_constraint = identity
    b_constraint = identity

    kwargs = {
        "filters": 3,
        "kernel_size": 3,
        "padding": "same",
        "low_freq_ratio": 0.5,
        "kernel_constraint": k_constraint,
        "bias_constraint": b_constraint,
        "strides": 1,
    }
    layer = OctaveConv1D(**kwargs)
    layer.build((None, 10, 4))
    # list of 2 kernels: one for self.conv_high_to_high and the other
    # for self.conv_high_to_low
    assert len(layer.kernel) == 2
    assert len(layer.bias) == 2
    assert layer.kernel[0].constraint, k_constraint
    assert layer.bias[0].constraint, b_constraint


def test_ocatve_conv1d_output():
    # verify numerical values with simple example case
    kwargs = {
        "filters": 3,
        "kernel_size": 3,
        "low_freq_ratio": 0.5,
        "kernel_initializer": "ones",
    }
    layer = OctaveConv1D(**kwargs)
    y = layer(np.ones((1, 2, 2)))
    assert len(y) == 2
    np.testing.assert_allclose([[[4.0, 4.0], [4.0, 4.0]]], y[0])
    np.testing.assert_allclose([[[2.0]]], y[1])


def test_octave_conv1d_serialization():
    # verify serialization
    kwargs = {"filters": 3, "kernel_size": 3, "low_freq_ratio": 0.5, "padding": "same"}
    layer = OctaveConv1D(**kwargs)
    serialized_layer = tf.keras.layers.serialize(layer)
    new_layer = tf.keras.layers.deserialize(serialized_layer)
    assert layer.get_config() == new_layer.get_config()
