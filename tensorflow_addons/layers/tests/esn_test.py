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
"""Tests for Echo State recurrent Network (ESN)."""

import pytest
import numpy as np
import tensorflow as tf
from tensorflow_addons.layers.esn import ESN
from tensorflow_addons.utils import test_utils


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def layer_test_esn(dtype):
    inp = np.asanyarray(
        [[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0, 3.0]]]
    ).astype(dtype)
    out = np.asarray([[2.5, 2.5, 2.5], [4.5, 4.5, 4.5], [6.5, 6.5, 6.5]]).astype(dtype)

    const_initializer = tf.constant_initializer(0.5)
    kwargs = {
        "units": 3,
        "connectivity": 1,
        "leaky": 1,
        "spectral_radius": 0.9,
        "use_norm2": True,
        "use_bias": True,
        "activation": None,
        "kernel_initializer": const_initializer,
        "recurrent_initializer": const_initializer,
        "bias_initializer": const_initializer,
        "dtype": dtype,
    }

    test_utils.layer_test(ESN, kwargs=kwargs, input_data=inp, expected_output=out)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_serialization(dtype):
    esn = ESN(
        units=3,
        connectivity=1,
        leaky=1,
        spectral_radius=0.9,
        use_norm2=False,
        use_bias=True,
        activation=None,
        kernel_initializer="ones",
        recurrent_initializer="ones",
        bias_initializer="ones",
    )
    serialized_esn = tf.keras.layers.serialize(esn)
    new_layer = tf.keras.layers.deserialize(serialized_esn)
    assert esn.get_config() == new_layer.get_config()
