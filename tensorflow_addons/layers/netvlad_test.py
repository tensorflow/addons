# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for NetVLAD layer."""


import pytest
import numpy as np
from tensorflow_addons.layers.netvlad import NetVLAD
from tensorflow_addons.utils import test_utils


pytestmark = pytest.mark.usefixtures("maybe_run_functions_eagerly")


@pytest.mark.parametrize("num_clusters", [1, 4])
def test_simple(num_clusters):
    test_utils.layer_test(
        NetVLAD,
        kwargs={"num_clusters": num_clusters},
        input_shape=(5, 4, 100),
        expected_output_shape=(None, num_clusters * 100),
    )


def test_unknown():
    inputs = np.random.random((5, 4, 100)).astype("float32")
    test_utils.layer_test(
        NetVLAD,
        kwargs={"num_clusters": 3},
        input_shape=(None, None, 100),
        input_data=inputs,
        expected_output_shape=(None, 3 * 100),
    )


def test_invalid_shape():
    with pytest.raises(ValueError) as exception_info:
        test_utils.layer_test(
            NetVLAD, kwargs={"num_clusters": 0}, input_shape=(5, 4, 20)
        )
    assert "`num_clusters` must be greater than 1" in str(exception_info.value)

    with pytest.raises(ValueError) as exception_info:
        test_utils.layer_test(
            NetVLAD, kwargs={"num_clusters": 2}, input_shape=(5, 4, 4, 20)
        )
    assert "must have rank 3" in str(exception_info.value)
