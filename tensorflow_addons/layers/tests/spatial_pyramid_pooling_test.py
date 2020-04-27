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
"""Tests for AdaptivePooling layers."""

import pytest
from tensorflow_addons.layers.spatial_pyramid_pooling import SpatialPyramidPooling2D


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_spp_shape_2d():
    spp = SpatialPyramidPooling2D([1, 3, 5])
    output_shape = [256, 35, 64]
    assert spp.compute_output_shape([256, None, None, 64]).as_list() == output_shape
