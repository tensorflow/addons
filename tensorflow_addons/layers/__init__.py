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
"""Additional layers that conform to Keras API."""

from tensorflow_addons.layers.adaptive_pooling import (
    AdaptiveAveragePooling1D,
    AdaptiveMaxPooling1D,
    AdaptiveAveragePooling2D,
    AdaptiveMaxPooling2D,
    AdaptiveAveragePooling3D,
    AdaptiveMaxPooling3D,
)
from tensorflow_addons.layers.gelu import GELU
from tensorflow_addons.layers.max_unpooling_2d import MaxUnpooling2D
from tensorflow_addons.layers.maxout import Maxout
from tensorflow_addons.layers.multihead_attention import MultiHeadAttention
from tensorflow_addons.layers.normalizations import FilterResponseNormalization
from tensorflow_addons.layers.normalizations import GroupNormalization
from tensorflow_addons.layers.normalizations import InstanceNormalization
from tensorflow_addons.layers.optical_flow import CorrelationCost
from tensorflow_addons.layers.poincare import PoincareNormalize
from tensorflow_addons.layers.polynomial import PolynomialCrossing
from tensorflow_addons.layers.snake import Snake
from tensorflow_addons.layers.sparsemax import Sparsemax
from tensorflow_addons.layers.spectral_normalization import SpectralNormalization
from tensorflow_addons.layers.spatial_pyramid_pooling import SpatialPyramidPooling2D
from tensorflow_addons.layers.tlu import TLU
from tensorflow_addons.layers.wrappers import WeightNormalization
from tensorflow_addons.layers.esn import ESN
from tensorflow_addons.layers.stochastic_depth import StochasticDepth
from tensorflow_addons.layers.noisy_dense import NoisyDense
from tensorflow_addons.layers.crf import CRF
