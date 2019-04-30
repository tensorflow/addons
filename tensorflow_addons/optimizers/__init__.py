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
"""Additional optimizers that conform to Keras API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_addons.optimizers.lazy_adam import LazyAdam
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW
from tensorflow_addons.optimizers.weight_decay_optimizers import SGDW
from tensorflow_addons.optimizers.weight_decay_optimizers import (
    extend_with_decoupled_weight_decay)
from tensorflow_addons.optimizers.moving_average import MovingAverage
