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

from tensorflow_addons.optimizers.average_wrapper import AveragedOptimizerWrapper
from tensorflow_addons.optimizers.conditional_gradient import ConditionalGradient
from tensorflow_addons.optimizers.cyclical_learning_rate import CyclicalLearningRate
from tensorflow_addons.optimizers.cyclical_learning_rate import (
    TriangularCyclicalLearningRate,
)
from tensorflow_addons.optimizers.cyclical_learning_rate import (
    Triangular2CyclicalLearningRate,
)
from tensorflow_addons.optimizers.cyclical_learning_rate import (
    ExponentialCyclicalLearningRate,
)
from tensorflow_addons.optimizers.lamb import LAMB
from tensorflow_addons.optimizers.lazy_adam import LazyAdam
from tensorflow_addons.optimizers.lookahead import Lookahead
from tensorflow_addons.optimizers.moving_average import MovingAverage
from tensorflow_addons.optimizers.novograd import NovoGrad
from tensorflow_addons.optimizers.proximal_adagrad import ProximalAdagrad
from tensorflow_addons.optimizers.rectified_adam import RectifiedAdam
from tensorflow_addons.optimizers.stochastic_weight_averaging import SWA
from tensorflow_addons.optimizers.weight_decay_optimizers import AdamW
from tensorflow_addons.optimizers.weight_decay_optimizers import SGDW
from tensorflow_addons.optimizers.weight_decay_optimizers import (
    extend_with_decoupled_weight_decay,
)
from tensorflow_addons.optimizers.yogi import Yogi
