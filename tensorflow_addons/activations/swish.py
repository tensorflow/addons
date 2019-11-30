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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import math_ops


@tf.keras.utils.register_keras_serializable(package='Addons')
@tf.function
def swish(x):
    """Computes the Swish activation function: `x * sigmoid(x)`.

    Source: "Searching for Activation Functions" (Ramachandran et al. 2017)
    https://arxiv.org/abs/1710.05941
    """
    x = tf.convert_to_tensor(x)
    return x * math_ops.sigmoid(x)
