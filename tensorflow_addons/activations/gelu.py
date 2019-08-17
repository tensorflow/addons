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
"""Implementing GELU activation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow_addons.utils import keras_utils


@tf.function
@keras_utils.register_keras_custom_object
def gelu(x, dtype='float32'):
    """Gaussian Error Linear Unit.

    A smoother version of ReLU generally used 
    in the BERT or BERT architecture based models.
    Original paper: https://arxiv.org/abs/1606.08415
    
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """

    x  = K.cast(x, dtype=dtype)
    pi = K.cast(math.pi, dtype=dtype)
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / pi) * \
             (x + 0.044715 * tf.pow(x, 3))))