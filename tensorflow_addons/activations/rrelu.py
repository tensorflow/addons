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
from tensorflow_addons.utils import keras_utils
from tensorflow_addons.utils.resource_loader import get_path_to_datafile

_activation_ops_so = tf.load_op_library(
    get_path_to_datafile("custom_ops/activations/_activation_ops.so"))


@keras_utils.register_keras_custom_object
@tf.function
def rrelu(x, lower=0.125, upper=0.3333333333333333, training=None, seed=None):
    """rrelu function.

    Computes rrelu function:
    `x if x > 0 else random(lower, upper) * x` or
    `x if x > 0 else x * (lower + upper) / 2`
    depending on whether training is enabled.

    See [Empirical Evaluation of Rectified Activations in Convolutional Network](https://arxiv.org/abs/1505.00853).

    Args:
        x: A `Tensor`. Must be one of the following types:
            `float16`, `float32`, `float64`.
        lower: `float`, lower bound for random alpha.
        upper: `float`, upper bound for random alpha.
        training: `bool`, indicating whether the `call`
        is meant for training or inference.
        seed: `int`, this sets the operation-level seed.
    Returns:
        result: A `Tensor`. Has the same type as `x`.
    """
    x = tf.convert_to_tensor(x)
    if training is None:
        training = tf.keras.backend.learning_phase()
        training = bool(tf.keras.backend.get_value(training))
    # TODO: get rid of v1 API
    seed1, seed2 = tf.compat.v1.random.get_seed(seed)
    result, _ = _activation_ops_so.addons_rrelu(x, lower, upper, training,
                                                seed1, seed2)
    return result


@tf.RegisterGradient("Addons>Rrelu")
def _rrelu_grad(op, *grad):
    return _activation_ops_so.addons_rrelu_grad(grad[0], op.inputs[0],
                                                op.outputs[1])
