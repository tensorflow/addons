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

import tensorflow as tf

from tensorflow_addons.utils import types


@tf.keras.utils.register_keras_serializable(package="Addons")
def snake(inputs: types.TensorLike, freq: types.Number = 1) -> tf.Tensor:
    """Snake activation to learn periodic functions.

    https://arxiv.org/abs/2006.08195
    """

    inputs = tf.convert_to_tensor(inputs)
    freq = tf.cast(freq, inputs.dtype)

    return inputs + (1 - tf.cos(2 * freq * inputs)) / (2 * freq)
