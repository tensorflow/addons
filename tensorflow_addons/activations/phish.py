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

import tensorflow as tf

from tensorflow_addons.utils.types import TensorLike


@tf.keras.utils.register_keras_serializable(package="Addons")
def phish(x: TensorLike) -> tf.Tensor:
    
    """
    Phish is defined as f(x) = xTanH(GELU(x)) with no discontinuities in the f(x) derivative.
    More details on the activation function can be found at DOI: https://www.techrxiv.org/articles/preprint/Phish_A_Novel_Hyper-Optimizable_Activation_Function/17283824
    """
    
    x = tf.convert_to_tensor(x)
    return x * tf.math.tanh(tf.nn.gelu(x))
