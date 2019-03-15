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

from tensorflow_addons.activations.sparsemax import sparsemax
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
class Sparsemax(tf.keras.layers.Layer):
    """Sparsemax activation function [1].

    The ouput shape is the same as the input shape.

    [1]: https://arxiv.org/abs/1602.02068

    Arguments:
        axis: Integer, axis along which the sparsemax normalization is applied.
    """

    def __init__(self, axis=-1, **kwargs):
        super(Sparsemax, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs):
        return sparsemax(inputs, axis=self.axis)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Sparsemax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
