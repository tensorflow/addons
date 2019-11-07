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
"""Implements GeLU activation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils import keras_utils
from tensorflow_addons.activations import gelu


@keras_utils.register_keras_custom_object
class GeLU(tf.keras.layers.Layer):
    """Gaussian Error Linear Unit.

    A smoother version of ReLU generally used
    in the BERT or BERT architecture based models.
    Original paper: https://arxiv.org/abs/1606.08415

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.
    """

    def __init__(self, approximate=True, **kwargs):
        super(GeLU, self).__init__(**kwargs)
        self.approximate = approximate
        self.supports_masking = True

    def call(self, inputs):
        return gelu(inputs, approximate=self.approximate)

    def get_config(self):
        config = {'approximate': self.approximate}
        base_config = super(GeLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
