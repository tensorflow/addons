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
"""Implements Thresholded Linear Unit."""

import tensorflow as tf
from typeguard import typechecked

from tensorflow_addons.utils import types


@tf.keras.utils.register_keras_serializable(package="Addons")
class TLU(tf.keras.layers.Layer):
    r"""Thresholded Linear Unit.

    An activation function which is similar to ReLU
    but with a learned threshold that benefits models using FRN(Filter Response
    Normalization). Original paper: https://arxiv.org/pdf/1911.09737.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Arguments:
        affine: `bool`. Whether to make it TLU-Affine or not
            which has the form $\max(x, \alpha*x + \tau)$`
    """

    @typechecked
    def __init__(
        self,
        affine: bool = False,
        tau_initializer: types.Initializer = "zeros",
        tau_regularizer: types.Regularizer = None,
        tau_constraint: types.Constraint = None,
        alpha_initializer: types.Initializer = "zeros",
        alpha_regularizer: types.Regularizer = None,
        alpha_constraint: types.Constraint = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.affine = affine
        self.tau_initializer = tf.keras.initializers.get(tau_initializer)
        self.tau_regularizer = tf.keras.regularizers.get(tau_regularizer)
        self.tau_constraint = tf.keras.constraints.get(tau_constraint)
        if self.affine:
            self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)
            self.alpha_regularizer = tf.keras.regularizers.get(alpha_regularizer)
            self.alpha_constraint = tf.keras.constraints.get(alpha_constraint)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.tau = self.add_weight(
            shape=param_shape,
            name="tau",
            initializer=self.tau_initializer,
            regularizer=self.tau_regularizer,
            constraint=self.tau_constraint,
            synchronization=tf.VariableSynchronization.AUTO,
            aggregation=tf.VariableAggregation.MEAN,
        )
        if self.affine:
            self.alpha = self.add_weight(
                shape=param_shape,
                name="alpha",
                initializer=self.alpha_initializer,
                regularizer=self.alpha_regularizer,
                constraint=self.alpha_constraint,
                synchronization=tf.VariableSynchronization.AUTO,
                aggregation=tf.VariableAggregation.MEAN,
            )

        axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        v = self.alpha * inputs if self.affine else 0
        return tf.maximum(inputs, self.tau + v)

    def get_config(self):
        config = {
            "tau_initializer": tf.keras.initializers.serialize(self.tau_initializer),
            "tau_regularizer": tf.keras.regularizers.serialize(self.tau_regularizer),
            "tau_constraint": tf.keras.constraints.serialize(self.tau_constraint),
            "affine": self.affine,
        }

        if self.affine:
            config["alpha_initializer"] = tf.keras.initializers.serialize(
                self.alpha_initializer
            )
            config["alpha_regularizer"] = tf.keras.regularizers.serialize(
                self.alpha_regularizer
            )
            config["alpha_constraint"] = tf.keras.constraints.serialize(
                self.alpha_constraint
            )

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape
