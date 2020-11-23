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
"""Discriminative Layer Training Optimizer for TensorFlow."""

from typing import List, Union

import tensorflow as tf
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class MultiOptimizer(tf.keras.optimizers.Optimizer):
    """Multi Optimizer Wrapper for Discriminative Layer Training.

    Creates a wrapper around a set of instantiated optimizer layer pairs.
    Generally useful for transfer learning of deep networks.

    Each optimizer will optimize only the weights associated with its paired layer.
    This can be used to implement discriminative layer training by assigning
    different learning rates to each optimizer layer pair.
    `(tf.keras.optimizers.Optimizer, List[tf.keras.layers.Layer])` pairs are also supported.
    Please note that the layers must be instantiated before instantiating the optimizer.

    Args:
        optimizers_and_layers: a list of tuples of an optimizer and a layer or model.
            Each tuple should contain exactly 1 instantiated optimizer and 1 object that
            subclasses `tf.keras.Model`, `tf.keras.Sequential` or `tf.keras.layers.Layer`.
            Nested layers and models will be automatically discovered.
            Alternatively, in place of a single layer, you can pass a list of layers.
        optimizer_specs: specialized list for serialization.
            Should be left as None for almost all cases.
            If you are loading a serialized version of this optimizer,
            please use `tf.keras.models.load_model` after saving a model compiled with this optimizer.

    Usage:

    >>> model = tf.keras.Sequential([
    ...     tf.keras.Input(shape=(4,)),
    ...     tf.keras.layers.Dense(8),
    ...     tf.keras.layers.Dense(16),
    ...     tf.keras.layers.Dense(32),
    ... ])
    >>> optimizers = [
    ...     tf.keras.optimizers.Adam(learning_rate=1e-4),
    ...     tf.keras.optimizers.Adam(learning_rate=1e-2)
    ... ]
    >>> optimizers_and_layers = [(optimizers[0], model.layers[0]), (optimizers[1], model.layers[1:])]
    >>> optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    >>> model.compile(optimizer=optimizer, loss="mse")

    Reference:
        - [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
        - [Collaborative Layer-wise Discriminative Learning in Deep Neural Networks](https://arxiv.org/abs/1607.05440)

    Note: Currently, `tfa.optimizers.MultiOptimizer` does not support callbacks that modify optimizers.
        However, you can instantiate optimizer layer pairs with
        `tf.keras.optimizers.schedules.LearningRateSchedule`
        instead of a static learning rate.

    This code should function on CPU, GPU, and TPU. Apply with `tf.distribute.Strategy().scope()` context as you
    would with any other optimizer.
    """

    @typechecked
    def __init__(
        self,
        optimizers_and_layers: Union[list, None] = None,
        optimizer_specs: Union[list, None] = None,
        name: str = "MultiOptimizer",
        **kwargs
    ):

        super(MultiOptimizer, self).__init__(name, **kwargs)

        if optimizer_specs is None and optimizers_and_layers is not None:
            self.optimizer_specs = [
                self.create_optimizer_spec(optimizer, layers_or_model)
                for optimizer, layers_or_model in optimizers_and_layers
            ]

        elif optimizer_specs is not None and optimizers_and_layers is None:
            self.optimizer_specs = [
                self.maybe_initialize_optimizer_spec(spec) for spec in optimizer_specs
            ]

        else:
            raise RuntimeError(
                "Must specify one of `optimizers_and_layers` or `optimizer_specs`."
            )

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Wrapped apply_gradient method.

        Returns an operation to be executed.
        """

        for spec in self.optimizer_specs:
            spec["gv"] = []

        for grad, var in tuple(grads_and_vars):
            for spec in self.optimizer_specs:
                for name in spec["weights"]:
                    if var.name == name:
                        spec["gv"].append((grad, var))

        return tf.group(
            [
                spec["optimizer"].apply_gradients(spec["gv"], **kwargs)
                for spec in self.optimizer_specs
            ]
        )

    def get_config(self):
        config = super(MultiOptimizer, self).get_config()
        config.update({"optimizer_specs": self.optimizer_specs})
        return config

    @classmethod
    def create_optimizer_spec(
        cls,
        optimizer: tf.keras.optimizers.Optimizer,
        layers_or_model: Union[
            tf.keras.Model,
            tf.keras.Sequential,
            tf.keras.layers.Layer,
            List[tf.keras.layers.Layer],
        ],
    ):
        """Creates a serializable optimizer spec.

        The name of each variable is used rather than `var.ref()` to enable serialization and deserialization.
        """
        if isinstance(layers_or_model, list):
            weights = [
                var.name for sublayer in layers_or_model for var in sublayer.weights
            ]
        else:
            weights = [var.name for var in layers_or_model.weights]

        return {
            "optimizer": optimizer,
            "weights": weights,
        }

    @classmethod
    def maybe_initialize_optimizer_spec(cls, optimizer_spec):
        if isinstance(optimizer_spec["optimizer"], dict):
            optimizer_spec["optimizer"] = tf.keras.optimizers.deserialize(
                optimizer_spec["optimizer"]
            )

        return optimizer_spec

    def __repr__(self):
        return "Multi Optimizer with %i optimizer layer pairs" % len(
            self.optimizer_specs
        )
