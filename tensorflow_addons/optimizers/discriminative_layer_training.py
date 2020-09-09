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

import tensorflow as tf
from typeguard import typechecked
from typing import Union
from tensorflow.keras.optimizers import Optimizer

# python -m flake8 tensorflow_addons/optimizers/discriminative_layer_training.py
# python -m black tensorflow_addons/optimizers/discriminative_layer_training.py


class FakeVar:
    def __init__(self, name):
        # probably can be refactored out
        self.name = name


@tf.keras.utils.register_keras_serializable(package="Addons")
class MultiOpt(Optimizer):
    @typechecked
    def __init__(
        self,
        optimizer_layer_pairs: Union[list, None] = None,
        optimizer_specs: Union[list, None] = None,
        name: str = "MultiOpt",
        **kwargs
    ):

        """
        Creates a wrapper around a set of instantiated optimizer layer pairs.

        Each optimizer will optimize only the weights associated with its paired layer. This can be used
        to implement discriminative layer training by assigning different learning rates to each optimizer
        layer pair. (Optimizer, list(Layers)) pairs are also supported.

        Currently, MultiOpt does not support callbacks that modify optimizers. However, you can instantiate
        optimizer layer pairs with tf.keras.optimizers.schedules.LearningRateSchedule instead of a static learning
        rate.

        This code should function on CPU, GPU, and TPU.

        Example:

        ```python

        model = get_model()

        opt1 = tf.keras.optimizers.Adam(learning_rate=1e-4)
        opt2 = tf.keras.optimizers.Adam(learning_rate=1e-2)

        opt_layer_pairs = [(opt1, model.layers[0]),
                            (opt2, model.layers[1:])]

        loss = tf.keras.losses.MSE
        optimizer = MultiOpt(opt_layer_pairs)

        model.compile(optimizer=optimizer, loss = loss)

        model.fit(x,y)

        ```


        """

        super(MultiOpt, self).__init__(name, **kwargs)

        if optimizer_specs is None and optimizer_layer_pairs is not None:
            self.optimizer_specs = [
                self.create_optimizer_spec(opt, layer)
                for opt, layer in optimizer_layer_pairs
            ]

        elif optimizer_specs is not None and optimizer_layer_pairs is None:
            self.optimizer_specs = optimizer_specs

        else:
            raise RuntimeError(
                "You must specify either an list of optimizer_layer_pairs or a list of optimizer_specs"
            )

        self.initialized_optimizer_specs = [
            self.initialize_from_optimizer_spec(spec) for spec in self.optimizer_specs
        ]

        self.lr = self.initialized_optimizer_specs[0]["optimizer"].lr

    def apply_gradients(self, grads_and_vars, name=None):
        """
        Wrapped Gradient Apply method. Returns a list of tf ops to be executed.
        """

        for spec in self.optimizer_specs:
            spec["gv"] = []

        for grad, var in tuple(grads_and_vars):
            for spec in self.optimizer_specs:
                for weight in spec["weights"]:
                    if var.name == weight.name:
                        spec["gv"].append((grad, var))

        return [
            spec["optimizer"].apply_gradients(spec["gv"])
            for spec in self.optimizer_specs
        ]

    def get_config(self):
        config = super(MultiOpt, self).get_config()
        config.update({"optimizer_specs": self.optimizer_specs})
        return config

    @classmethod
    def initialize_from_optimizer_spec(cls, optimizer_spec):
        optimizer_spec["optimizer"] = optimizer_spec["optimizer_class"].from_config(
            optimizer_spec["optimizer_config"]
        )
        return optimizer_spec

    @classmethod
    def create_optimizer_spec(cls, optimizer_instance, layer):
        if type(layer) == list:
            weights = [
                FakeVar(var.name) for sublayer in layer for var in sublayer.weights
            ]
        else:
            weights = [FakeVar(var.name) for var in layer.weights]

        return {
            "optimizer_class": optimizer_instance.__class__,
            "optimizer_config": optimizer_instance.get_config(),
            "weights": weights,
        }
