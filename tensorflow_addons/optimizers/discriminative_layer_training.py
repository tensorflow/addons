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

from typing import Union

import tensorflow as tf
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class MultiOptimizer(tf.keras.optimizers.Optimizer):
    """Multi Optimizer Wrapper for Discriminative Layer Training.

    Creates a wrapper around a set of instantiated optimizer layer pairs. Generally useful for transfer learning
    of deep networks.

    Each optimizer will optimize only the weights associated with its paired layer. This can be used
    to implement discriminative layer training by assigning different learning rates to each optimizer
    layer pair. (Optimizer, list(Layers)) pairs are also supported. Please note that the layers must be
    instantiated before instantiating the optimizer.

    Args:
        optimizers_and_layers: a list of tuples of an optimizer and a layer or model. Each tuple should contain
            exactly 1 instantiated optimizer and 1 object that subclasses tf.keras.Model or tf.keras.Layer. Nested
            layers and models will be automatically discovered. Alternatively, in place of a single layer, you can pass
            a list of layers.
        optimizer_specs: specialized list for serialization. Should be left as None for almost all cases. If you are
            loading a serialized version of this optimizer, please use tf.keras.models.load_model after saving a
            model compiled with this optimizer.

    Usage:

    ```python
    model = get_model()

    opt1 = tf.keras.optimizers.Adam(learning_rate=1e-4)
    opt2 = tf.keras.optimizers.Adam(learning_rate=1e-2)

    opt_layer_pairs = [(opt1, model.layers[0]), (opt2, model.layers[1:])]

    loss = tf.keras.losses.MSE
    optimizer = tfa.optimizers.MultiOpt(opt_layer_pairs)

    model.compile(optimizer=optimizer, loss = loss)

    model.fit(x,y)
    '''

    Reference:

    [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146)
    [Collaborative Layer-wise Discriminative Learning in Deep Neural Networks](https://arxiv.org/abs/1607.05440)

    Notes:

    Currently, MultiOpt does not support callbacks that modify optimizers. However, you can instantiate
    optimizer layer pairs with tf.keras.optimizers.schedules.LearningRateSchedule instead of a static learning
    rate.

    This code should function on CPU, GPU, and TPU. Apply the with strategy.scope() context as you
    would with any other optimizer.

    """

    @typechecked
    def __init__(
        self,
        optimizers_and_layers: Union[list, None] = None,
        optimizer_specs: Union[list, None] = None,
        name: str = "MultiOptimzer",
        **kwargs
    ):

        super(MultiOptimizer, self).__init__(name, **kwargs)

        if optimizer_specs is None and optimizers_and_layers is not None:
            self.optimizer_specs = [
                self.create_optimizer_spec(opt, layer)
                for opt, layer in optimizers_and_layers
            ]

        elif optimizer_specs is not None and optimizers_and_layers is None:
            self.optimizer_specs = [
                self.maybe_initialize_optimizer_spec(spec) for spec in optimizer_specs
            ]

        else:
            raise RuntimeError(
                "You must specify either an list of optimizers and layers or a list of optimizer_specs"
            )

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        """Wrapped apply_gradient method.

        Returns a list of tf ops to be executed.
        Name of variable is used rather than var.ref() to enable serialization and deserialization.
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
    def create_optimizer_spec(cls, optimizer_instance, layer):

        assert isinstance(
            optimizer_instance, tf.keras.optimizers.Optimizer
        ), "Object passed is not an instance of tf.keras.optimizers.Optimizer"

        assert isinstance(layer, tf.keras.layers.Layer) or isinstance(
            layer, tf.keras.Model
        ), "Object passed is not an instance of tf.keras.layers.Layer nor tf.keras.Model"

        if type(layer) == list:
            weights = [var.name for sublayer in layer for var in sublayer.weights]
        else:
            weights = [var.name for var in layer.weights]

        return {
            "optimizer": optimizer_instance,
            "weights": weights,
        }

    @classmethod
    def maybe_initialize_optimizer_spec(cls, optimizer_spec):
        if type(optimizer_spec["optimizer"]) == dict:
            optimizer_spec["optimizer"] = tf.keras.optimizers.deserialize(
                optimizer_spec["optimizer"]
            )

        return optimizer_spec

    def __repr__(self):
        return "Multi Optimizer with %i optimizer layer pairs" % len(
            self.optimizer_specs
        )
