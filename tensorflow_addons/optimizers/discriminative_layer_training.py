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
"""Discriminative Layer Training Manager for TensorFlow."""

import tensorflow as tf
import numpy as np
from typeguard import typechecked
import logging


class ModelManager:
    """Class for grouping functions related to model lr_mult management"""

    def _get_layers(self, layer):
        """Helper method to access a layer's sublayers as a list or return an empty list
        """

        try:
            return layer.layers
        except AttributeError:
            return []

    def _get_lr_mult(self, layer):

        """Helper method to access a layer's learning rate multiplier, which defaults to 1 if lr mult is not set
        """

        try:
            return layer.lr_mult
        except AttributeError:
            return 1.0

    def _assign_lr_mult(self, layer, lr_mult, override=False):

        """Helper method to assign a layer's learning rate multiplier, which does nothing if lr mult is already set
        """

        try:
            if layer.lr_mult and override:
                layer.lr_mult = lr_mult  # check if layer has lr mult and if override, then assign the new lr mult
        except AttributeError:
            layer.lr_mult = lr_mult  # since layer has no lr mult, assign the mult

    def _get_lowest_layers(self, layer, propagate_lr_mult_to_sub_layers=True):

        """Helper method iterate through all nested layers of an object that behaves like a layer or model
        By default, we want to propagate the lr mult to the lower layers.
        tbh I can't properly explain how this works so see this post

        https://stackoverflow.com/questions/6340351/iterating-through-list-of-list-in-python
        """

        mult = self._get_lr_mult(layer)
        layers = self._get_layers(layer)

        if len(layers) > 0:
            for sublayer in layers:

                # we generally want to propagate the lr mult to the lower layers
                if propagate_lr_mult_to_sub_layers:
                    self._assign_lr_mult(sublayer, mult)

                # recursively iterate through the nested layers
                for nested_sublayer in self._get_lowest_layers(sublayer):
                    yield nested_sublayer

        else:
            yield layer

    def _apply_lr_mult_to_var(self, layer):
        """Helper method to apply the lr mult to the trainable variables of a layer
        """

        lr_mult = self._get_lr_mult(layer)

        for var in layer.trainable_variables:
            var.lr_mult = tf.convert_to_tensor(lr_mult, tf.float32)  # 0D tensor
            var.lr_mult_value = (
                lr_mult  # easier to check vars lr mult in graph and eager
            )

    def _check_for_lr_mult(self, layer, verbose=True, propagate=True):
        """Identify which layers have an LR mult not equal to 1
        """

        layers_with_lr_mult = []

        for sub_layer in self._get_lowest_layers(
            layer, propagate_lr_mult_to_sub_layers=propagate
        ):
            lr_mult = self._get_lr_mult(sub_layer)
            if lr_mult != 1.0:
                layers_with_lr_mult.append(sub_layer)
                if verbose:
                    logging.info("layer %s lr_mult : %f" % (sub_layer.name, lr_mult))

        return layers_with_lr_mult

    def _compute_params(self, var_list):
        """helps compute params to provide a summary that aligns with model.summary()
        """
        return np.sum([np.prod(list(var.shape)) for var in var_list])

    def _prepare_model(self, model, verbose=True):
        """Prepares a built model for disc training
        """

        layers_with_lr_mult = self._check_for_lr_mult(model, verbose=verbose)
        if len(layers_with_lr_mult) == 0:

            logging.warning(
                "Discriminative Layer Training requires an lr_mult attribute on at least one layer"
            )

            logging.warning(
                "The assigned lr_mult must not be equal to 1. eg: model.layers[0].lr_mult = 0.01"
            )

        for layer in self._get_lowest_layers(model):
            self._apply_lr_mult_to_var(layer)

        vars_with_lr_mult = [
            var for var in model.trainable_variables if var.lr_mult_value != 1.0
        ]

        if verbose:
            tf.print(
                "%i params of %i will learn at a different rate"
                % (
                    self._compute_params(vars_with_lr_mult),
                    self._compute_params(model.trainable_variables),
                )
            )


class DiscriminativeWrapper(tf.keras.optimizers.Optimizer):
    @typechecked
    def __init__(
        self,
        base_optimizer: object,
        model: tf.keras.Model,
        learning_rate: float,
        verbose: bool = True,
        name: str = "discrim_opt",
        *args,
        **kwargs
    ):
        """Discriminative Layer Training Wrapper

            Discriminative layer training is a technique that applies different learning rates to
            different layers in a model. Generally, a lower learning rate is applied to the
            layers closest to the input and a higher learning rate is applied to layers closer
            to the output. This method helps in transfer learning by quickly calibrating the head
            of a model while preserving the useful weights in the main part of the model.

            You should assign the lr_mult attribute to a layer. This will multiply the learning rate
            used by the base optimizer for that layer.

            This method creates a copy of the base optimizer for each unique learning rate multipler.

            Performance is similar to using a single copy of the base optimizer as gradients are computed
            only once and then passed on.

            This optimizer does not support from_config or get_config. To try to preserve the state, you may
            serialize the optimizers in the optimizer_group attribute in an instance of this class.

            Example usage
                model = tf.keras.Sequential()
                model.add(tf.keras.applications.resnet.ResNet50(include_top = False, pooling = 'avg'))
                model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
                model.layers[0].lr_mult = 0.01
                opt = DiscriminativeWrapper(tf.keras.optimizers.Adam, model, learning_rate = 0.01)
                model.compile(loss = tf.keras.losses.BinaryCrossentropy, optimizer = opt)
                model.fit(x, y)

            Arguments
                base_optimizer: a class that inherits from tf.keras.optimizers.Optimizer. Do not
                    pass an instance of the class.

                model: tf.keras.Model, The model to be used for discriminative learning.
                    It should have at least 1 layer with the attribute lr_mult. The lr_mult should
                    be set to a value not equal to 1. Otherwise, you will have the exact same
                    result as not using discriminative learning.

                learning_rate: float, the learning rate for the model

                verbose: Bool, to generate a report on how many parameters are affected

                *args: Args to pass to the base optimizer

                **kwargs: Kwargs to pass to the base optimizer

            Returns
                Optimizer - A keras optimizer to use with model.compile

            References
                - [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/pdf/1801.06146.pdf)
        """
        assert issubclass(
            base_optimizer, tf.keras.optimizers.Optimizer
        ), "Base optimizer must be a class that inherits from tf.keras.optimizers.Optimizer"

        super().__init__(lr=learning_rate, name=name, *args, **kwargs)

        ModelManager()._prepare_model(model, verbose=verbose)

        self.opt_class = base_optimizer

        # find unique lr_mult
        variable_groups = {var.lr_mult_value: None for var in model.trainable_variables}

        self.optimizer_group = []

        for lr_mult_value in variable_groups.keys():
            opt = self.opt_class(learning_rate=learning_rate * lr_mult_value, **kwargs)
            opt.lr_mult_value = lr_mult_value
            self.optimizer_group.append(opt)

    def apply_gradients(self, grads_and_vars, name=None):
        """allocates gradients to each optimizer based on the variable's learning rate multiplier
        then applies the gradients. In graph mode, it returns 1 operation per optimizer
        """

        # create gradvar buckets for each opt
        gvdict = {}
        for opt in self.optimizer_group:
            gvdict[opt.lr_mult_value] = []

        # load the gradvars into the appropriate bucket
        for grad, var in tuple(grads_and_vars):
            gvdict[var.lr_mult_value].append((grad, var))

        # return results from each opt
        return [
            opt.apply_gradients(tuple(gvdict[opt.lr_mult_value]))
            for opt in self.optimizer_group
        ]

    def get_config(self):
        raise NotImplementedError("Optimizer wrapper does not support get config")

    def from_config(self):
        raise NotImplementedError("Optimizer wrapper does not support from config")
