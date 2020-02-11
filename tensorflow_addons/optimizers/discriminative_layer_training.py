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


class DiscriminativeModelManager:
    """Class for grouping functions related to model lr_mult management"""

    @staticmethod
    def _get_layers(layer):
        """Helper method to access a layer's sublayers as a list or return an empty list
        """
        return getattr(layer, "layers", [])

    @staticmethod
    def _get_lr_mult(layer):
        """Helper method to access a layer's learning rate multiplier, which defaults to 1 if lr mult is not set
        """
        return getattr(layer, "lr_mult", 1.0)

    @staticmethod
    def _assign_lr_mult(layer, lr_mult):
        """Helper method to assign a layer's learning rate multiplier, which does nothing if lr mult is already set
        """
        if not hasattr(layer, "lr_mult"):
            layer.lr_mult = lr_mult  # since layer has no lr mult, assign the mult
            # this method should be called after the user has already assigned some lr mults
            # to some layers. We just don't want to override any lr mults they assigned
        else:
            # we pass here because of propagation to nested layers
            # users should be able to speficy model.layers[0].layers[0].lr_mult = 0.01
            # and model.layers[0].lr_mult = 0.1, such that the model.layers[0].layers[0]
            # keeps its assigned lr mult of 0.01
            pass

    @staticmethod
    def _recursively_assign_sublayer_lr_mult(layer):

        """Helper method iterate through all nested layers of an object that behaves like a layer or model
        By default, we want to propagate the lr mult to the lower layers.


        https://stackoverflow.com/questions/6340351/iterating-through-list-of-list-in-python
        """

        mult = DiscriminativeModelManager._get_lr_mult(layer)
        layers = DiscriminativeModelManager._get_layers(layer)

        if len(layers) > 0:
            for sublayer in layers:

                # we always assign the lr mult to the sublayers of the current layer
                # the assign method will avoid overwritting lr mults
                # so if you have a resnet and you specifically assign the first resnet layer
                # to have lr_mult of 0.01 and the resnet model to have lr_mult of 0.1, all
                # resnet layers except the first should get lr_mult of 0.1 and the first
                # keeps its lr_mult of 0.01
                DiscriminativeModelManager._assign_lr_mult(sublayer, mult)

                # recursively iterate through the nested layers
                for (
                    nested_sublayer
                ) in DiscriminativeModelManager._recursively_assign_sublayer_lr_mult(
                    sublayer
                ):
                    yield nested_sublayer
        else:
            yield layer

    @staticmethod
    def _apply_lr_mult_to_var(layer):
        """Helper method to apply the lr mult to the trainable variables of a layer
        """
        lr_mult = DiscriminativeModelManager._get_lr_mult(layer)
        for var in layer.trainable_variables:
            var.lr_mult = lr_mult
            # the lr_mult behaves as a hyper parameter and not a variable. it will not be a tensor
            # there's not benefit in setting the lr_mult as a variable because it does not interact with tensors

    @staticmethod
    def _check_for_lr_mult(layer, verbose=True, propagate=True):
        """Identify which layers have an LR mult not equal to 1
        """

        layers_with_lr_mult = []

        for (
            sub_layer
        ) in DiscriminativeModelManager._recursively_assign_sublayer_lr_mult(layer):
            lr_mult = DiscriminativeModelManager._get_lr_mult(sub_layer)
            if lr_mult != 1.0:
                layers_with_lr_mult.append(sub_layer)
                if verbose:
                    logging.info("layer %s lr_mult : %f" % (sub_layer.name, lr_mult))

        return layers_with_lr_mult

    @staticmethod
    def _compute_params(var_list):
        """helps compute params to provide a summary that aligns with model.summary()
        """
        return np.sum([np.prod(list(var.shape)) for var in var_list])

    @staticmethod
    def _prepare_model(model, verbose=True):
        """Prepares a model for disc training
        """

        layers_with_lr_mult = DiscriminativeModelManager._check_for_lr_mult(
            model, verbose=verbose
        )
        if len(layers_with_lr_mult) == 0:

            logging.warning(
                """No Layer has been assigned an lr_mult attribute != 1.0
                Discriminative Layer Training will apply the same learning rate to all layers
                It will perform as if you did not use Discriminative Layer Training
                """
            )

        for layer in DiscriminativeModelManager._recursively_assign_sublayer_lr_mult(
            model
        ):
            DiscriminativeModelManager._apply_lr_mult_to_var(layer)

        vars_with_lr_mult = [
            var for var in model.trainable_variables if var.lr_mult != 1.0
        ]

        if verbose:
            logging.info(
                "%i params of %i will learn at a different rate"
                % (
                    DiscriminativeModelManager._compute_params(vars_with_lr_mult),
                    DiscriminativeModelManager._compute_params(
                        model.trainable_variables
                    ),
                )
            )


class DiscriminativeLayerOptimizer(tf.keras.optimizers.Optimizer):
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

        DiscriminativeModelManager._prepare_model(model, verbose=verbose)

        self.opt_class = base_optimizer

        # find unique lr_mult
        variable_groups = {var.lr_mult_value: None for var in model.trainable_variables}

        self.optimizer_group = []

        for lr_mult in variable_groups.keys():
            opt = self.opt_class(learning_rate=learning_rate * lr_mult, **kwargs)
            opt.lr_mult = lr_mult
            self.optimizer_group.append(opt)

    def apply_gradients(self, grads_and_vars, name=None):
        """allocates gradients to each optimizer based on the variable's learning rate multiplier
        then applies the gradients. In graph mode, it returns 1 operation per optimizer
        Please use the model.fit method instead of accessing this directly
        """

        # create gradvar buckets for each opt
        gvdict = {}
        for opt in self.optimizer_group:
            gvdict[opt.lr_mult] = []

        # load the gradvars into the appropriate bucket
        for grad, var in tuple(grads_and_vars):
            gvdict[var.lr_mult].append((grad, var))

        # return results from each opt
        # in eager mode, this will return a list of irrelevant results for each optimizer
        # in eager mode, the function apply_gradients actually applies gradients to the model
        # in graph mode, this will return a list of tensor ops for each opt
        # in graph mode, apply_gradients creates the tensor ops for applying gradients on the graph
        return [
            opt.apply_gradients(tuple(gvdict[opt.lr_mult]))
            for opt in self.optimizer_group
        ]

    def get_config(self):
        raise NotImplementedError("Optimizer wrapper does not support get config")

    def from_config(self):
        raise NotImplementedError("Optimizer wrapper does not support from config")
