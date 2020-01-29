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
"""Discriminative Layer Training Manager for TensorFlow."""

import tensorflow as tf
import numpy as np


def apply_gradients(self, grads_and_vars, *args, **kwargs):
    """New apply gradients function.
    This intercepts the grads and vars, scales them, then passes them to the old apply gradients function

    :TODO finish docstring

    """

    if self.testing_flag:
        print("Training with layerwise learning rates")
        self.testing_flag = False

    grads = []
    var_list = []

    # scale each grad based on var's lr_mult
    for grad, var in grads_and_vars:
        grad = tf.math.scalar_mul(var.lr_mult, grad)
        grads.append(grad)
        var_list.append(var)

    grads_and_vars = list(zip(grads, var_list))

    return self._apply_gradients(grads_and_vars, *args, **kwargs)


# @tf.keras.utils.register_keras_serializable(package="Addons") :TODO figure out why other classes have this wrapper
class DiscriminativeLearning(object):
    def __init__(self, model, verbose=True):
        """Apply logic for discriminative learning to a compiled model
        :TODO finish docstring
        """

        self._prepare_model(model=model, verbose=verbose)

    def _get_layers(self, layer):
        """Helper method to access a layer's sublayers as a list or return an empty list
        :TODO finish docstring

        """

        try:
            return layer.layers
        except AttributeError:
            return []

    def _get_lr_mult(self, layer):

        """Helper method to access a layer's learning rate multiplier, which defaults to 1 if lr mult is not set
        :TODO finish docstring
        """

        try:
            return layer.lr_mult
        except AttributeError:
            return 1.0

    def _assign_lr_mult(self, layer, lr_mult, override=False):

        """Helper method to assign a layer's learning rate multiplier, which does nothing if lr mult is already set
        :TODO finish docstring
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
        :TODO finish docstring

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
        :TODO finish docstring
        """

        lr_mult = self._get_lr_mult(layer)

        for var in layer.trainable_variables:
            var.lr_mult = tf.convert_to_tensor(lr_mult, tf.float32)  # 0D tensor
            var.lr_mult_value = (
                lr_mult  # easier to check vars lr mult in graph and eager
            )

    # :TODO float16 testing? not sure what would happen atm

    def _check_for_lr_mult(self, layer, verbose=True, propagate=True):
        """Identify which layers have an LR mult not equal to 1
        :TODO finish docstring
        """

        layers_with_lr_mult = []

        for sub_layer in self._get_lowest_layers(
            layer, propagate_lr_mult_to_sub_layers=propagate
        ):
            lr_mult = self._get_lr_mult(sub_layer)
            if lr_mult != 1.0:
                layers_with_lr_mult.append(sub_layer)
                if verbose:
                    # :TODO this should be info
                    print("layer %s lr_mult : %f" % (sub_layer.name, lr_mult))

        return layers_with_lr_mult

    def _compute_params(self, var_list):
        """helps compute params to provide a summary that aligns with model.summary()
        :TODO finish docstring
        """
        return np.sum([np.prod(list(var.shape)) for var in var_list])

    def _prepare_model(self, model, verbose=True):
        """Prepares a compiled model for discriminative layer training

        Model must be compiled first
        :TODO finish docstring
        """

        layers_with_lr_mult = self._check_for_lr_mult(model, verbose=verbose)
        if len(layers_with_lr_mult) == 0:
            # :TODO this should be a warning

            print(
                "Discriminative Layer Training requires an lr_mult attribute on at least one layer"
            )

            print(
                "The assigned lr_mult must not be equal to 1. eg: model.layers[0].lr_mult = 0.01"
            )

        for layer in self._get_lowest_layers(model):
            self._apply_lr_mult_to_var(layer)

        # get opt, move the original apply fn to a safe place, assign new apply fn

        opt = model.optimizer
        opt._apply_gradients = opt.apply_gradients
        opt.apply_gradients = apply_gradients.__get__(opt)
        opt.testing_flag = True
        vars_with_lr_mult = [
            var for var in model.trainable_variables if var.lr_mult_value != 1.0
        ]

        # :TODO this should be info
        if verbose:
            print(
                "%i params of %i will learn at a different rate"
                % (
                    self._compute_params(vars_with_lr_mult),
                    self._compute_params(model.trainable_variables),
                )
            )
