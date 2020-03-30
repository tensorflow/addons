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

# Orginal implementation from keras_contrib/layer/normalization
# =============================================================================

import logging
import tensorflow as tf
from typeguard import typechecked

from tensorflow_addons.utils import types


@tf.keras.utils.register_keras_serializable(package="Addons")
class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer.

    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.

    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical
    to Layer Normalization.

    Relation to Instance Normalization:
    If the number of groups is set to the
    input dimension (number of groups is equal
    to number of channels), then this operation becomes
    identical to Instance Normalization.

    Arguments
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
        axis: Integer, the axis that should be normalized.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape
        Same shape as input.
    References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    @typechecked
    def __init__(
        self,
        groups: int = 2,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: types.Initializer = "zeros",
        gamma_initializer: types.Initializer = "ones",
        beta_regularizer: types.Regularizer = None,
        gamma_regularizer: types.Regularizer = None,
        beta_constraint: types.Constraint = None,
        gamma_constraint: types.Constraint = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):

        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape
        )

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        outputs = tf.reshape(normalized_inputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True
        )

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis " + str(self.axis) + " of "
                "input tensor should have a defined dimension "
                "but the layer received an input with shape " + str(input_shape) + "."
            )

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):

        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") cannot be "
                "more than the number of channels (" + str(dim) + ")."
            )

        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") must be a "
                "multiple of the number of channels (" + str(dim) + ")."
            )

    def _check_axis(self):

        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. "
                "Use tf.layers.batch_normalization instead."
            )

    def _create_input_spec(self, input_shape):

        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape


@tf.keras.utils.register_keras_serializable(package="Addons")
class InstanceNormalization(GroupNormalization):
    """Instance normalization layer.

    Instance Normalization is an specific case of ```GroupNormalization```since
    it normalizes all features of one channel. The Groupsize is equal to the
    channel size. Empirically, its accuracy is more stable than batch norm in a
    wide range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.

    Arguments
        axis: Integer, the axis that should be normalized.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape
        Same shape as input.

    References
        - [Instance Normalization: The Missing Ingredient for Fast Stylization]
        (https://arxiv.org/abs/1607.08022)
    """

    def __init__(self, **kwargs):
        if "groups" in kwargs:
            logging.warning("The given value for groups will be overwritten.")

        kwargs["groups"] = -1
        super().__init__(**kwargs)


@tf.keras.utils.register_keras_serializable(package="Addons")
class FilterResponseNormalization(tf.keras.layers.Layer):
    """Filter response normalization layer.

    Filter Response Normalization (FRN), a normalization
    method that enables models trained with per-channel
    normalization to achieve high accuracy. It performs better than
    all other normalization techniques for small batches and is par
    with Batch Normalization for bigger batch sizes.

    Arguments
        axis: List of axes that should be normalized. This should represent the
              spatial dimensions.
        channel_idx: Index of the channel axis in `input_shape`.
        epsilon: Small positive float value added to variance to avoid dividing by zero.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
        learned_epsilon: (bool) Whether to add another learnable
        epsilon parameter or not.
        name: Optional name for the layer

    Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model. This layer supports
        arbitrary tensors.

    Output shape
        Same shape as input.

    References
        - [Filter Response Normalization Layer: Eliminating Batch Dependence
        in the training of Deep Neural Networks]
        (https://arxiv.org/abs/1911.09737)
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        axis: list = [1, 2],
        channel_idx: int = -1,
        beta_initializer: types.Initializer = "zeros",
        gamma_initializer: types.Initializer = "ones",
        beta_regularizer: types.Regularizer = None,
        gamma_regularizer: types.Regularizer = None,
        beta_constraint: types.Constraint = None,
        gamma_constraint: types.Constraint = None,
        learned_epsilon: bool = False,
        learned_epsilon_constraint: types.Constraint = None,
        name: str = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.epsilon = tf.math.abs(tf.cast(epsilon, dtype=self.dtype))
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self.use_eps_learned = learned_epsilon
        self.supports_masking = True

        if self.use_eps_learned:
            self.eps_learned_initializer = tf.keras.initializers.Constant(1e-4)
            self.eps_learned_constraint = tf.keras.constraints.get(
                learned_epsilon_constraint
            )
            self.eps_learned = self.add_weight(
                shape=(1,),
                name="learned_epsilon",
                dtype=self.dtype,
                initializer=tf.keras.initializers.get(self.eps_learned_initializer),
                regularizer=None,
                constraint=self.eps_learned_constraint,
            )
        else:
            self.eps_learned_initializer = None
            self.eps_learned_constraint = None

        self._check_axis(axis)

    def build(self, input_shape):
        self._check_if_input_shape_is_none(input_shape)
        self._create_input_spec(input_shape)
        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        epsilon = self.epsilon
        if self.use_eps_learned:
            epsilon += tf.math.abs(self.eps_learned)
        nu2 = tf.reduce_mean(tf.square(inputs), axis=self.axis, keepdims=True)
        normalized_inputs = inputs * tf.math.rsqrt(nu2 + epsilon)
        return self.gamma * normalized_inputs + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "axis": self.axis,
            "channel_idx": self.channel_idx,
            "epsilon": self.epsilon,
            "learned_epsilon": self.use_eps_learned,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
            "learned_epsilon_constraint": tf.keras.constraints.serialize(
                self.eps_learned_constraint
            ),
        }
        base_config = super().get_config()
        return dict(**base_config, **config)

    def _create_input_spec(self, input_shape):
        ndims = len(input_shape)
        for idx, x in enumerate(self.axis):
            if x < 0:
                self.axis[idx] = ndims + x

        axis_to_dim = {x: input_shape[x] for x in self.axis}
        self.input_spec = tf.keras.layers.InputSpec(ndim=ndims, axes=axis_to_dim)

    def _check_axis(self, axis):
        if not isinstance(axis, list) and not isinstance(axis, int):
            raise TypeError(
                """Expected a list of values but got {}.""".format(type(axis))
            )
        else:
            self.axis = axis

        if isinstance(axis, list):
            for x in self.axis:
                if x < 0 or x >= len(input_shape):
                    raise ValueError("Invalid axis: %d" % x)

                elif x == 0 or x == -len(input_shape):
                    raise ValueError(
                        "You are trying to normalize your batch axis. "
                        "Use tf.layers.batch_normalization instead."
                    )

        else:
            self.axis = self.channel_idx = -1

    def _check_if_input_shape_is_none(self, input_shape):
        dim1, dim2 = input_shape[self.axis[0]], input_shape[self.axis[1]]
        if dim1 is None or dim2 is None:
            raise ValueError(
                """Axis {} of input tensor should have a defined dimension but
                the layer received an input with shape {}.""".format(
                    self.axis, input_shape
                )
            )

    def _add_gamma_weight(self, input_shape):
        # Get the channel dimension
        dim = input_shape[self.channel_idx]
        shape = [1 if i != self.channel_idx else dim for i in range(len(input_shape))]
        # Initialize gamma with ones except the channel axis
        self.gamma = self.add_weight(
            shape=shape,
            name="gamma",
            dtype=self.dtype,
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint,
        )

    def _add_beta_weight(self, input_shape):
        # Get the channel dimension
        dim = input_shape[self.channel_idx]
        shape = [1 if i != self.channel_idx else dim for i in range(len(input_shape))]
        # Initialize beta with ones except the channel axis
        self.beta = self.add_weight(
            shape=shape,
            name="beta",
            dtype=self.dtype,
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            constraint=self.beta_constraint,
        )
