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
# =============================================================================

import typing

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Addons")
class MultiHeadAttention(tf.keras.layers.Layer):
    r"""
    MultiHead Attention layer.

    Defines the MultiHead Attention operation as defined in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) which takes in a `query`, `key` and `value` tensors returns the dot-product attention between them:

        ```python
        mha = MultiHeadAttention(head_size=128, num_heads=128)

        query = tf.random.uniform((32, 20, 200)) # (B, N, Dq)
        key = tf.random.uniform((32, 15, 300)) # (B, M, Dk)
        value = tf.random.uniform((32, 15, 400)) # (B, M, Dv)

        attention = mha([query, key, value]) # (B, N, Dv)
        ```

    If `value` is not given then internally `value = key` will be used:

         ```python
        mha = MultiHeadAttention(head_size=128, num_heads=128)

        query = tf.random.uniform((32, 20, 200)) # (B, N, Dq)
        key = tf.random.uniform((32, 15, 300)) # (B, M, Dk)

        attention = mha([query, key]) # (B, N, Dk)
        ```

    Arguments
        head_size: int, dimensionality of the `query`, `key` and `value` tensors after the linear transformation.
        num_heads: int, number of attention heads.
        output_size: int, dimensionality of the output space, if `None` then the input dimension of `value` or `key` will be used, default `None`.
        dropout_rate: float, `rate` parameter for the dropout layer that is applied to attention after softmax, default `0`.
        use_projection_bias: bool, whether to use a bias term after the linear output projection.
        return_attn_coef: bool, if `True`, return the attention coefficients as an additional output argument.
        kernel_initializer: initializer, initializer for the kernel weights.
        kernel_regularizer: regularizer, regularizer for the kernel weights.
        kernel_constraint: constraint, constraint for the kernel weights.
        bias_initializer: initializer, initializer for the bias weights.
        bias_regularizer: regularizer, regularizer for the bias weights.
        bias_constraint: constraint, constraint for the bias weights.

    Input shape
        query: `(..., N, Dq)`
        key: `(..., M, Dk)`
        value: `(..., M, Dv)` (optional)

    Output shape
        - `(..., N, output_size)` if `output_size` is given, else
        - `(..., N, Dv)` if `value` is given, else
        - `(..., N, Dk)`
    """

    def __init__(
        self,
        head_size: int,
        num_heads: int,
        output_size: int = None,
        dropout_rate: float = 0.0,
        use_projection_bias: bool = True,
        return_attn_coef: bool = False,
        kernel_initializer: typing.Union[str, typing.Callable] = "glorot_uniform",
        kernel_regularizer: typing.Union[str, typing.Callable] = None,
        kernel_constraint: typing.Union[str, typing.Callable] = None,
        bias_initializer: typing.Union[str, typing.Callable] = "zeros",
        bias_regularizer: typing.Union[str, typing.Callable] = None,
        bias_constraint: typing.Union[str, typing.Callable] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.head_size = head_size
        self.num_heads = num_heads
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.use_projection_bias = use_projection_bias
        self.return_attn_coef = return_attn_coef

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape):

        num_query_features = input_shape[0][-1]
        num_key_features = input_shape[1][-1]
        num_value_features = input_shape[2][-1]
        output_size = self.output_size if self.output_size else num_value_features

        self.query_kernel = self.add_weight(
            name="query_kernel",
            shape=[self.num_heads, num_query_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.key_kernel = self.add_weight(
            name="key_kernel",
            shape=[self.num_heads, num_key_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.value_kernel = self.add_weight(
            name="value_kernel",
            shape=[self.num_heads, num_value_features, self.head_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.projection_kernel = self.add_weight(
            name="projection_kernel",
            shape=[self.num_heads, self.head_size, output_size],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )

        if self.use_projection_bias:
            self.projection_bias = self.add_weight(
                name="projection_bias",
                shape=[output_size],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.projection_bias = None

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        super().build(input_shape)

    def call(self, inputs, training=None):

        # einsum nomenclature
        # ------------------------
        # N = query elements
        # M = key/value elements
        # H = heads
        # I = input features
        # O = output features

        query = inputs[0]
        key = inputs[1]
        value = inputs[2]
        attention_mask = inputs[3] if len(inputs) == 4 else None

        # verify shapes
        if attention_mask is not None:
            assert (
                len(attention_mask.shape) >= 2
            ), "'attention_mask' must have atleast 2 dimensions"
            assert (
                query.shape[-2] == attention_mask.shape[-2]
            ), "attention_mask's second to last dimension must be equal to the number of elements in 'query'"
            assert (
                key.shape[-2] == attention_mask.shape[-1]
            ), "attention_mask's last dimension must be equal to the number of elements in 'key'"
            assert (
                key.shape[-2] == value.shape[-2]
            ), "the number of elements in 'key' must be equal to the same as the number of elements in 'value'"

        # Linear transformations
        query = tf.einsum("...NI , HIO -> ...NHO", query, self.query_kernel)
        key = tf.einsum("...MI , HIO -> ...MHO", key, self.key_kernel)
        value = tf.einsum("...MI , HIO -> ...MHO", value, self.value_kernel)

        # Scale dot-product, doing the division to either query or key
        # instead of their product saves some computation
        depth = tf.constant(self.head_size, dtype=tf.float32)
        query /= tf.sqrt(depth)

        # Calculate dot product attention
        logits = tf.einsum("...NHO,...MHO->...HNM", query, key)

        # apply mask
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.float32)

            # possibly expand on the head dimension so broadcasting works
            if len(attention_mask.shape) != len(logits.shape):
                attention_mask = tf.expand_dims(attention_mask, -3)

            logits += -10e9 * (1.0 - attention_mask)

        attn_coef = tf.nn.softmax(logits)

        # attention dropout
        attn_coef_dropout = self.dropout(attn_coef, training=training)

        # attention * value
        multihead_output = tf.einsum("...HNM,...MHI->...NHI", attn_coef_dropout, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done.
        output = tf.einsum(
            "...NHI,HIO->...NO", multihead_output, self.projection_kernel
        )

        if self.projection_bias is not None:
            output += self.projection_bias

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def compute_output_shape(self, input_shape):
        num_value_features = input_shape[2][-1]
        output_size = self.output_size if self.output_size else num_value_features

        return input_shape[0][:-1] + (output_size,)

    def get_config(self):
        return dict(
            head_size=self.head_size,
            num_heads=self.num_heads,
            output_size=self.output_size,
            dropout_rate=self.dropout_rate,
            use_projection_bias=self.use_projection_bias,
            return_attn_coef=self.return_attn_coef,
            kernel_initializer=tf.keras.initializers.serialize(self.kernel_initializer),
            kernel_regularizer=tf.keras.regularizers.serialize(self.kernel_regularizer),
            kernel_constraint=tf.keras.constraints.serialize(self.kernel_constraint),
            bias_initializer=tf.keras.initializers.serialize(self.bias_initializer),
            bias_regularizer=tf.keras.regularizers.serialize(self.bias_regularizer),
            bias_constraint=tf.keras.constraints.serialize(self.bias_constraint),
            **super().get_config(),
        )
