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
#
# Orginal implementation from keras_contrib/layers/crf
# ==============================================================================
"""Implementing Conditional Random Field layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.text.crf import crf_decode, crf_log_likelihood


@tf.keras.utils.register_keras_serializable(package='Addons')
class CRF(tf.keras.layers.Layer):
    """Linear chain conditional random field (CRF).

    Examples:

    ```python
        from tensorflow_addons.layers import CRF
        from tensorflow_addons.losses import crf_loss

        model = Sequential()
        model.add(Embedding(3001, 300, mask_zero=True)

        crf = CRF(10)
        model.add(crf)

        model.compile('adam', loss=crf_loss)

        model.fit(x, y)
    ```

    Arguments:
        units: Positive integer, dimensionality of the output space,
            should equal to tag num.
        chain_initializer: Initializer for the `chain_kernel` weights matrix,
            used for the CRF chain energy.
            (see [initializers](../initializers.md)).
        chain_regularizer: Regularizer function applied to
            the `chain_kernel` weights matrix.
        chain_constraint: Constraint function applied to
            the `chain_kernel` weights matrix.
        use_boundary: Boolean (default True), indicating if trainable
            start-end chain energies should be added to model.
        boundary_initializer: Initializer for the `left_boundary`,
            'right_boundary' weights vectors,
            used for the start/left and end/right boundary energy.
        boundary_regularizer: Regularizer function applied to
            the 'left_boundary', 'right_boundary' weight vectors.
        boundary_constraint: Constraint function applied to
            the `left_boundary`, `right_boundary` weights vectors.
        use_kernel: Boolean (default True), indicating if apply
            a fully connected layer before CRF op.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        use_bias: Boolean (default True), whether the layer uses a bias vector.
        bias_initializer: Initializer for the bias vector.
        bias_regularizer: Regularizer function applied to the bias vector.
        bias_constraint: Constraint function applied to the bias vector.
        activation: default value is 'linear', Activation function to use.

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, feature_size)`.

    Output shape:
        2D tensor (dtype: int32) with shape: `(batch_size, sequence_length)`.

    Masking:
        This layer supports masking
        (2D tensor, shape: `(batch_size, sequence_length)`)
        for input data with a variable number of timesteps.
        This layer output same make tensor,
        NOTICE this may cause issue when you
        use some keras loss and metrics function which usually expect 1D mask.

    Loss function:
        Due to the TF 2.0 version support eager execution be default,
        there is no way can implement CRF loss as independent loss function.
        Thus, user should use loss method of this layer.
        See Examples (above) for detailed usage.

    References:
        - [Conditional Random Field](https://en.wikipedia.org/wiki/Conditional_random_field)
    """

    def __init__(self,
                 units,
                 chain_initializer="orthogonal",
                 chain_regularizer=None,
                 chain_constraint=None,
                 use_boundary=True,
                 boundary_initializer="zeros",
                 boundary_regularizer=None,
                 boundary_constraint=None,
                 use_kernel=True,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 bias_initializer="zeros",
                 bias_regularizer=None,
                 bias_constraint=None,
                 activation="linear",
                 **kwargs):
        super(CRF, self).__init__(**kwargs)

        # setup mask supporting flag, used by base class (the Layer)
        # because base class's init method will set it to False unconditionally
        # So this assigned must be executed after call base class's init method
        self.supports_masking = True

        self.units = units  # numbers of tags

        self.use_boundary = use_boundary
        self.use_bias = use_bias
        self.use_kernel = use_kernel

        self.activation = tf.keras.activations.get(activation)

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.chain_initializer = tf.keras.initializers.get(chain_initializer)
        self.boundary_initializer = tf.keras.initializers.get(
            boundary_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.chain_regularizer = tf.keras.regularizers.get(chain_regularizer)
        self.boundary_regularizer = tf.keras.regularizers.get(
            boundary_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.chain_constraint = tf.keras.constraints.get(chain_constraint)
        self.boundary_constraint = tf.keras.constraints.get(
            boundary_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        # values will be assigned in method
        self.input_spec = None

        # value remembered for loss/metrics function
        self.potentials = None
        self.sequence_length = None
        self.mask = None

        # global variable
        self.kernel = None
        self.chain_kernel = None
        self.bias = None
        self.left_boundary = None
        self.right_boundary = None

    def build(self, input_shape):
        input_shape = tuple(tf.TensorShape(input_shape).as_list())

        # see API docs of InputSpec for more detail
        self.input_spec = [tf.keras.layers.InputSpec(shape=input_shape)]

        feature_size = input_shape[-1]

        if self.use_kernel:
            # weights that mapping arbitrary tensor to correct shape
            self.kernel = self.add_weight(
                shape=(feature_size, self.units),
                name="kernel",
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )

        # weights that work as transfer probability of each tags
        self.chain_kernel = self.add_weight(
            shape=(self.units, self.units),
            name="chain_kernel",
            initializer=self.chain_initializer,
            regularizer=self.chain_regularizer,
            constraint=self.chain_constraint,
        )

        # bias that works with self.kernel
        if self.use_kernel and self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name="bias",
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = 0

        # weight of <START> to tag probability and tag to <END> probability
        if self.use_boundary:
            self.left_boundary = self.add_weight(
                shape=(self.units,),
                name="left_boundary",
                initializer=self.boundary_initializer,
                regularizer=self.boundary_regularizer,
                constraint=self.boundary_constraint,
            )
            self.right_boundary = self.add_weight(
                shape=(self.units,),
                name="right_boundary",
                initializer=self.boundary_initializer,
                regularizer=self.boundary_regularizer,
                constraint=self.boundary_constraint,
            )

        # or directly call self.built = True
        super(CRF, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        # mask: Tensor(shape=(batch_size, sequence_length), dtype=bool) or None

        if mask is not None:
            if tf.keras.backend.ndim(mask) != 2:
                raise ValueError(
                    "Input mask to CRF must have dim 2 if not None")

        # left padding of mask is not supported, due the underline CRF function
        # detect it and report it to user
        first_mask = None
        if mask is not None:
            left_boundary_mask = self._compute_mask_left_boundary(mask)
            first_mask = left_boundary_mask[:, 0]

        # remember this value for later use
        self.mask = mask

        if first_mask is not None:
            no_left_padding = tf.math.reduce_all(first_mask)
            msg = "Currently, CRF layer do not support left padding"
            with tf.control_dependencies([
                    tf.debugging.assert_equal(
                        no_left_padding, tf.constant(True), message=msg)
            ]):
                self.potentials = self._dense_layer(inputs)
        else:
            self.potentials = self._dense_layer(inputs)

        # appending boundary probability info
        if self.use_boundary:
            self.potentials = self.add_boundary_energy(
                self.potentials, mask, self.left_boundary, self.right_boundary)

        self.sequence_length = self._get_sequence_length(inputs, mask)

        decoded_sequence, _ = self.get_viterbi_decoding(
            self.potentials, self.sequence_length)

        return decoded_sequence

    def _get_sequence_length(self, input_, mask):
        """Currently underline CRF fucntion (provided by
        tensorflow_addons.text.crf) do not support bi-direction masking (left
        padding / right padding), it support right padding by tell it the
        sequence length.

        this function is compute the sequence length from input and
        mask.
        """
        if mask is not None:
            sequence_length = self.mask_to_sequence_length(mask)
        else:
            # make a mask tensor from input, then used to generate sequence_length
            input_energy_shape = tf.shape(input_)
            raw_input_shape = tf.slice(input_energy_shape, [0], [2])
            alt_mask = tf.ones(raw_input_shape)

            sequence_length = self.mask_to_sequence_length(alt_mask)

        return sequence_length

    def mask_to_sequence_length(self, mask):
        """compute sequence length from mask."""
        sequence_length = tf.keras.backend.cast(
            tf.keras.backend.sum(tf.keras.backend.cast(mask, tf.int8), 1),
            tf.int64)
        return sequence_length

    @staticmethod
    def _compute_mask_right_boundary(mask):
        """input mask: 0011100, output left_boundary: 0000100."""
        # shift mask to left by 1: 0011100 => 0111000
        offset = 1
        left_shifted_mask = tf.keras.backend.concatenate(
            [mask[:, offset:],
             tf.keras.backend.zeros_like(mask[:, :offset])],
            axis=1)

        # TODO(howl-anderson): for below code
        # Original code in keras_contrib:
        # end_mask = K.cast(
        #   K.greater(self.shift_left(mask), mask),
        #   K.floatx()
        # )
        # May have a bug, it's better confirmed
        # by the original keras_contrib maintainer
        # Luiz Felix (github: lzfelix),
        # mailed him already and waiting for reply.

        # 0011100 > 0111000 => 0000100
        right_boundary = tf.keras.backend.greater(mask, left_shifted_mask)

        return right_boundary

    @staticmethod
    def _compute_mask_left_boundary(mask):
        """input mask: 0011100, output left_boundary: 0010000."""
        # shift mask to right by 1: 0011100 => 0001110
        offset = 1
        right_shifted_mask = tf.keras.backend.concatenate(
            [tf.keras.backend.zeros_like(mask[:, :offset]), mask[:, :-offset]],
            axis=1)

        # 0011100 > 0001110 => 0010000
        left_boundary = tf.keras.backend.greater(
            tf.dtypes.cast(mask, tf.int32),
            tf.dtypes.cast(right_shifted_mask, tf.int32))
        # left_boundary = tf.keras.backend.greater(mask, right_shifted_mask)

        return left_boundary

    def add_boundary_energy(self, potentials, mask, start, end):
        def expend_scalar_to_3d(x):
            # expend tensor from shape (x, ) to (1, 1, x)
            return tf.keras.backend.expand_dims(
                tf.keras.backend.expand_dims(x, 0), 0)

        start = expend_scalar_to_3d(start)
        end = expend_scalar_to_3d(end)
        if mask is None:
            potentials = tf.keras.backend.concatenate(
                [potentials[:, :1, :] + start, potentials[:, 1:, :]], axis=1)
            potentials = tf.keras.backend.concatenate(
                [potentials[:, :-1, :], potentials[:, -1:, :] + end], axis=1)
        else:
            mask = tf.keras.backend.expand_dims(
                tf.keras.backend.cast(mask, start.dtype), axis=-1)
            start_mask = tf.keras.backend.cast(
                self._compute_mask_left_boundary(mask),
                start.dtype,
            )

            end_mask = tf.keras.backend.cast(
                self._compute_mask_right_boundary(mask),
                end.dtype,
            )
            potentials = potentials + start_mask * start
            potentials = potentials + end_mask * end
        return potentials

    def get_viterbi_decoding(self, potentials, sequence_length):
        # decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`
        decode_tags, best_score = crf_decode(potentials, self.chain_kernel,
                                             sequence_length)

        return decode_tags, best_score

    def get_config(self):
        # used for loading model from disk
        config = {
            "units":
            self.units,
            "use_boundary":
            self.use_boundary,
            "use_bias":
            self.use_bias,
            "use_kernel":
            self.use_kernel,
            "kernel_initializer":
            tf.keras.initializers.serialize(self.kernel_initializer),
            "chain_initializer":
            tf.keras.initializers.serialize(self.chain_initializer),
            "boundary_initializer":
            tf.keras.initializers.serialize(self.boundary_initializer),
            "bias_initializer":
            tf.keras.initializers.serialize(self.bias_initializer),
            "activation":
            tf.keras.activations.serialize(self.activation),
            "kernel_regularizer":
            tf.keras.regularizers.serialize(self.kernel_regularizer),
            "chain_regularizer":
            tf.keras.regularizers.serialize(self.chain_regularizer),
            "boundary_regularizer":
            tf.keras.regularizers.serialize(self.boundary_regularizer),
            "bias_regularizer":
            tf.keras.regularizers.serialize(self.bias_regularizer),
            "kernel_constraint":
            tf.keras.constraints.serialize(self.kernel_constraint),
            "chain_constraint":
            tf.keras.constraints.serialize(self.chain_constraint),
            "boundary_constraint":
            tf.keras.constraints.serialize(self.boundary_constraint),
            "bias_constraint":
            tf.keras.constraints.serialize(self.bias_constraint)
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:2]
        return output_shape

    def compute_mask(self, input_, mask=None):
        """keep mask shape [batch_size, max_seq_len]"""
        return mask

    def get_negative_log_likelihood(self, y_true):
        # TODO(howl-anderson): remove unnecessary typing cast
        self.potentials = tf.keras.backend.cast(self.potentials, tf.float32)
        y_true = tf.keras.backend.cast(y_true, tf.int32)
        self.sequence_length = tf.keras.backend.cast(self.sequence_length,
                                                     tf.int32)
        # self.chain_kernel = tf.keras.backend.cast(self.chain_kernel,
        #                                           tf.float32)

        log_likelihood, _ = crf_log_likelihood(
            self.potentials, y_true, self.sequence_length, self.chain_kernel)

        return -log_likelihood

    def get_loss(self, y_true, y_pred):
        # we don't use y_pred, but caller pass it anyway, ignore it
        return self.get_negative_log_likelihood(y_true)

    def get_accuracy(self, y_true, y_pred):
        judge = tf.keras.backend.cast(
            tf.keras.backend.equal(y_pred, y_true), tf.keras.backend.floatx())
        if self.mask is None:
            return tf.keras.backend.mean(judge)
        else:
            mask = tf.keras.backend.cast(self.mask, tf.keras.backend.floatx())
            return (tf.keras.backend.sum(judge * mask) /
                    tf.keras.backend.sum(mask))

    def _dense_layer(self, input_):
        if self.use_kernel:
            output = self.activation(
                tf.keras.backend.dot(input_, self.kernel) + self.bias)
        else:
            output = input_

        return tf.keras.backend.cast(output, self.chain_kernel.dtype)

    def __call__(self, inputs, *args, **kwargs):
        outputs = super(CRF, self).__call__(inputs, *args, **kwargs)

        # A hack that add _keras_history to EagerTensor, make it more like normal Tensor
        for tensor in tf.nest.flatten(outputs):
            if not hasattr(tensor, '_keras_history'):
                tensor._keras_history = (self, 0, 0)

        return outputs

    @property
    def _compute_dtype(self):
        # fixed output dtype from underline CRF functions
        return tf.int32
