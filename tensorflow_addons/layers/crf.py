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
from tensorflow_addons.utils import keras_utils

"""
TODO

* decide input_dim should be keep or drop
* left padding of mask is not supported (future version should fix it)
* not test yet if CRF is the first layer
* Add docs
"""

@keras_utils.register_keras_custom_object
class CRF(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 use_boundary=False,
                 use_bias=True,
                 activation="linear",
                 kernel_initializer="glorot_uniform",
                 chain_initializer="orthogonal",
                 bias_initializer="zeros",
                 boundary_initializer="zeros",
                 kernel_regularizer=None,
                 chain_regularizer=None,
                 boundary_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 chain_constraint=None,
                 boundary_constraint=None,
                 bias_constraint=None,
                 input_dim=None,
                 **kwargs):
        super(CRF, self).__init__(**kwargs)

        # setup mask supporting flag, used by base class (the Layer)
        # because base class's init method will set it to False unconditionally
        # So this assigned must be executed after call base class's init method
        self.supports_masking = True

        self.units = units  # numbers of tags

        self.use_boundary = use_boundary
        self.use_bias = use_bias

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

        self.input_dim = input_dim

        # values will be assigned in method
        self.input_spec = None

        # value remembered for loss/metrics function
        self.logits = None
        self.nwords = None
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

        self.input_dim = input_shape[-1]

        # weights that mapping arbitrary tensor to correct shape
        self.kernel = self.add_weight(
            shape=(self.input_dim, self.units),
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
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units, ),
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
                shape=(self.units, ),
                name="left_boundary",
                initializer=self.boundary_initializer,
                regularizer=self.boundary_regularizer,
                constraint=self.boundary_constraint,
            )
            self.right_boundary = self.add_weight(
                shape=(self.units, ),
                name="right_boundary",
                initializer=self.boundary_initializer,
                regularizer=self.boundary_regularizer,
                constraint=self.boundary_constraint,
            )

        # or directly call self.built = True
        super(CRF, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        # mask: Tensor(shape=(?, ?), dtype=bool) or None

        if mask is not None:
            assert (tf.keras.backend.ndim(mask) == 2
                    ), "Input mask to CRF must have dim 2 if not None"

        # remember this value for later use
        self.mask = mask

        logits = self._dense_layer(inputs)

        # appending boundary probability info
        if self.use_boundary:
            logits = self.add_boundary_energy(logits, mask, self.left_boundary,
                                              self.right_boundary)

        # remember this value for later use
        self.logits = logits

        nwords = self._get_nwords(inputs, mask)

        # remember this value for later use
        self.nwords = nwords

        test_output = self.get_viterbi_decoding(logits, nwords)

        # WHY: don't remove this line, useless but remote it will cause bug
        test_output = tf.cast(test_output, tf.float32)
        out = test_output

        return out

    def _get_nwords(self, input, mask):
        if mask is not None:
            int_mask = tf.keras.backend.cast(mask, tf.int8)
            nwords = self.mask_to_nwords(int_mask)
        else:
            # make a mask tensor from input, then used to generate nwords
            input_energy_shape = tf.shape(input)
            raw_input_shape = tf.slice(input_energy_shape, [0], [2])
            alt_mask = tf.ones(raw_input_shape)

            nwords = self.mask_to_nwords(alt_mask)

        return nwords

    def mask_to_nwords(self, mask):
        nwords = tf.keras.backend.cast(tf.keras.backend.sum(mask, 1), tf.int64)
        return nwords

    @staticmethod
    def shift_left(x, offset=1):
        assert offset > 0
        return tf.keras.backend.concatenate(
            [x[:, offset:],
             tf.keras.backend.zeros_like(x[:, :offset])],
            axis=1)

    @staticmethod
    def shift_right(x, offset=1):
        assert offset > 0
        return tf.keras.backend.concatenate(
            [tf.keras.backend.zeros_like(x[:, :offset]), x[:, :-offset]],
            axis=1)

    def add_boundary_energy(self, energy, mask, start, end):
        def expend_scalar_to_3d(x):
            # expend tensor from shape (x, ) to (1, 1, x)
            return tf.keras.backend.expand_dims(
                tf.keras.backend.expand_dims(x, 0), 0)

        start = expend_scalar_to_3d(start)
        end = expend_scalar_to_3d(end)
        if mask is None:
            energy = tf.keras.backend.concatenate(
                [energy[:, :1, :] + start, energy[:, 1:, :]], axis=1)
            energy = tf.keras.backend.concatenate(
                [energy[:, :-1, :], energy[:, -1:, :] + end], axis=1)
        else:
            mask = tf.keras.backend.expand_dims(tf.keras.backend.cast(
                mask, tf.keras.backend.floatx()),
                                                axis=-1)
            start_mask = tf.keras.backend.cast(
                tf.keras.backend.greater(mask, self.shift_right(mask)),
                tf.keras.backend.floatx(),
            )

            # original code:
            # end_mask = K.cast(
            #   K.greater(self.shift_left(mask), mask),
            #   K.floatx()
            # )
            # Note: original code should have a bug,
            # need confirmed with @lzfelix (Luiz Felix),
            # mailed to him but no reply for months,
            # patch applied here.
            end_mask = tf.keras.backend.cast(
                tf.keras.backend.greater(mask, self.shift_left(mask)),
                tf.keras.backend.floatx(),
            )
            energy = energy + start_mask * start
            energy = energy + end_mask * end
        return energy

    def get_viterbi_decoding(self, input_energy, nwords):
        pred_ids, _ = crf_decode(input_energy, self.chain_kernel, nwords)

        return pred_ids

    def get_config(self):
        # used for loading model from disk
        config = {
            "units":
            self.units,
            "use_boundary":
            self.use_boundary,
            "use_bias":
            self.use_bias,
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
            tf.keras.constraints.serialize(self.bias_constraint),
            "input_dim":
            self.input_dim,
        }
        base_config = super(CRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:2]
        return output_shape

    def compute_mask(self, input, mask=None):
        if mask is not None:
            # transform mask from shape (?, ?) to (?, )
            new_mask = tf.keras.backend.any(mask, axis=1)
            return new_mask

        return mask

    def get_decode_result(self, logits, mask):
        nwords = tf.keras.backend.cast(tf.keras.backend.sum(mask, 1), tf.int64)

        pred_ids, _ = crf_decode(logits, self.chain_kernel, nwords)

        return pred_ids

    def get_negative_log_likelihood(self, y_true):
        y_preds = self.logits

        nwords = self.nwords

        y_preds = tf.keras.backend.cast(y_preds, tf.float32)
        y_true = tf.keras.backend.cast(y_true, tf.int32)
        nwords = tf.keras.backend.cast(nwords, tf.int32)
        self.chain_kernel = tf.keras.backend.cast(self.chain_kernel,
                                                  tf.float32)

        log_likelihood, _ = crf_log_likelihood(y_preds, y_true, nwords,
                                               self.chain_kernel)

        return -log_likelihood

    def get_accuracy(self, y_true, y_pred):
        judge = tf.keras.backend.cast(tf.keras.backend.equal(y_pred, y_true),
                                      tf.keras.backend.floatx())
        if self.mask is None:
            return tf.keras.backend.mean(judge)
        else:
            mask = tf.keras.backend.cast(self.mask, tf.keras.backend.floatx())
            return (tf.keras.backend.sum(judge * mask) /
                    tf.keras.backend.sum(mask))

    def _dense_layer(self, input_):
        return self.activation(
            tf.keras.backend.dot(input_, self.kernel) + self.bias)
