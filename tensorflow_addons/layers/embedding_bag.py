# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf
from typeguard import typechecked

from tensorflow_addons.utils.types import Constraint, Initializer, Regularizer
from tensorflow_addons.utils.resource_loader import LazySO

_embedding_bag_so = LazySO("custom_ops/layers/_embedding_bag_ops.so")


def _embedding_bag(
    indices,
    params,
    weights=None,
    combiner="sum",
    name=None,
):
    """EmbeddingBag computation.

    See [PyTorch op](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html).

    Equivalent to tf.gather() followed by tf.reduce_{sum,mean}() across the last dimension, with optional
    weights. Fusing these into a single op has massive benefits for execution speed and particularly
    memory usage, as the intermediate output of the gather never needs to be materialized.

    Args:
      indices: An int32 or int64 `Tensor` of the indices to gather from
          `params`. Must be at least 2-dimensional, as the last dimension
          will be summed out. Maximum value must be less than params.shape[0].
      params: A float32 `Tensor` from which to gather params. Must be rank 2.
      weights: A float32 `Tensor` of weights which will be applied to each of
          the gathered embedding vectors before the sum step.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of the format specified by `data_format`.
    """
    if weights is None:
        weights = tf.ones_like(indices, dtype=params.dtype)
    elif combiner != "sum":
        raise RuntimeError(
            "Combiner mode must be 'sum' when weights are supplied to EmbeddingBag!"
        )

    return _embedding_bag_so.ops.addons_embedding_bag(
        indices, params, weights, combiner=combiner.upper(), name=name
    )


@tf.RegisterGradient("Addons>EmbeddingBag")
def _embedding_bag_grad(op, grads):
    indices, params, weights = op.inputs[:3]
    combiner = op.get_attr("combiner")
    value_grads, weight_grads = _embedding_bag_so.ops.addons_embedding_bag_grad(
        indices, params, weights, grads, combiner=combiner
    )
    return [None, value_grads, weight_grads]


@tf.keras.utils.register_keras_serializable(package="Addons")
class EmbeddingBag(tf.keras.layers.Layer):
    """EmbeddingBag Layer.

    See [PyTorch op](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html).

    Equivalent to tf.gather() followed by tf.reduce_sum() across the last dimension, with optional
    weights. Fusing these into a single op has massive benefits for execution speed and particularly
    memory usage, as the intermediate output of the gather never needs to be materialized.

    Input Shapes:
      indices: An int32 or int64 `Tensor` of the indices to gather from
          `params`. Must be at least 2-dimensional, as the last dimension
          will be summed out. Maximum value must be less than params.shape[0].
      params: A float32 `Tensor` from which to gather params. Must be rank 2.
      weights: A float32 `Tensor` of weights which will be applied to each of
          the gathered embedding vectors before the sum step.

    Output shape:
        indices.shape[:-1], params.shape[-1]
    """

    @typechecked
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embeddings_initializer: Initializer = "uniform",
        embeddings_regularizer: Regularizer = None,
        embeddings_constraint: Constraint = None,
        mask_zero: bool = False,
        combiner: str = "sum",
        **kwargs,
    ):
        super(EmbeddingBag, self).__init__(**kwargs)
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError(
                "Both `input_dim` and `output_dim` should be positive, "
                "found input_dim {} and output_dim {}".format(input_dim, output_dim)
            )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = tf.keras.constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.combiner = combiner

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            name="embeddings",
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
        )
        self.built = True

    def call(self, indices, weights=None):
        return _embedding_bag(indices, self.embeddings, weights, combiner=self.combiner)

    def get_config(self):
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "embeddings_initializer": tf.keras.initializers.serialize(
                self.embeddings_initializer
            ),
            "embeddings_regularizer": tf.keras.regularizers.serialize(
                self.embeddings_regularizer
            ),
            "embeddings_constraint": tf.keras.constraints.serialize(
                self.embeddings_constraint
            ),
            "mask_zero": self.mask_zero,
            "input_length": self.input_length,
            "combiner": self.combiner,
        }
        base_config = super(EmbeddingBag, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
