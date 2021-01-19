import tensorflow as tf
from typeguard import typechecked

from tensorflow_addons.utils.resource_loader import LazySO

_embedding_bag_so = LazySO("custom_ops/layers/_embedding_bag_ops.so")


def _embedding_bag(
    indices,
    values,
    weights=None,
    combiner="sum",
    name=None,
):
    """EmbeddingBag computation.

    See [PyTorch op](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html).

    Equivalent to tf.gather() followed by tf.reduce_sum() across the last dimension, with optional
    weights. Fusing these into a single op has massive benefits for execution speed and particularly
    memory usage, as the intermediate output of the gather never needs to be materialized.

    Args:
      indices: An int32 or int64 `Tensor` of the indices to gather from
          `values`. Must be at least 2-dimensional, as the last dimension
          will be summed out. Maximum value must be less than values.shape[0].
      values: A float32 `Tensor` from which to gather values. Must be rank 2.
      weights: A float32 `Tensor` of weights which will be applied to each of
          the gathered embedding vectors before the sum step.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of the format specified by `data_format`.
    """
    if weights is None:
        weights = tf.ones_like(indices, dtype=tf.float32)

    return _embedding_bag_so.ops.addons_embedding_bag(
        indices, values, weights, combiner=combiner.upper(), name=name
    )


@tf.RegisterGradient("Addons>EmbeddingBag")
def _embedding_bag_grad(op, grad):
    indices, values, weights = op.inputs[:3]
    op_call = _embedding_bag_so.ops.addons_embedding_bag_grad
    values_grad, weights_grad = op_call(indices, values, weights, grad)[
        :2
    ]  # Drop the dummy outputs
    return [None, values_grad, weights_grad]


@tf.keras.utils.register_keras_serializable(package="Addons")
class EmbeddingBag(tf.keras.layers.Layer):
    """EmbeddingBag Layer.

    See [PyTorch op](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html).

    Equivalent to tf.gather() followed by tf.reduce_sum() across the last dimension, with optional
    weights. Fusing these into a single op has massive benefits for execution speed and particularly
    memory usage, as the intermediate output of the gather never needs to be materialized.

    Input Shapes:
      indices: An int32 or int64 `Tensor` of the indices to gather from
          `values`. Must be at least 2-dimensional, as the last dimension
          will be summed out. Maximum value must be less than values.shape[0].
      values: A float32 `Tensor` from which to gather values. Must be rank 2.
      weights: A float32 `Tensor` of weights which will be applied to each of
          the gathered embedding vectors before the sum step.

    Output shape:
        indices.shape[:-1], values.shape[-1]
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        combiner="sum",
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
