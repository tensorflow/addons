import tensorflow as tf
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from typeguard import typechecked
from tensorflow_addons.utils.resource_loader import LazySO

_embeddingbag_so = LazySO("custom_ops/layers/_embeddingbag_ops.so")


def embeddingbag(
    indices,
    values,
    weights,
    name=None,
):
    """EmbeddingBag computation

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

    with tf.name_scope(name or "embedding_bag"):
        op_call = _embeddingbag_so.ops.addons_embedding_bag

        if weights is None:
            weights = tf.ones_like(indices, dtype=tf.float32)

        ret = op_call(
            indices,
            values,
            weights
        )
        return ret


@tf.RegisterGradient("Addons>EmbeddingBag")
def _embedding_bag_grad(op, grad):
    indices, values, weights = op.inputs[:3]
    op_call = _embeddingbag_so.ops.addons_embedding_bag_grad
    values_grad, weights_grad = op_call(indices, values, weights, grad)[:2]  # Drop the dummy outputs
    return [None, values_grad, weights_grad]


@tf.keras.utils.register_keras_serializable(package="Addons")
class EmbeddingBag(tf.keras.layers.Layer):
    """EmbeddingBag Layer

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

    def __init__(self, num_embeddings, embedding_dim, embeddings_initializer=None,
                 embeddings_regularizer=None, embeddings_constraint=None, mask_zero=False,
                 **kwargs):
        super(EmbeddingBag, self).__init__(**kwargs)
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError('Both `num_embeddings` and `embedding_dim` should be positive, '
                             'found num_embeddings {} and embedding_dim {}'.format(
                num_embeddings, embedding_dim))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if embeddings_initializer is None:
            self.embeddings_initializer = initializers.normal(mean=0, stddev=embedding_dim ** (-0.5))
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero

    def build(self, input_shape):
        self.embeddings = self.add_weight(
          shape=(self.num_embeddings, self.embedding_dim),
          initializer=self.embeddings_initializer,
          name='embeddings',
          regularizer=self.embeddings_regularizer,
          constraint=self.embeddings_constraint,
          experimental_autocast=False)
        self.built = True

    def call(self, indices, weights):
        return embeddingbag(indices, self.embeddings, weights)

    def get_config(self):
        config = {
            'num_embeddings': self.num_embeddings,
            'embeddings_dim': self.embeddings_dim,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer':
                regularizers.serialize(self.embeddings_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'embeddings_constraint':
                constraints.serialize(self.embeddings_constraint),
            'mask_zero': self.mask_zero,
            'input_length': self.input_length
        }
        base_config = super(EmbeddingBag, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))