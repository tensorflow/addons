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
"""NetVLAD keras layer."""

import math
import tensorflow as tf
from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class NetVLAD(tf.keras.layers.Layer):
    """Applies NetVLAD to the input.

        This is a fully-differentiable version of "Vector of Locally Aggregated Descriptors" commonly used in image
        retrieval. It is also used in audio retrieval, and audio represenation learning (ex
        "Towards Learning a Universal Non-Semantic Representation of Speech", https://arxiv.org/abs/2002.12764).

        "NetVLAD: CNN architecture for weakly supervised place recognition"
        Relja Arandjelovic, Petr Gronat, Akihiko Torii, Tomas Pajdla, Josef Sivic.
        https://arxiv.org/abs/1511.07247

    Arguments:
        num_clusters: The number of clusters to use.
    Input shape:
        3D tensor with shape: `(batch_size, time, feature_dim)`.
    Output shape:
        2D tensor with shape: `(batch_size, feature_dim * num_clusters)`.
    """

    @typechecked
    def __init__(self, num_clusters: int, **kwargs):
        super().__init__(**kwargs)
        if num_clusters <= 0:
            raise ValueError("`num_clusters` must be greater than 1: %i" % num_clusters)
        self.num_clusters = num_clusters

    def build(self, input_shape):
        """Keras build method."""
        feature_dim = input_shape[-1]
        if not isinstance(feature_dim, int):
            feature_dim = feature_dim.value
        self.fc = tf.keras.layers.Dense(
            units=self.num_clusters,
            activation=tf.nn.softmax,
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        )
        self.cluster_centers = self.add_weight(
            name="cluster_centers",
            shape=(1, feature_dim, self.num_clusters),
            initializer=tf.keras.initializers.TruncatedNormal(
                stddev=1.0 / math.sqrt(feature_dim)
            ),
            trainable=True,
        )
        super(NetVLAD, self).build(input_shape)

    def call(self, frames):
        """Apply the NetVLAD module to the given frames.

        Args:
            frames: A tensor with shape [batch_size, max_frames, feature_dim].

        Returns:
            A tensor with shape [batch_size, feature_dim * num_clusters].

        Raises:
            ValueError: If the `feature_dim` of input is not defined.
        """
        frames.shape.assert_has_rank(3)
        feature_dim = frames.shape.as_list()[-1]
        if feature_dim is None:
            raise ValueError("Last dimension must be defined.")
        max_frames = tf.shape(frames)[-2]

        # Compute soft-assignment from frames to clusters.
        # Essentially: softmax(w*x + b), although BN can be used instead of bias.
        frames = tf.reshape(frames, (-1, feature_dim))
        activation = self.fc(frames)
        activation = tf.reshape(activation, (-1, max_frames, self.num_clusters))

        # Soft-count of number of frames assigned to each cluster.
        # Output shape: [batch_size, 1, num_clusters]
        a_sum = tf.math.reduce_sum(activation, axis=-2, keepdims=True)

        # Compute sum_{i=1}^N softmax(w_k * x_i + b_k) * c_k(j),
        # for all clusters and dimensions.
        # Output shape: [batch_size, feature_dim, num_clusters]
        a = a_sum * self.cluster_centers

        # Compute sum_{i=1}^N softmax(w_k * x_i + b_k) * x_i(j),
        # for all clusters and dimensions.
        # Output shape: (batch_size, feature_dim, num_clusters)
        frames = tf.reshape(frames, (-1, max_frames, feature_dim))
        b = tf.transpose(
            tf.matmul(tf.transpose(activation, perm=(0, 2, 1)), frames), perm=(0, 2, 1)
        )

        # Output shape: (batch_size, feature_dim, num_clusters)
        vlad = b - a

        # Normalize first across the feature dimensions.
        vlad = tf.nn.l2_normalize(vlad, 1)

        # Output shape: [batch_size, feature_dim * num_clusters]
        vlad = tf.reshape(vlad, (-1, feature_dim * self.num_clusters))

        # Renormalize across both the feature dimensions (already normalized) and
        # the cluster centers.
        vlad = tf.nn.l2_normalize(vlad, 1)

        return vlad

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([input_shape[0], input_shape[-1] * self.num_clusters])

    def get_config(self):
        config = {"num_clusters": self.num_clusters}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
