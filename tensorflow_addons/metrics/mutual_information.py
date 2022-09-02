# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
from typing import Optional

import tensorflow as tf

from tensorflow_addons.metrics._streaming_buffer import StreamingBuffer
from tensorflow_addons.utils.types import AcceptableDTypes


@tf.keras.utils.register_keras_serializable(package="Addons")
class MutualInformation(StreamingBuffer):
    """Implementation based on the mutual information estimator described in
    'Weihao Gao, Sreeram Kannan, Sewoong Oh, and Pramod Viswanath. Estimating
    mutual information for discrete-continuous mixtures'

    The estimator works on any kind of distribution: discrete,
    continuous or a mix of both.

    This implementation has a memory complexity of
    O(`buffer_size`x`compute_batch_size`) and a time complexity of
    O(`buffer_size`^2). They can be adjusted with the corresponding parameters.
    Smaller value of `buffer_size` increases the bias of the estimator while
    small value of `compute_batch_size` increases the computation time.

    Args:
        n_neighbors: The number of nearest neighbors, larger value reduces
        variance but may introduce a bias.

        buffer_size: Size of the batches on which the MI will be estimated,
        larger value reduces the bias, but increases the compute complexity.

        compute_batch_size: Size of the batches used for computing the
        distances, larger value increases the memory usage.

        name: (Optional) String name of the metric instance.
        dtype: (Optional) Data type of the metric result.

    Returns:
        mutual information: double.

    Usage:

    >>> metric = tfa.metrics.MutualInformation(n_neighbors=1)
    >>> metric.update_state([1, 0, 1, 0, 1], [0, 1, 0, 1, 0])
    >>> metric.result().numpy()
    0.88665357733

    """

    def __init__(
        self,
        n_neighbors: int = 3,
        buffer_size: int = 1024,
        compute_batch_size: int = 1024,
        name: Optional[str] = None,
        dtype: AcceptableDTypes = None,
    ):
        super().__init__(buffer_size, name, dtype)
        """Creates a `MutualInformation` instance."""
        self.n_neighbors = n_neighbors
        self.compute_batch_size = compute_batch_size
        self.epsilon = self.add_weight(
            "epsilon", (), initializer="zeros", dtype=tf.float64
        )
        self.count = self.add_weight("count", (), initializer="zeros", dtype=tf.int64)

    @tf.function
    def _compute_epsilon(self, x, y, n_neighbors):
        size = tf.shape(x)[0]
        processed_data = 0
        sum_epsilon = tf.constant(0.0, tf.float64)
        fsize = tf.cast(size, tf.float64)
        while processed_data < size:
            sub_x = x[processed_data : processed_data + self.compute_batch_size]
            sub_y = y[processed_data : processed_data + self.compute_batch_size]
            x_distances = tf.abs(tf.expand_dims(x, 0) - tf.expand_dims(sub_x, -1))
            y_distances = tf.abs(tf.expand_dims(y, 0) - tf.expand_dims(sub_y, -1))
            distances = tf.maximum(x_distances, y_distances)
            sorted_distances = tf.sort(distances, axis=1)
            radius = sorted_distances[:, n_neighbors]
            radius = tf.expand_dims(radius, axis=-1)
            n_x = tf.reduce_sum(
                tf.cast(
                    (x_distances < radius) | ((radius == 0.0) & (x_distances == 0.0)),
                    tf.int32,
                ),
                axis=1,
            )
            n_y = tf.reduce_sum(
                tf.cast(
                    (y_distances < radius) | ((radius == 0.0) & (y_distances == 0.0)),
                    tf.int32,
                ),
                axis=1,
            )
            k = (
                tf.reduce_sum(
                    tf.cast(sorted_distances[:, n_neighbors:] == 0.0, tf.int32), axis=1
                )
                + n_neighbors
            )
            eps = (
                tf.math.digamma(tf.cast(k, tf.float64))
                - tf.math.digamma(tf.cast(n_x, tf.float64))
                - tf.math.digamma(tf.cast(n_y, tf.float64))
            )
            sum_epsilon = sum_epsilon + tf.reduce_sum(eps)
            processed_data = processed_data + tf.shape(sub_x)[0]

        return sum_epsilon + fsize * tf.math.log(fsize)

    @tf.function
    def _update_state(self, y_true_buffer, y_pred_buffer):
        epsilon = self._compute_epsilon(y_true_buffer, y_pred_buffer, self.n_neighbors)
        self.epsilon.assign_add(epsilon)
        self.count.assign_add(tf.shape(y_true_buffer, out_type=tf.int64)[0])

    def _result(self):
        return self.epsilon / tf.cast(self.count, tf.float64)

    def reset_state(self):
        """Resets all of the metric state variables."""
        super().reset_state()
        self.epsilon.assign(0.0)
        self.count.assign(0)

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "n_neighbors": self.n_neighbors,
            "compute_batch_size": self.compute_batch_size,
        }
        base_config = super().get_config()
        return {**base_config, **config}
