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
"""Implements center loss."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Addons')
@tf.function
def center_loss(labels, feature, alpha):
    """Computes the center loss.

    Arguments:
        labels: ground truth labels with shape (batch_size)
        feature: feature with shape (batch_size, num_classes)
        alpha: a scalar between 0-1 to control the leanring rate of the centers
        num_classes: an `int`. The number of possible classes
    
    Return：
        loss: Tensor
    """
    len_features = feature.get_shape()[1]
    num_classes = labels.shape[-1]
    centers = tf.Variable(tf.constant(0.0, shape=[num_classes, len_features]), name='centers', 
        dtype=tf.float32)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)
    diff = centers_batch - feature
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    # update centers
    centers_update_op = tf.scatter_sub(centers, labels, diff)

    # enforce computing centers before updating center loss
    with tf.control_dependencies([centers_update_op]):
        # compute center-loss
        loss = tf.nn.l2_loss(features - centers_batch)

    return loss


@tf.keras.utils.register_keras_serializable(package='Addons')
class CenterLoss(tf.keras.losses.Loss):
    """Computes the center loss.

    See: https://ydwen.github.io/papers/WenECCV16.pdf

    The loss was designed to develop an effectively improve the discriminative
    power of the deep learned features. In fact, the key is to minimize the 
    intra-class variations while keeping the features of different classes 
    separable. Usually combining with softmax losss to jointly supervise the 
    CNNs to have a better result. 

    We expect labels `y_pred` must be the output of the last fully connected 
    layer, which is 2-D float `Tensor` of l2 normalized embedding vectors and 
    `y_true` to be provided as 1-D integer `Tensor` with shape [batch_size] 
    of multi-class integer labels. 

    Args:
      name: Optional name for the op.    
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `SUM_OVER_BATCH_SIZE`.
    """

    def __init__(self,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 alpha=0.2,
                 name="center_loss"):
        super(CenterLoss, self).__init__(reduction=reduction, name=name)
        self.alpha = alpha

    def call(self, y_true, y_pred, x):
        return center_loss(y_true, y_pred, self.alpha)`

    def get_config(self):
        config = {
            "alpha": self.alpha,
        }
        base_config = super(CenterLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
