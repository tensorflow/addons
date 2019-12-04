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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Addons')
@tf.function
def center_loss(feature, labels, num_classes):
    """Computes the center loss.

    Arguments:
        feature: feature with shape (batch_size, feat_dim)
        labels: ground truth labels with shape (batch_size)
    
    Return：
        loss: Tensor
    """
    len_features = feature.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)
    loss = tf.nn.l2_loss(feature - centers_batch)
    diff = centers_batch - feature
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    centers_update_op = tf.scatter_sub(centers, labels, diff)

    return loss, centers, centers_update_op


@tf.keras.utils.register_keras_serializable(package='Addons')
class CenterLoss(tf.keras.losses.Loss):
    """Computes the center loss.

    See: https://ydwen.github.io/papers/WenECCV16.pdf


    Args:
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `SUM_OVER_BATCH_SIZE`.
      name: Optional name for the op.
    """

    def __init__(self,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 feat_dim=3,
                 num_classes=5,
                 name="center_loss"):
        super(CenterLoss, self).__init__(reduction=reduction, name=name)
        self.feat_dim = feat_dim
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        return center_loss(feature, labels, self.num_classes)

    def get_config(self):
        config = {
            "margin": self.margin,
        }
        base_config = super(CenterLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
