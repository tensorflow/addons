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
"""Implements GIOU loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
class GIOULoss(tf.keras.losses.Loss):
    """Implements the GIOU loss function.

    GIOU loss was first introduced in the Generalized Intersection over Union
    paper (https://giou.stanford.edu/GIoU.pdf). GIOU is a enhance for model
    which use IOU in object detection.

    Usage:

    ```python
    gl = tfa.losses.GIOU()
    loss = gl(
      [[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 2, 2]],
      [[0, 0, 1, 1], [0, 0, 1, 2], [2, 2, 4, 4]])
    print('Loss: ', loss.numpy())  # Loss: [0, 0.5, 1.25]
    ```
    Usage with tf.keras API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=tf.keras.losses.GIOULoss())
    ```

    Args
        reduction
      name: Op name

    Returns:
      GIOU loss float `Tensor`.
    """

    def __init__(self,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name='giou_loss'):
        super(GIOULoss, self).__init__(name=name, reduction=reduction)

    def call(self, y_true, y_pred):
        return giou_loss(y_true, y_pred)

    def get_config(self):
        base_config = super(GIOULoss, self).get_config()
        return base_config


@keras_utils.register_keras_custom_object
def giou_loss(y_true, y_pred):
    """
    Args
        y_true: true targets tensor.
        y_pred: predictions tensor.

    Returns:
        GIOU loss float `Tensor`.
    """

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    giou = giou_calculate(y_pred, y_true)

    # compute the final loss and return
    return 1 - giou


def giou_calculate(b1, b2):
    """
    Args
        b1: bbox.
        b1: the other bbox.

    Returns:
        iou float `Tensor`, between [-1, 1].
    """
    return do_iou_calculate(b1, b2, mode='giou')


def do_iou_calculate(b1, b2, mode='iou'):
    """
    Args
        b1: bbox.
        b1: the other bbox.
        mode: one of ['iou', 'giou']

    Returns:
        iou float `Tensor`.
    """
    b1_ymin = tf.minimum(b1[:, 0], b1[:, 2])
    b1_xmin = tf.minimum(b1[:, 1], b1[:, 3])
    b1_ymax = tf.maximum(b1[:, 0], b1[:, 2])
    b1_xmax = tf.maximum(b1[:, 1], b1[:, 3])
    b2_ymin = tf.minimum(b2[:, 0], b2[:, 2])
    b2_xmin = tf.minimum(b2[:, 1], b2[:, 3])
    b2_ymax = tf.maximum(b2[:, 0], b2[:, 2])
    b2_xmax = tf.maximum(b2[:, 1], b2[:, 3])
    b1_area = (b1_ymax - b1_ymin) * (b1_xmax - b1_xmin)
    b2_area = (b2_ymax - b2_ymin) * (b2_xmax - b1_xmin)
    illegal_area_indexes = tf.cast(
        tf.where(tf.logical_or(b1_area < 0, b2_area < 0)), tf.int32)
    valid_area_indexes = tf.cast(
        tf.where(tf.logical_and(b1_area >= 0, b2_area >= 0)), tf.int32)

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_area = tf.maximum(0,
                                intersect_ymax - intersect_ymin) * tf.maximum(
                                    0, intersect_xmax - intersect_xmin)

    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / union_area
    indices = [valid_area_indexes, illegal_area_indexes]
    data = [
        tf.gather(iou, valid_area_indexes),
        tf.zeros([tf.shape(illegal_area_indexes)[0], 1], tf.float64)
    ]
    iou = tf.dynamic_stitch(indices, data)
    if mode == 'iou':
        return iou
    bc_ymin = tf.minimum(b1_ymin, b2_ymin)
    bc_xmin = tf.minimum(b1_xmin, b2_xmin)
    bc_ymax = tf.maximum(b1_ymax, b2_ymax)
    bc_xmax = tf.maximum(b1_xmax, b2_xmax)

    enclose_area = tf.maximum(0, bc_ymax - bc_ymin) * tf.maximum(
        0, bc_xmax - bc_xmin)
    giou = iou - tf.cast(enclose_area - union_area, tf.float64) / (
        tf.cast(enclose_area, tf.float64) + 1e-8)
    return giou
