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
"""Implements IoU losses."""

import tensorflow as tf
from tensorflow_addons.image import iou, ciou, diou, giou


def _common_loss(y_pred, y_true, iou_fn):
    y_pred = tf.convert_to_tensor(y_pred)
    if not y_pred.dtype.is_floating:
        y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)
    return 1 - iou_fn(y_pred, y_true)


@tf.keras.utils.register_keras_serializable(package='Addons')
class IoULoss(tf.keras.losses.Loss):
    """Implements the IoU loss function.

    Usage:

    ```python
    il = tfa.losses.IoULoss()
    boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
    loss = il(boxes1, boxes2)
    print('Loss: ', loss.numpy())  # Loss: [0.875, 1.]
    ```
    Usage with tf.keras API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=tfa.losses.IoULoss())
    ```
    """

    def call(self, y_true, y_pred):
        return iou_loss(y_true, y_pred)


@tf.keras.utils.register_keras_serializable(package='Addons')
def iou_loss(y_pred, y_true):
    return _common_loss(y_pred, y_true, iou)


@tf.keras.utils.register_keras_serializable(package='Addons')
class CIoULoss(tf.keras.losses.Loss):
    """Implements the CIoU loss function.

    CIoU loss was first introduced in the
    [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression]
    (https://arxiv.org/abs/1911.08287).
    CIoU is an enhancement for models which use IoU in object detection.

    Usage:

    ```python
    cl = tfa.losses.CIoULoss()
    boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
    loss = cl(boxes1, boxes2)
    print('Loss: ', loss.numpy())  # Loss: [1.40889336451548444, 1.5487535732151345]
    ```
    Usage with tf.keras API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=tfa.losses.CIoULoss())
    ```
    """

    def call(self, y_true, y_pred):
        return ciou_loss(y_pred, y_true)


@tf.keras.utils.register_keras_serializable(package='Addons')
def ciou_loss(y_pred, y_true):
    return _common_loss(y_pred, y_true, ciou)


@tf.keras.utils.register_keras_serializable(package='Addons')
class DIoULoss(tf.keras.losses.Loss):
    """Implements the DIoU loss function.

    DIoU loss was first introduced in the
    [Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression]
    (https://arxiv.org/abs/1911.08287).
    DIoU is an enhancement for models which use IoU in object detection.

    Usage:

    ```python
    dl = tfa.losses.DIoULoss()
    boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
    loss = dl(boxes1, boxes2)
    print('Loss: ', loss.numpy())  # Loss: [1.4065315315315314, 1.5315315315315314]
    ```
    Usage with tf.keras API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=tfa.losses.DIoULoss())
    ```

    Args:
      mode: one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.
    """

    def call(self, y_true, y_pred):
        return diou_loss(y_pred, y_true)


@tf.keras.utils.register_keras_serializable(package='Addons')
def diou_loss(y_pred, y_true):
    return _common_loss(y_pred, y_true, diou)


@tf.keras.utils.register_keras_serializable(package='Addons')
class GIoULoss(tf.keras.losses.Loss):
    """Implements the GIoU loss function.

    GIoU loss was first introduced in the
    [Generalized Intersection over Union:
    A Metric and A Loss for Bounding Box Regression]
    (https://giou.stanford.edu/GIoU.pdf).
    GIoU is an enhancement for models which use IoU in object detection.

    Usage:

    ```python
    gl = tfa.losses.GIoULoss()
    boxes1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    boxes2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
    loss = gl(boxes1, boxes2)
    print('Loss: ', loss.numpy())  # Loss: [1.07500000298023224, 1.9333333373069763]
    ```
    Usage with tf.keras API:

    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile('sgd', loss=tfa.losses.GIoULoss())
    ```

    Args:
      mode: one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.
    """

    def call(self, y_true, y_pred):
        return giou_loss(y_pred, y_true)


@tf.keras.utils.register_keras_serializable(package='Addons')
def giou_loss(y_pred, y_true):
    return _common_loss(y_pred, y_true, giou)
