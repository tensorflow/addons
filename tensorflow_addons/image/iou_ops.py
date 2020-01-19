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
"""Implements IoUs."""

import tensorflow as tf
import math
import numpy as np
from typing import Union

CompatibleFloatTensorLike = Union[tf.Tensor, float, np.float32, np.float64]


def _get_v(
    b1_height: CompatibleFloatTensorLike,
    b1_width: CompatibleFloatTensorLike,
    b2_height: CompatibleFloatTensorLike,
    b2_width: CompatibleFloatTensorLike,
) -> tf.Tensor:
    @tf.custom_gradient
    def _get_grad_v(height, width):
        arctan = tf.atan(tf.math.divide_no_nan(b1_width, b1_height)) - tf.atan(
            tf.math.divide_no_nan(width, height)
        )
        v = 4 * ((arctan / math.pi) ** 2)

        def _grad_v(dv):
            gdw = dv * 8 * arctan * height / (math.pi ** 2)
            gdh = -dv * 8 * arctan * width / (math.pi ** 2)
            return [gdh, gdw]

        return v, _grad_v

    return _get_grad_v(b2_height, b2_width)


def _common_iou(
    b1: CompatibleFloatTensorLike, b2: CompatibleFloatTensorLike, mode: str = "iou"
) -> tf.Tensor:
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['iou', 'ciou', 'diou', 'giou'], decided to calculate IoU or CIoU or DIoU or GIoU.

    Returns:
        IoU loss float `Tensor`.
    """
    b1 = tf.convert_to_tensor(b1)
    if not b1.dtype.is_floating:
        b1 = tf.cast(b1, tf.float32)
    b2 = tf.cast(b2, b1.dtype)

    def _inner():
        zero = tf.convert_to_tensor(0.0, b1.dtype)
        b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
        b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
        b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
        b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
        b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
        b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
        b1_area = b1_width * b1_height
        b2_area = b2_width * b2_height

        intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
        intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
        intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
        intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
        intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
        intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
        intersect_area = intersect_width * intersect_height

        union_area = b1_area + b2_area - intersect_area
        iou = tf.math.divide_no_nan(intersect_area, union_area)
        if mode == "iou":
            return iou

        elif mode in ["ciou", "diou"]:
            enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
            enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
            enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
            enclose_xmax = tf.maximum(b1_xmax, b2_xmax)

            b1_center = tf.stack([(b1_ymin + b1_ymax) / 2, (b1_xmin + b1_xmax) / 2])
            b2_center = tf.stack([(b2_ymin + b2_ymax) / 2, (b2_xmin + b2_xmax) / 2])
            euclidean = tf.linalg.norm(b2_center - b1_center)
            diag_length = tf.linalg.norm(
                [enclose_ymax - enclose_ymin, enclose_xmax - enclose_xmin]
            )
            diou = iou - (euclidean ** 2) / (diag_length ** 2)
            if mode == "ciou":
                v = _get_v(b1_height, b1_width, b2_height, b2_width)
                alpha = tf.math.divide_no_nan(v, ((1 - iou) + v))
                return diou - alpha * v

            return diou
        elif mode == "giou":
            enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
            enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
            enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
            enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
            enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
            enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
            enclose_area = enclose_width * enclose_height
            giou = iou - tf.math.divide_no_nan(
                (enclose_area - union_area), enclose_area
            )
            return giou
        else:
            raise ValueError(
                "Value of mode should be one of ['iou','giou','ciou','diou']"
            )

    return tf.squeeze(_inner())


def iou(b1: CompatibleFloatTensorLike, b2: CompatibleFloatTensorLike) -> tf.Tensor:
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].

    Returns:
        IoU loss float `Tensor`.
    """
    return _common_iou(b1, b2, "iou")


def ciou(b1: CompatibleFloatTensorLike, b2: CompatibleFloatTensorLike) -> tf.Tensor:
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].

    Returns:
        CIoU loss float `Tensor`.
    """
    return _common_iou(b1, b2, "ciou")


def diou(b1: CompatibleFloatTensorLike, b2: CompatibleFloatTensorLike) -> tf.Tensor:
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].

    Returns:
        DIoU loss float `Tensor`.
    """
    return _common_iou(b1, b2, "diou")


def giou(b1: CompatibleFloatTensorLike, b2: CompatibleFloatTensorLike) -> tf.Tensor:
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].

    Returns:
        GIoU loss float `Tensor`.
    """
    return _common_iou(b1, b2, "giou")
