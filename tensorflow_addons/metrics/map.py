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
import tensorflow as tf
import numpy as np


class MAP(tf.metrics.Metric):
    def __init__(self, num_classes, iou_threshold=0.5, name='mAP'):
        super(MAP, self).__init__(name=name)
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError(
                "iou threshold value should be greater equal than zero and less equal than one"
            )

        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.accumulate_true_positives = self.add_weight(
            'accumulate_true_positives',
            shape=[self.num_classes],
            initializer='zeros',
            dtype=tf.float32)
        self.accumulate_false_positives = self.add_weight(
            'accumulate_false_positives',
            shape=[self.num_classes],
            initializer='zeros',
            dtype=tf.float32)
        self.accumulate_ground_truth = self.add_weight(
            'accumulate_ground_truth',
            shape=[self.num_classes],
            initializer='zeros',
            dtype=tf.float32)

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        total_tp_count = tf.constant([], shape=[0])
        total_fp_count = tf.constant([], shape=[0])
        total_ground_truth_count = tf.constant([], shape=[0])
        for class_index in range(self.num_classes):
            y_true_class = tf.gather(y_true,
                                     tf.where(y_true[:, 0] == class_index))
            y_pred_class = tf.gather(y_pred,
                                     tf.where(y_pred[:, 0] == class_index))
            y_true_boxes = y_true_class[:, 1:5]
            y_pred_boxes = y_pred_class[:, 2:]

            def calcluate_iou(y_true_box):
                ious = self.iou(tf.expand_dims(y_true_box, 0), y_pred_boxes)
                ious = tf.gather(ious, tf.where(ious > self.iou_threshold))
                if tf.equal(tf.shape(ious), 0):
                    return -1
                return tf.arg_max(ious)

            iou_result = tf.map_fn(calcluate_iou, y_true_boxes)
            tp = tf.cast(iou_result >= 0, tf.float32)
            tp_count = tf.math.count_nonzero(tp)
            fp_count = tf.shape(tp)[0] - tp_count
            ground_truth_count = tf.shape(y_true)[0]
            total_tp_count = tf.concat([total_tp_count, [tp_count]], 0)
            total_fp_count = tf.concat([total_fp_count, [fp_count]], 0)
            total_ground_truth_count = tf.concat(
                [total_ground_truth_count, [ground_truth_count]], 0)

        self.accumulate_true_positives.assign_add(total_tp_count)
        self.accumulate_false_positives.assign_add(total_fp_count)
        self.accumulate_ground_truth.assign_add(total_ground_truth_count)

    def iou(self, b1, b2):
        """
        Args
            b1: bbox.
            b1: the other bbox.
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
        intersect_area = tf.maximum(
            0, intersect_ymax - intersect_ymin) * tf.maximum(
                0, intersect_xmax - intersect_xmin)

        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / union_area
        indices = [valid_area_indexes, illegal_area_indexes]
        data = [
            tf.gather(iou, valid_area_indexes),
            tf.zeros([tf.shape(illegal_area_indexes)[0], 1], tf.float64)
        ]
        iou = tf.dynamic_stitch(indices, data)
        return iou

    @tf.function
    def result(self):
        true_positives_classes = tf.unstack(self.accumulate_true_positives)
        false_positives_classes = tf.unstack(self.accumulate_false_positives)
        ground_truth_classes = tf.unstack(self.accumulate_ground_truth)
        aps = tf.constant([], shape=[self.num_classes])
        for (true_positives_class,
             false_positives_class, ground_truth_class) in zip(
                 true_positives_classes, false_positives_classes,
                 ground_truth_classes):
            recall_class = tf.math.divide_no_nan(true_positives_class,
                                                 ground_truth_class)
            precision_class = tf.math.divide_no_nan(
                true_positives_class,
                true_positives_class + false_positives_class)
            sorted_idx = tf.arg_sort(recall_class)
            recall_class = tf.sort(recall_class)
            precision_class = tf.gather(precision_class, sorted_idx)
            _, idx, count = tf.unique_with_counts(recall_class)
            partitions = tf.dynamic_partition(precision_class, idx,
                                              tf.shape(count)[0])
            results = tf.constant([])
            for partition in partitions:
                max_precision = tf.reduce_max(partition)
                results = tf.concat([results, max_precision], 0)
            ap = tf.reduce_mean(results, 0)
            aps = tf.concat([aps, [ap]], 0)
        return tf.reduce_mean(aps)

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "iou_threshold": self.iou_threshold,
        }
        base_config = super(MAP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        self.accumulate_true_positives.assign(
            np.zeros(self.num_classes), np.float32)
        self.accumulate_false_positives.assign(
            np.zeros(self.num_classes), np.float32)
        self.accumulate_ground_truth.assign(
            np.zeros(self.num_classes), np.float32)
