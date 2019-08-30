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

        self.vars = []
        self.ground_truth = tf.constant([], shape=[0, 8])
        self.detection_result = tf.constant([], shape=[0, 7])
        self.index = 0.

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # add image index to data
        indexs = tf.fill([tf.shape(y_pred)[0], 1], self.index)
        y_pred = tf.concat([y_pred, indexs], 1)
        self.detection_result = tf.concat([self.detection_result, y_pred], 0)
        # add image index to data
        indexs = tf.fill([tf.shape(y_true)[0], 1], self.index)
        y_true = tf.concat([y_true, indexs], 1)
        self.ground_truth = tf.concat([self.ground_truth, y_true], 0)
        self.index += 1.

    @tf.function
    def result(self):
        APs = tf.TensorArray(tf.float32, self.num_classes)
        APs.write(0, 0)
        # separate detection result by class
        cls_results = tf.dynamic_partition(
            self.detection_result,
            tf.cast(self.detection_result[:, 0], tf.int32), self.num_classes)
        cls = 0
        for cls_result in cls_results:
            npos = 0
            nd = tf.shape(cls_result)[0]
            tp = tf.fill([nd], 0.)
            fp = tf.fill([nd], 0.)
            # sort result by confidence
            cls_result = tf.sort(cls_result, 1, 'DESCENDING')
            bbox = cls_result[:, 2:6]

            # loop every cls_result image indexes
            for index in cls_result[:, 6]:
                ground_truth = tf.boolean_mask(
                    self.ground_truth, self.ground_truth[:, 7] == index)
                index = tf.cast(index, tf.int32)
                objs = tf.boolean_mask(ground_truth, ground_truth[:, 0] == cls)
                ovmax = -float('inf')
                BBGT = objs[:, 1:5]
                npos += tf.shape(objs)[0]
                ixmin = tf.maximum(BBGT[:, 0], bbox[index, 0])
                iymin = tf.maximum(BBGT[:, 1], bbox[index, 1])
                ixmax = tf.minimum(BBGT[:, 2], bbox[index, 2])
                iymax = tf.minimum(BBGT[:, 3], bbox[index, 3])
                iw = tf.maximum(ixmax - ixmin + 1., 0.)
                ih = tf.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bbox[index, 2] - bbox[index, 0] + 1.) *
                       (bbox[index, 3] - bbox[index, 1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                # calculate iou
                overlaps = inters / uni
                ovmax = tf.maximum(tf.reduce_max(overlaps), ovmax)
                idx = tf.argmax(overlaps)
                gt_match = objs[idx]
                # ensure max iou greater than iou threshold
                if ovmax > self.iou_threshold:
                    # difficult is useless for now
                    if tf.equal(gt_match[5], 0):
                        # if item is not used,set it used and set tp=1
                        if tf.equal(gt_match[6], 0):
                            tp[idx] = 1.
                            gt_match[6] = 1
                        else:
                            fp[idx] = 1.
                else:
                    fp[idx] = 1.


            fp = tf.cumsum(fp)
            tp = tf.cumsum(tp)
            rec = tp / tf.maximum(tf.cast(npos, tf.float32), 1e-8)
            prec = tp / tf.maximum(tp + fp, 1e-8)
            ap = self._voc_ap(rec, prec)
            APs.write(cls, ap)
            cls += 1
        return APs

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "iou_threshold": self.iou_threshold,
        }
        base_config = super(MAP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        self.ground_truth = tf.constant([], shape=[0, 8])
        self.detection_result = tf.constant([], shape=[0, 7])

    def _voc_ap(self, recall, precision):
        # voc2010+ mAP algorithm,use AUC
        mrec = tf.concat([[0.], recall, [1.]], 0)
        mpre = tf.concat([[0.], precision, [0.]], 0)
        mpre_arr = tf.TensorArray(tf.float32, tf.shape(mpre)[0])
        last_idx = tf.shape(mpre)[0] - 1
        mpre_arr.write(last_idx, mpre[last_idx])
        for i in tf.range(last_idx, 0, -1):
            mpre_arr.write(i - 1, tf.maximum(mpre[i - 1], mpre[i]))
        mpre = mpre_arr.stack()
        idxs = tf.where(mrec[1:] != mrec[:-1])[:, 0]
        ap = 0
        for i in idxs:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
        return ap
