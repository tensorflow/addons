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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
class MAPCallback(tf.keras.callbacks.Callback):
    def _voc_ap(self, rec, prec):
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def calculate_aps(self):
        true_res = []
        pred_res = []
        APs = {}
        for ground_truth, detection_result in self.generate():
            pred_res.append(detection_result)
            true_res.append(ground_truth)

        for cls in range(self.num_classes):
            true_res_cls = {}
            npos = 0
            ids = []
            scores = []
            bboxs = []
            for index in range(len(true_res)):
                item = [x for x in pred_res[index] if x[0] == cls]
                ids += [index for i in range(len(item))]
                scores += [x[1] for x in item]
                bboxs += [x[2:] for x in item]
                objs = [obj for obj in true_res[index] if obj[0] == cls]
                npos += len(objs)
                difficult = [x[1] for x in objs]
                BBGT = np.array([x[2:] for x in objs])
                true_res_cls[index] = {
                    'bbox': BBGT,
                    'difficult': np.array(difficult, dtype='bool'),
                    'det': [False] * len(objs)
                }
            if len(ids) == 0:
                APs[cls] = 0
                continue
            scores = np.array(scores, dtype='float32')
            bboxs = np.array(bboxs, dtype='float32')
            sorted_ind = np.argsort(-scores)
            bboxs = bboxs[sorted_ind, :]
            ids = [ids[x] for x in sorted_ind]

            nd = len(ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for j in range(nd):
                res = true_res_cls[ids[j]]
                bbox = bboxs[j, :].astype(float)
                ovmax = -np.inf
                BBGT = res['bbox'].astype(float)
                if BBGT.size > 0:
                    ixmin = np.maximum(BBGT[:, 0], bbox[0])
                    iymin = np.maximum(BBGT[:, 1], bbox[1])
                    ixmax = np.minimum(BBGT[:, 2], bbox[2])
                    iymax = np.minimum(BBGT[:, 3], bbox[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bbox[2] - bbox[0] + 1.) * (bbox[3] - bbox[1] + 1.)
                           + (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
                if ovmax > self.iou:
                    if not res['difficult'][jmax]:
                        if not res['det'][jmax]:
                            tp[j] = 1.
                            res['det'][jmax] = 1
                        else:
                            fp[j] = 1.
                else:
                    fp[j] = 1.

            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / np.maximum(float(npos), np.finfo(np.float64).eps)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self._voc_ap(rec, prec)
            APs[cls] = ap
        return APs

    def __init__(self, generate, class_names, iou=.5):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.iou = iou
        self.generate = generate

    def on_train_end(self, logs={}):
        logs = logs or {}
        APs = self.calculate_aps()
        for cls in range(self.num_classes):
            if cls in APs:
                print(self.class_names[cls] + ' ap: ', APs[cls])
        mAP = np.mean([APs[cls] for cls in APs])
        print('mAP: ', mAP)
        logs['mAP'] = mAP
