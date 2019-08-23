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
"""Tests for MAPCallback."""
import tensorflow as tf
from tensorflow_addons.callbacks import map_callback
import numpy as np


def generate_data():
    ground_truth = np.array([[0, 0, 4, 13, 79, 154], [1, 0, 93, 37, 194, 121],
                             [2, 0, 277, 2, 444, 101], [3, 0, 469, 4, 552, 91],
                             [4, 0, 11, 152, 84, 250],
                             [5, 0, 516, 5, 638, 410]])
    detection_result = np.array([[0, 0.529134, 3, 12, 78, 153],
                                 [1, 0.523199, 92, 37, 193, 121],
                                 [6, 0.386569, 63, 77, 560, 477],
                                 [1, 0.374142, 292, 6, 438, 99],
                                 [3, 0.273336, 436, 0, 564, 105],
                                 [0, 0.346044, 13, 20, 124, 173]])
    return [[ground_truth, detection_result]]


class MAPCallbackTest(tf.test.TestCase):
    def test_calculate_ap(self):
        class_names = [
            "pottedplant", "tvmonitor", "shelf", "windowblind", "coffeetable",
            "door", "refrigerator"
        ]
        callback = map_callback.MAPCallback(generate_data, class_names)
        APs = callback.calculate_aps()
        for cls in range(len(class_names)):
            if cls in APs:
                print(class_names[cls] + ' ap: ', APs[cls])
        mAP = np.mean([APs[cls] for cls in APs])
        print('mAP: ', mAP)


if __name__ == '__main__':
    tf.test.main()
