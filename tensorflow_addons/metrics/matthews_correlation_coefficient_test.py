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
"""Matthews Correlation Coefficient Test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow_addons.utils import test_utils
from tensorflow_addons.metrics import MatthewsCorrelationCoefficient


@test_utils.run_all_in_graph_and_eager_modes
class MatthewsCorrelationCoefficientTest(tf.test.TestCase):
    def test_config(self):
        # mcc object
        mcc1 = MatthewsCorrelationCoefficient(num_classes=1)
        self.assertEqual(mcc1.num_classes, 1)
        self.assertEqual(mcc1.dtype, tf.float32)
        # check configure
        mcc2 = MatthewsCorrelationCoefficient.from_config(mcc1.get_config())
        self.assertEqual(mcc2.num_classes, 1)
        self.assertEqual(mcc2.dtype, tf.float32)

    def initialize_vars(self, n_classes):
        mcc = MatthewsCorrelationCoefficient(num_classes=n_classes)
        self.evaluate(tf.compat.v1.variables_initializer(mcc.variables))
        return mcc

    def update_obj_states(self, obj, gt_label, preds):
        update_op = obj.update_state(gt_label, preds)
        self.evaluate(update_op)

    def check_results(self, obj, value):
        self.assertAllClose(value, self.evaluate(obj.result()), atol=1e-5)

    def test_binary_classes(self):
        gt_label = tf.constant([[1.0], [1.0], [1.0], [0.0]], dtype=tf.float32)
        preds = tf.constant([[1.0], [0.0], [1.0], [1.0]], dtype=tf.float32)
        # Initialize
        mcc = self.initialize_vars(n_classes=1)
        # Update
        self.update_obj_states(mcc, gt_label, preds)
        # Check results
        self.check_results(mcc, [-0.33333334])

    def test_multiple_classes(self):
        gt_label = tf.constant([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
                               dtype=tf.float32)
        preds = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0],
                             [1.0, 1.0, 0.0]],
                            dtype=tf.float32)
        # Initialize
        mcc = self.initialize_vars(n_classes=3)
        # Update
        self.update_obj_states(mcc, gt_label, preds)
        # Check results
        self.check_results(mcc, [-0.33333334, 1.0, 0.57735026])

    # Keras model API check
    def test_keras_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='softmax'))
        mcc = MatthewsCorrelationCoefficient(num_classes=1)
        model.compile(
            optimizer='Adam',
            loss='binary_crossentropy',
            metrics=['accuracy', mcc])
        # data preparation
        data = np.random.random((10, 1))
        labels = np.random.random((10, 1))
        labels = np.where(labels > 0.5, 1.0, 0.0)
        model.fit(data, labels, epochs=1, batch_size=32, verbose=0)


if __name__ == '__main__':
    tf.test.main()
