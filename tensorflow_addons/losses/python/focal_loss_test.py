## Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for focal loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_addons.utils.python import keras_utils
from tensorflow_addons.utils.python import test_utils
from tensorflow_addons.losses.python import focal_loss


@test_utils.run_all_in_graph_and_eager_modes
class SigmoidFocalCrossEntropyTest(tf.test.TestCase):
    def test_config(self):
        bce_obj = keras.losses.SigmoidFocalCrossEntropy(
            reduction=tf.keras.losses.Reduction.NONE,
            name='sigmoid_focal_crossentropy')
        self.assertEqual(bce_obj.name, 'sigmoid_focal_crossentropy')
        self.assertEqual(bce_obj.reduction, tf.keras.losses.Reduction.NONE)

    def to_logit(self, prob):
        logit = np.log(prob / (1. - prob))
        return logit

    def log10(self, x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    ## Test with logits
    def test_with_logits(self):
        # predictiions represented as logits
        prediction_tensor = tf.constant(
            [[self.to_logit(0.97)], [self.to_logit(0.91)],
             [self.to_logit(0.73)], [self.to_logit(0.27)],
             [self.to_logit(0.09)], [self.to_logit(0.03)]], tf.float32)
        # Ground truth
        target_tensor = tf.constant([[1], [1], [1], [0], [0], [0]], tf.float32)

        FL = sigmoid_focal_crossentropy(
            y_true=target_tensor,
            y_pred=prediction_tensor,
            from_logits=True,
            alpha=None,
            gamma=None)
        bce = K.binary_crossentropy(
            target_tensor, prediction_tensor, from_logits=True)

        # When alpha and gamma are None, it should be equal to BCE
        self.assertAllClose(FL, bce)

        # When gamma==2.0
        FL = sigmoid_focal_crossentropy(
            y_true=target_tensor,
            y_pred=prediction_tensor,
            from_logits=True,
            alpha=None,
            gamma=2.0)

        #order_of_ratio = np.power(10, np.floor(np.log10(bce/FL)))
        order_of_ratio = tf.pow(10.0, tf.math.floor(self.log10(bce / FL)))
        pow_values = tf.constant([[1000], [100], [10], [10], [100], [1000]])
        self.assertAllClose(order_of_ratio, pow_values)

    ## Test without logits
    def test_without_logits(self):
        # predictiions represented as logits
        prediction_tensor = tf.constant(
            [[0.97], [0.91], [0.73], [0.27], [0.09], [0.03]], tf.float32)
        # Ground truth
        target_tensor = tf.constant([[1], [1], [1], [0], [0], [0]], tf.float32)

        FL = sigmoid_focal_crossentropy(
            y_true=target_tensor,
            y_pred=prediction_tensor,
            alpha=None,
            gamma=None)
        bce = K.binary_crossentropy(target_tensor, prediction_tensor)

        # When alpha and gamma are None, it should be equal to BCE
        self.assertAllClose(FL, bce)

        # When gamma==2.0
        FL = sigmoid_focal_crossentropy(
            y_true=target_tensor,
            y_pred=prediction_tensor,
            alpha=None,
            gamma=2.0)

        order_of_ratio = tf.pow(10.0, tf.math.floor(self.log10(bce / FL)))
        pow_values = tf.constant([[1000], [100], [10], [10], [100], [1000]])
        self.assertAllClose(order_of_ratio, pow_values)


if __name__ == '__main__':
    tf.test.main()
