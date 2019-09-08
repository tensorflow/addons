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
"""Implements Cohen's Kappa."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
class CohenKappa(Metric):
    """Computes Kappa score between two raters.

    The score lies in the range [-1, 1]. A score of -1 represents
    complete disagreement between two raters whereas a score of 1
    represents complete agreement between the two raters.
    A score of 0 means agreement by chance.

    Note: As of now, this implementation considers all labels
    while calculating the Cohen's Kappa score.

    Usage:
    ```python
    actuals = np.array([4, 4, 3, 4, 2, 4, 1, 1], dtype=np.int32)
    preds = np.array([4, 4, 3, 4, 4, 2, 1, 1], dtype=np.int32)

    m = tf.keras.metrics.CohenKappa(num_classes=5)
    m.update_state(actuals, preds, "quadratic")
    print('Final result: ', m.result().numpy()) # Result: 0.68932
    ```
    Usage with tf.keras API:
    ```python
    model = keras.models.Model(inputs, outputs)
    model.add_metric(tf.keras.metrics.CohenKappa(num_classes=5)(outputs))
    model.compile('sgd', loss='mse')
    ```

    Args:
      num_classes : Number of unique classes in your dataset
      weightage   : Weighting to be considered for calculating
                    kappa statistics. A valid value is one of
                    [None, 'linear', 'quadratic']. Defaults to None.

    Returns:
      kappa_score : float
        The kappa statistic, which is a number between -1 and 1. The maximum
        value means complete agreement; zero or lower means chance agreement.

    Raises:
      ValueError: If the value passed for `weightage` is invalid
        i.e. not any one of [None, 'linear', 'quadratic']
    """

    def __init__(self,
                 num_classes,
                 name='cohen_kappa',
                 weightage=None,
                 dtype=tf.float32):
        super(CohenKappa, self).__init__(name=name, dtype=dtype)

        if weightage not in (None, 'linear', 'quadratic'):
            raise ValueError("Unknown kappa weighting type.")
        else:
            self.weightage = weightage

        self.num_classes = num_classes
        self.conf_mtx = self.add_weight(
            'conf_mtx',
            shape=(self.num_classes, self.num_classes),
            initializer=tf.keras.initializers.zeros,
            dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix condition statistics.

        Args:
          y_true : array, shape = [n_samples]
                   Labels assigned by the first annotator.
          y_pred : array, shape = [n_samples]
                   Labels assigned by the second annotator. The kappa statistic
                   is symmetric, so swapping ``y_true`` and ``y_pred`` doesn't
                   change the value.
          sample_weight(optional) : for weighting labels in confusion matrix
                   Default is None. The dtype for weights should be the same
                   as the dtype for confusion matrix. For more details,
                   please check tf.math.confusion_matrix.


        Returns:
          Update op.
        """
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_pred = tf.cast(y_pred, dtype=tf.int32)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                "Number of samples in y_true and y_pred are different")

        # compute the new values of the confusion matrix
        new_conf_mtx = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            weights=sample_weight)

        # update the values in the original confusion matrix
        return self.conf_mtx.assign_add(new_conf_mtx)

    def result(self):
        nb_ratings = tf.shape(self.conf_mtx)[0]
        weight_mtx = tf.ones([nb_ratings, nb_ratings], dtype=tf.int32)

        # 2. Create a weight matrix
        if self.weightage is None:
            diagonal = tf.zeros([nb_ratings], dtype=tf.int32)
            weight_mtx = tf.linalg.set_diag(weight_mtx, diagonal=diagonal)
            weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

        else:
            weight_mtx += tf.range(nb_ratings, dtype=tf.int32)
            weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

            if self.weightage == 'linear':
                weight_mtx = tf.abs(weight_mtx - tf.transpose(weight_mtx))
            else:
                weight_mtx = tf.pow((weight_mtx - tf.transpose(weight_mtx)), 2)
            weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

        # 3. Get counts
        actual_ratings_hist = tf.reduce_sum(self.conf_mtx, axis=1)
        pred_ratings_hist = tf.reduce_sum(self.conf_mtx, axis=0)

        # 4. Get the outer product
        out_prod = pred_ratings_hist[..., None] * \
                    actual_ratings_hist[None, ...]

        # 5. Normalize the confusion matrix and outer product
        conf_mtx = self.conf_mtx / tf.reduce_sum(self.conf_mtx)
        out_prod = out_prod / tf.reduce_sum(out_prod)

        conf_mtx = tf.cast(conf_mtx, dtype=tf.float32)
        out_prod = tf.cast(out_prod, dtype=tf.float32)

        # 6. Calculate Kappa score
        numerator = tf.reduce_sum(conf_mtx * weight_mtx)
        denominator = tf.reduce_sum(out_prod * weight_mtx)
        kp = 1 - (numerator / denominator)
        return kp

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "weightage": self.weightage,
        }
        base_config = super(CohenKappa, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        """Resets all of the metric state variables."""

        for v in self.variables:
            K.set_value(
                v, np.zeros((self.num_classes, self.num_classes), np.int32))