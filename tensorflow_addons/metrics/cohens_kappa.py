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
"""Implements Cohen's Kappa"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.math import confusion_matrix
from tensorflow.keras.metrics import Metric
from tensorflow_addons.utils import keras_utils


@keras_utils.register_keras_custom_object
class CohensKappa(Metric):
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

  m = tf.keras.metrics.CohensKappa()
  m.update_state(actuals, preds, "quadratic")
  print('Final result: ', m.result().numpy()) # Result: 0.68932
  ```
  Usage with tf.keras API:
  ```python
  model = keras.models.Model(inputs, outputs)
  model.add_metric(tf.keras.metrics.CohensKappa(name='kp_score')(outputs))
  model.compile('sgd', loss='mse')
  ```

  Args:
    y1 : array, shape = [n_samples]
      Labels assigned by the first annotator.
    y2 : array, shape = [n_samples]
      Labels assigned by the second annotator. The kappa statistic is
      symmetric, so swapping ``y1`` and ``y2`` doesn't change the value.
    sample_weight(optional) : None or str 
      A string denoting the type of weighting to be used.
      Valid values for this parameter are [None, 'linear', 'quadratic'].
      Default value is None.

  Returns:
    kappa_score : float
      The kappa statistic, which is a number between -1 and 1. The maximum
      value means complete agreement; zero or lower means chance agreement.

  Raises:
    ValueError: If the value passed for `sample_weight` is invalid
      i.e. not any one of [None, 'linear', 'quadratic']

  """
  def __init__(self, name='cohens_kappa', dtype=tf.float32):
    super(CohensKappa, self).__init__(name=name, dtype=dtype)
    self.kappa_score = self.add_weight('kappa_score', 
                                       initializer=None)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_pred = tf.cast(y_pred, dtype=tf.int32)

    # check if weighting type is valid
    if sample_weight not in (None, 'linear', 'quadratic'):
      raise ValueError("Unknown kappa weighting type.")

    # 1. Get the confusion matrix
    conf_mtx = confusion_matrix(labels=y_true, predictions=y_pred)     
    nb_ratings = tf.shape(conf_mtx)[0]
    weight_mtx = tf.ones([nb_ratings, nb_ratings], dtype=tf.int32)
    
    # 2. Create a weight matrix
    if sample_weight is None:
      diagonal = tf.zeros([5], dtype=tf.int32)
      weight_mtx = tf.linalg.set_diag(weight_mtx, diagonal=diagonal)
      weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)
    
    else:
      weight_mtx += tf.range(nb_ratings, dtype=tf.int32)
      weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

      if sample_weight=='linear':
        weight_mtx = tf.abs(weight_mtx - K.transpose(weight_mtx))
      else:
        weight_mtx = K.pow((weight_mtx - K.transpose(weight_mtx)), 2)
      weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)
    
    # 3. Get counts
    actual_ratings_hist = K.sum(conf_mtx, axis=1)
    pred_ratings_hist = K.sum(conf_mtx, axis=0)
    
    # 4. Get the outer product
    out_prod = pred_ratings_hist[..., None] * actual_ratings_hist[None, ...]
    
    # 5. Normalize the confusion matrix and outer product
    conf_mtx = conf_mtx / K.sum(conf_mtx)
    out_prod = out_prod / K.sum(out_prod)
    
    conf_mtx = tf.cast(conf_mtx, dtype=tf.float32)
    out_prod = tf.cast(out_prod, dtype=tf.float32)
    
    # 6. Calculate Kappa score
    numerator = K.sum(conf_mtx * weight_mtx)
    denominator = K.sum(out_prod * weight_mtx)
    kp = 1-(numerator/denominator)
    
    return self.kappa_score.assign(kp)

  def result(self):
    return self.kappa_score