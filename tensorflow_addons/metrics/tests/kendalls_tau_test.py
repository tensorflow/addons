# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Kendall's Tau-b Metric."""
import pytest
import numpy as np
import tensorflow as tf
from scipy import stats
from tensorflow_addons.metrics import KendallsTau
from tensorflow_addons.testing.serialization import check_metric_serialization


def test_config():
    kp_obj = KendallsTau(name="kendalls_tau")
    assert kp_obj.name == "kendalls_tau"
    assert kp_obj.dtype == tf.float32
    assert kp_obj.actual_min == 0.0
    assert kp_obj.actual_max == 1.0

    # Check save and restore config
    kp_obj2 = KendallsTau.from_config(kp_obj.get_config())
    assert kp_obj2.name == "kendalls_tau"
    assert kp_obj2.dtype == tf.float32
    assert kp_obj2.actual_min == 0.0
    assert kp_obj2.actual_max == 1.0


def test_scoring_with_ties():
    actuals = [12, 2, 1, 12, 2]
    preds = [1, 4, 7, 1, 0]
    actuals = tf.constant(actuals, dtype=tf.int32)
    preds = tf.constant(preds, dtype=tf.int32)

    metric = KendallsTau(0, 13, 0, 8)
    metric.update_state(actuals, preds)
    np.testing.assert_almost_equal(metric.result(), stats.kendalltau(actuals, preds)[0])


def test_perfect():
    actuals = [1, 2, 3, 4, 5, 6, 7, 8]
    preds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    actuals = tf.constant(actuals, dtype=tf.int32)
    preds = tf.constant(preds, dtype=tf.float32)

    metric = KendallsTau(0, 10, 0.0, 1.0)
    metric.update_state(actuals, preds)
    np.testing.assert_almost_equal(metric.result(), 1.0)


def test_reversed():
    actuals = [1, 2, 3, 4, 5, 6, 7, 8]
    preds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8][::-1]
    actuals = tf.constant(actuals, dtype=tf.int32)
    preds = tf.constant(preds, dtype=tf.float32)

    metric = KendallsTau(0, 10, 0.0, 1.0)
    metric.update_state(actuals, preds)
    np.testing.assert_almost_equal(metric.result(), -1.0)


def test_scoring_iterative():
    actuals = [12, 2, 1, 12, 2]
    preds = [1, 4, 7, 1, 0]

    metric = KendallsTau(0, 13, 0, 8)
    for actual, pred in zip(actuals, preds):
        metric.update_state(tf.constant([[actual]]), tf.constant([[pred]]))
    np.testing.assert_almost_equal(metric.result(), stats.kendalltau(actuals, preds)[0])


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_keras_binary_classification_model():
    kp = KendallsTau()
    inputs = tf.keras.layers.Input(shape=(10,))
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(inputs)
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=[kp])

    x = np.random.rand(1000, 10).astype(np.float32)
    y = np.random.rand(1000, 1).astype(np.float32)

    model.fit(x, y, epochs=1, verbose=0, batch_size=32)


def test_kendalls_tau_serialization():
    actuals = np.array([4, 4, 3, 3, 2, 2, 1, 1], dtype=np.int32)
    preds = np.array([1, 2, 4, 1, 3, 3, 4, 4], dtype=np.int32)

    kt = KendallsTau(0, 5, 0, 5, 10, 10)
    check_metric_serialization(kt, actuals, preds)
