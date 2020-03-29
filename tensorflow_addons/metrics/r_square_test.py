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
"""Tests for R-Square Metric."""

import sys

import numpy as np
import pytest
import tensorflow as tf
from sklearn.metrics import r2_score as sklearn_r2_score
from tensorflow_addons.metrics import RSquare
from tensorflow_addons.metrics.r_square import VALID_MULTIOUTPUT


def test_config():
    r2_obj = RSquare(name="r_square")
    assert r2_obj.name == "r_square"
    assert r2_obj.dtype == tf.float32
    # Check save and restore config
    r2_obj2 = RSquare.from_config(r2_obj.get_config())
    assert r2_obj2.name == "r_square"
    assert r2_obj2.dtype == tf.float32


def initialize_vars(y_shape=(), multioutput: str = "uniform_average"):
    return RSquare(y_shape=y_shape, multioutput=multioutput)


def update_obj_states(obj, actuals, preds, sample_weight=None):
    obj.update_state(actuals, preds, sample_weight=sample_weight)


def check_results(obj, value):
    np.testing.assert_allclose(value, obj.result(), atol=1e-5)


def test_r2_perfect_score():
    actuals = tf.constant([100, 700, 40, 5.7], dtype=tf.float32)
    preds = tf.constant([100, 700, 40, 5.7], dtype=tf.float32)
    actuals = tf.cast(actuals, dtype=tf.float32)
    preds = tf.cast(preds, dtype=tf.float32)
    # Initialize
    r2_obj = initialize_vars()
    # Update
    update_obj_states(r2_obj, actuals, preds)
    # Check results
    check_results(r2_obj, 1.0)


def test_r2_worst_score():
    actuals = tf.constant([10, 600, 4, 9.77], dtype=tf.float32)
    preds = tf.constant([1, 70, 40, 5.7], dtype=tf.float32)
    actuals = tf.cast(actuals, dtype=tf.float32)
    preds = tf.cast(preds, dtype=tf.float32)
    # Initialize
    r2_obj = initialize_vars()
    # Update
    update_obj_states(r2_obj, actuals, preds)
    # Check results
    check_results(r2_obj, -0.073607)


def test_r2_random_score():
    actuals = tf.constant([10, 600, 3, 9.77], dtype=tf.float32)
    preds = tf.constant([1, 340, 40, 5.7], dtype=tf.float32)
    actuals = tf.cast(actuals, dtype=tf.float32)
    preds = tf.cast(preds, dtype=tf.float32)
    # Initialize
    r2_obj = initialize_vars()
    # Update
    update_obj_states(r2_obj, actuals, preds)
    # Check results
    check_results(r2_obj, 0.7376327)


def test_r2_sklearn_comparison():
    """Test that RSquare behaves similarly to the scikit-learn
    implementation of the same metric, given random input.
    """
    for multioutput in VALID_MULTIOUTPUT:
        for i in range(10):
            actuals = np.random.rand(64, 3)
            preds = np.random.rand(64, 3)
            sample_weight = np.random.rand(64, 1)
            tensor_actuals = tf.constant(actuals, dtype=tf.float32)
            tensor_preds = tf.constant(preds, dtype=tf.float32)
            tensor_sample_weight = tf.constant(sample_weight, dtype=tf.float32)
            tensor_actuals = tf.cast(tensor_actuals, dtype=tf.float32)
            tensor_preds = tf.cast(tensor_preds, dtype=tf.float32)
            tensor_sample_weight = tf.cast(tensor_sample_weight, dtype=tf.float32)
            # Initialize
            r2_obj = initialize_vars(y_shape=(3,), multioutput=multioutput)
            # Update
            update_obj_states(
                r2_obj,
                tensor_actuals,
                tensor_preds,
                sample_weight=tensor_sample_weight,
            )
            # Check results by comparing to results of scikit-learn r2 implementation
            sklearn_result = sklearn_r2_score(
                actuals, preds, sample_weight=sample_weight, multioutput=multioutput
            )
            check_results(r2_obj, sklearn_result)


def test_unrecognized_multioutput():
    with pytest.raises(ValueError):
        initialize_vars(multioutput="meadian")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
