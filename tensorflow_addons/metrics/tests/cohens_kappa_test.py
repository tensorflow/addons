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
"""Tests for Cohen's Kappa Metric."""
import pytest
import numpy as np
import tensorflow as tf
from tensorflow_addons.metrics import CohenKappa
from tensorflow_addons.testing.serialization import check_metric_serialization


def test_config():
    kp_obj = CohenKappa(name="cohen_kappa", num_classes=5)
    assert kp_obj.name == "cohen_kappa"
    assert kp_obj.dtype == tf.float32
    assert kp_obj.num_classes == 5

    # Check save and restore config
    kb_obj2 = CohenKappa.from_config(kp_obj.get_config())
    assert kb_obj2.name == "cohen_kappa"
    assert kb_obj2.dtype == tf.float32
    assert kp_obj.num_classes == 5


def initialize_vars():
    kp_obj1 = CohenKappa(num_classes=5, sparse_labels=True)
    kp_obj2 = CohenKappa(num_classes=5, sparse_labels=True, weightage="linear")
    kp_obj3 = CohenKappa(num_classes=5, sparse_labels=True, weightage="quadratic")

    return kp_obj1, kp_obj2, kp_obj3


def update_obj_states(obj1, obj2, obj3, actuals, preds, weights):
    obj1.update_state(actuals, preds, sample_weight=weights)
    obj2.update_state(actuals, preds, sample_weight=weights)
    obj3.update_state(actuals, preds, sample_weight=weights)


def reset_obj_states(obj1, obj2, obj3):
    obj1.reset_state()
    obj2.reset_state()
    obj3.reset_state()


def check_results(objs, values):
    obj1, obj2, obj3 = objs
    val1, val2, val3 = values

    np.testing.assert_allclose(val1, obj1.result(), atol=1e-5)
    np.testing.assert_allclose(val2, obj2.result(), atol=1e-5)
    np.testing.assert_allclose(val3, obj3.result(), atol=1e-5)


def test_kappa_random_score():
    actuals = [4, 4, 3, 4, 2, 4, 1, 1]
    preds = [4, 4, 3, 4, 4, 2, 1, 1]
    actuals = tf.constant(actuals, dtype=tf.int32)
    preds = tf.constant(preds, dtype=tf.int32)

    # Initialize
    kp_obj1, kp_obj2, kp_obj3 = initialize_vars()

    # Update
    update_obj_states(kp_obj1, kp_obj2, kp_obj3, actuals, preds, None)

    # Check results
    check_results([kp_obj1, kp_obj2, kp_obj3], [0.61904761, 0.62790697, 0.68932038])


def test_kappa_perfect_score():
    actuals = [4, 4, 3, 3, 2, 2, 1, 1]
    preds = [4, 4, 3, 3, 2, 2, 1, 1]
    actuals = tf.constant(actuals, dtype=tf.int32)
    preds = tf.constant(preds, dtype=tf.int32)

    # Initialize
    kp_obj1, kp_obj2, kp_obj3 = initialize_vars()

    # Update
    update_obj_states(kp_obj1, kp_obj2, kp_obj3, actuals, preds, None)

    # Check results
    check_results([kp_obj1, kp_obj2, kp_obj3], [1.0, 1.0, 1.0])


def test_kappa_worse_than_random():
    actuals = [4, 4, 3, 3, 2, 2, 1, 1]
    preds = [1, 2, 4, 1, 3, 3, 4, 4]
    actuals = tf.constant(actuals, dtype=tf.int32)
    preds = tf.constant(preds, dtype=tf.int32)

    # Initialize
    kp_obj1, kp_obj2, kp_obj3 = initialize_vars()

    # Update
    update_obj_states(kp_obj1, kp_obj2, kp_obj3, actuals, preds, None)

    # check results
    check_results([kp_obj1, kp_obj2, kp_obj3], [-0.3333333, -0.52380952, -0.72727272])


def test_kappa_with_sample_weights():
    actuals = [4, 4, 3, 3, 2, 2, 1, 1]
    preds = [1, 2, 4, 1, 3, 3, 4, 4]
    weights = [1, 1, 2, 5, 10, 2, 3, 3]
    actuals = tf.constant(actuals, dtype=tf.int32)
    preds = tf.constant(preds, dtype=tf.int32)
    weights = tf.constant(weights, dtype=tf.int32)

    # Initialize
    kp_obj1, kp_obj2, kp_obj3 = initialize_vars()

    # Update
    update_obj_states(kp_obj1, kp_obj2, kp_obj3, actuals, preds, weights)

    # check results
    check_results([kp_obj1, kp_obj2, kp_obj3], [-0.25473321, -0.38992332, -0.60695344])


def test_kappa_reset_state():
    # Initialize
    kp_obj1, kp_obj2, kp_obj3 = initialize_vars()

    # reset states
    reset_obj_states(kp_obj1, kp_obj2, kp_obj3)

    # check results
    check_results([kp_obj1, kp_obj2, kp_obj3], [0.0, 0.0, 0.0])


def test_large_values():
    y_true = [1] * 10000 + [0] * 20000 + [1] * 20000
    y_pred = [0] * 20000 + [1] * 30000

    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)

    obj = CohenKappa(num_classes=2)

    obj.update_state(y_true, y_pred)
    np.testing.assert_allclose(0.166666666, obj.result(), 1e-6, 1e-6)


def test_with_sparse_labels():
    y_true = np.array([4, 4, 3, 4], dtype=np.int32)
    y_pred = np.array([4, 4, 1, 2], dtype=np.int32)

    obj = CohenKappa(num_classes=5, sparse_labels=True)

    obj.update_state(y_true, y_pred)
    np.testing.assert_allclose(0.19999999, obj.result())


def test_keras_binary_reg_model():
    kp = CohenKappa(num_classes=2)
    inputs = tf.keras.layers.Input(shape=(10,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer="sgd", loss="mse", metrics=[kp])

    x = np.random.rand(1000, 10).astype(np.float32)
    y = np.random.randint(2, size=(1000, 1)).astype(np.float32)

    model.fit(x, y, epochs=1, verbose=0, batch_size=32)


def test_keras_multiclass_reg_model():
    kp = CohenKappa(num_classes=5, regression=True, sparse_labels=True)
    inputs = tf.keras.layers.Input(shape=(10,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer="sgd", loss="mse", metrics=[kp])

    x = np.random.rand(1000, 10).astype(np.float32)
    y = np.random.randint(5, size=(1000,)).astype(np.float32)

    model.fit(x, y, epochs=1, verbose=0, batch_size=32)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_keras_binary_classification_model():
    kp = CohenKappa(num_classes=2)
    inputs = tf.keras.layers.Input(shape=(10,))
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(inputs)
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=[kp])

    x = np.random.rand(1000, 10).astype(np.float32)
    y = np.random.randint(2, size=(1000, 1)).astype(np.float32)

    model.fit(x, y, epochs=1, verbose=0, batch_size=32)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_keras_multiclass_classification_model():
    kp = CohenKappa(num_classes=5)
    inputs = tf.keras.layers.Input(shape=(10,))
    outputs = tf.keras.layers.Dense(5, activation="softmax")(inputs)
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=[kp])

    x = np.random.rand(1000, 10).astype(np.float32)
    y = np.random.randint(5, size=(1000,)).astype(np.float32)
    y = tf.keras.utils.to_categorical(y, num_classes=5)

    model.fit(x, y, epochs=1, verbose=0, batch_size=32)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_with_ohe_labels():
    y_true = np.array([4, 4, 3, 4], dtype=np.int32)
    y_true = tf.keras.utils.to_categorical(y_true, num_classes=5)
    y_pred = np.array([4, 4, 1, 2], dtype=np.int32)

    obj = CohenKappa(num_classes=5, sparse_labels=False)

    obj.update_state(y_true, y_pred)
    np.testing.assert_allclose(0.19999999, obj.result().numpy())


def test_cohen_kappa_serialization():
    actuals = np.array([4, 4, 3, 3, 2, 2, 1, 1], dtype=np.int32)
    preds = np.array([1, 2, 4, 1, 3, 3, 4, 4], dtype=np.int32)
    weights = np.array([1, 1, 2, 5, 10, 2, 3, 3], dtype=np.int32)

    ck = CohenKappa(num_classes=5, sparse_labels=True, weightage="quadratic")
    check_metric_serialization(ck, actuals, preds, weights)


def test_cohen_kappa_single_batch():
    # Test for issue #1962
    obj = CohenKappa(5, regression=True, sparse_labels=True)

    # Test single batch update
    obj.update_state(tf.ones(1), tf.zeros(1))

    np.testing.assert_allclose(0, obj.result().numpy())
