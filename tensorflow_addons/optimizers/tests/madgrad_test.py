import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.optimizers import MadGrad


def run_dense_sample(iterations, expected, optimizer):
    var_0 = tf.Variable([1.0, 2.0], dtype=tf.dtypes.float32)
    var_1 = tf.Variable([3.0, 4.0], dtype=tf.dtypes.float32)

    grad_0 = tf.constant([0.1, 0.2], dtype=tf.dtypes.float32)
    grad_1 = tf.constant([0.3, 0.4], dtype=tf.dtypes.float32)

    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

    for _ in range(iterations):
        optimizer.apply_gradients(grads_and_vars)

    np.testing.assert_allclose(var_0.read_value(), expected[0], atol=2e-4)
    np.testing.assert_allclose(var_1.read_value(), expected[1], atol=2e-4)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample():
    run_dense_sample(
        iterations=1,
        expected=[[0.90999997, 1.8866072], [0.17019762, 0.257134]],
        optimizer=MadGrad(lr=0.1, epsilon=1e-8),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_dense_sample_with_weight_decay():
    run_dense_sample(
        iterations=1,
        expected=[[0.8866071, 1.8571339], [0.1364592, 0.22000009]],
        optimizer=MadGrad(lr=0.1, weight_decay=0.1, epsilon=1e-8),
    )


def run_sparse_sample(iterations, expected, optimizer):
    var_0 = tf.Variable([1.0, 2.0])
    var_1 = tf.Variable([3.0, 4.0])

    grad_0 = tf.IndexedSlices(
        tf.constant([0.1]), tf.constant([0]), tf.constant([2])
    )
    grad_1 = tf.IndexedSlices(
        tf.constant([0.4]), tf.constant([1]), tf.constant([2])
    )

    grads_and_vars = list(zip([grad_0, grad_1], [var_0, var_1]))

    for _ in range(iterations):
        optimizer.apply_gradients(grads_and_vars)

    np.testing.assert_allclose(var_0.read_value(), expected[0], atol=2e-4)
    np.testing.assert_allclose(var_1.read_value(), expected[1], atol=2e-4)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample():
    run_sparse_sample(
        iterations=2,
        expected=[[0.8290331, 2.0], [0.03000001, -0.23139302]],
        optimizer=MadGrad(lr=0.1, epsilon=1e-8),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_sample_with_weight_decay():
    run_sparse_sample(
        iterations=2,
        expected=[[0.78693587, 2.0], [0.03000001, -0.2593708]],
        optimizer=MadGrad(lr=0.1, weight_decay=0.1, epsilon=1e-8),
    )


def test_fit_simple_linear_model():
    np.random.seed(0x2020)
    tf.random.set_seed(0x2020)

    x = np.random.standard_normal((100000, 3))
    w = np.random.standard_normal((3, 1))
    y = np.dot(x, w) + np.random.standard_normal((100000, 1)) * 1e-5

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(input_shape=(3,), units=1))
    model.compile(MadGrad(), loss="mse")

    model.fit(x, y, epochs=2)

    x = np.random.standard_normal((100, 3))
    y = np.dot(x, w)
    predicted = model.predict(x)

    max_abs_diff = np.max(np.abs(predicted - y))
    assert max_abs_diff < 1e-2


def test_fit_sparse_linear_model():
    x = tf.keras.Input(shape=(4,), sparse=True)
    y = tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Zeros()
    )(x)
    model = tf.keras.Model(x, y)

    sparse_data = tf.SparseTensor(
        indices=[(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1)],
        values=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
        dense_shape=(6, 4),
    )
    labels = tf.constant([0, 0.01, 0.04, 0.09, 0.16, 0.25], tf.float32)

    model.compile(optimizer=MadGrad(1e-3), loss="mse")
    model.fit(sparse_data, labels, epochs=350, verbose=0)

    y = labels.numpy()
    predicted = model.predict(sparse_data).squeeze(-1)
    max_abs_diff = np.max(np.abs(predicted - y))
    assert max_abs_diff < 4e-2


def test_get_config():
    opt = MadGrad(lr=1e-2, momentum=0.5, weight_decay=0.1, epsilon=1e-6)
    config = opt.get_config()
    assert config["learning_rate"] == 1e-2
    assert config["momentum"] == 0.5
    assert config["epsilon"] == 1e-6
    assert config["weight_decay"] == 0.1


def test_serialization():
    optimizer = MadGrad(lr=1e-4, weight_decay=0.0)
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert new_optimizer.get_config() == optimizer.get_config()
