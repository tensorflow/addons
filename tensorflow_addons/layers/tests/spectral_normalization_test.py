# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.layers import spectral_normalization
from tensorflow_addons.utils import test_utils


def test_keras():
    input_data = np.random.random((10, 3, 4)).astype(np.float32)
    test_utils.layer_test(
        spectral_normalization.SpectralNormalization,
        kwargs={"layer": tf.keras.layers.Dense(2), "input_shape": (3, 4)},
        input_data=input_data,
    )


def test_from_to_config():
    base_layer = tf.keras.layers.Dense(1)
    sn = spectral_normalization.SpectralNormalization(base_layer)
    config = sn.get_config()

    new_sn = spectral_normalization.SpectralNormalization.from_config(config)
    assert sn.power_iterations == new_sn.power_iterations


def test_save_load_model(tmpdir):
    base_layer = tf.keras.layers.Dense(1)
    input_shape = [1]

    inputs = tf.keras.layers.Input(shape=input_shape)
    sn_layer = spectral_normalization.SpectralNormalization(base_layer)
    model = tf.keras.models.Sequential(layers=[inputs, sn_layer])

    # initialize model
    model.predict(np.random.uniform(size=(2, 1)))

    model_path = str(tmpdir / "test.h5")
    model.save(model_path)
    new_model = tf.keras.models.load_model(model_path)

    assert model.layers[0].get_config() == new_model.layers[0].get_config()


@pytest.mark.parametrize(
    "base_layer_fn, input_shape, output_shape",
    [
        (lambda: tf.keras.layers.Dense(2), [3, 2], [3, 2]),
        (
            lambda: tf.keras.layers.Conv2D(3, (2, 2), padding="same"),
            [4, 4, 3],
            [4, 4, 3],
        ),
        (lambda: tf.keras.layers.Embedding(2, 10), [2], [2, 10]),
    ],
)
def test_model_fit(base_layer_fn, input_shape, output_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    base_layer = base_layer_fn()

    sn_layer = spectral_normalization.SpectralNormalization(base_layer)
    model = tf.keras.models.Sequential(layers=[inputs, sn_layer])
    model.add(tf.keras.layers.Activation("relu"))

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss="mse"
    )
    model.fit(
        np.random.random((2, *input_shape)),
        np.random.random((2, *output_shape)),
        epochs=3,
        batch_size=10,
        verbose=0,
    )
    assert hasattr(model.layers[0], "u")


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize(
    "base_layer_fn, input_shape",
    [
        (lambda: tf.keras.layers.Dense(2), [3, 2]),
        (lambda: tf.keras.layers.Conv2D(3, (2, 2), padding="same"), [4, 4, 3]),
        (lambda: tf.keras.layers.Embedding(2, 10), [2]),
    ],
)
def test_model_build(base_layer_fn, input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    base_layer = base_layer_fn()
    sn_layer = spectral_normalization.SpectralNormalization(base_layer)
    model = tf.keras.models.Sequential(layers=[inputs, sn_layer])
    model.build()
    assert hasattr(model.layers[0], "u")


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.usefixtures("run_with_mixed_precision_policy")
def test_normalization():
    inputs = tf.keras.layers.Input(shape=[2, 2, 1])

    base_layer = tf.keras.layers.Conv2D(
        1, (2, 2), kernel_initializer=tf.constant_initializer(value=2)
    )
    sn_layer = spectral_normalization.SpectralNormalization(base_layer)
    model = tf.keras.models.Sequential(layers=[inputs, sn_layer])

    weights = np.squeeze(model.layers[0].w.numpy())
    # This wrapper normalizes weights by the maximum eigen value
    eigen_val, _ = np.linalg.eig(weights)
    weights_normalized = weights / np.max(eigen_val)

    for training in [False, True]:
        _ = model(
            tf.constant(np.ones((1, 2, 2, 1), dtype=np.float32)), training=training
        )
        if training:
            w = weights_normalized
        else:
            w = weights
        np.testing.assert_allclose(w, np.squeeze(model.layers[0].w.numpy()))


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_apply_layer():
    images = tf.ones((1, 2, 2, 1))
    sn_wrapper = spectral_normalization.SpectralNormalization(
        tf.keras.layers.Conv2D(
            1, [2, 2], kernel_initializer=tf.constant_initializer(value=1)
        ),
        input_shape=(2, 2, 1),
    )

    result = sn_wrapper(images, training=False)
    result_train = sn_wrapper(images, training=True)
    expected_output = np.array([[[[4.0]]]], dtype=np.float32)

    np.testing.assert_allclose(result, expected_output)
    # max eigen value of 2x2 matrix of ones is 2
    np.testing.assert_allclose(result_train, expected_output / 2)
    assert hasattr(sn_wrapper, "u")


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_no_layer():
    images = tf.random.uniform((2, 4, 43))
    with pytest.raises(AssertionError):
        spectral_normalization.SpectralNormalization(images)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_no_kernel():
    with pytest.raises(AttributeError):
        spectral_normalization.SpectralNormalization(
            tf.keras.layers.MaxPooling2D(2, 2)
        ).build((2, 2))
