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

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.constraints.spectral_norm_constraint import (
    SpectralNorm,
    SpectralNormBuilder,
)


@pytest.fixture
def global_policy():
    tf.keras.mixed_precision.set_global_policy("float32")


def test_from_to_config(global_policy):
    layer = tf.keras.layers.Dense(1, kernel_constraint=SpectralNorm(1))
    config = layer.get_config()

    new_layer = tf.keras.layers.Dense.from_config(config)
    assert (
        layer.kernel_constraint.power_iterations
        == new_layer.kernel_constraint.power_iterations
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_compute_policy(global_policy):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    layer = tf.keras.layers.Dense(1, kernel_constraint=SpectralNorm(1))
    assert layer.kernel_constraint.compute_dtype == "float16"
    assert layer.kernel_constraint.dtype == "float32"

    layer = tf.keras.layers.Dense(
        1, kernel_constraint=SpectralNorm(1, dtype=tf.float32)
    )
    assert layer.kernel_constraint.compute_dtype == "float32"
    assert layer.kernel_constraint.dtype == "float32"


def test_wrong_output_channels(global_policy):
    inputs = tf.keras.layers.Input(shape=[10])
    sn_layer = tf.keras.layers.Dense(1, kernel_constraint=SpectralNorm(3))
    model = tf.keras.models.Sequential(layers=[inputs, sn_layer])

    optimizer = tf.keras.optimizers.Adam()
    model_input = np.random.uniform(size=(2, 10))
    with tf.GradientTape() as tape:
        loss = model(model_input)
    gradients = tape.gradient(loss, model.trainable_variables)
    with pytest.raises(ValueError):
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test_wrong_output_channels_eager(global_policy):
    tf.config.run_functions_eagerly(True)

    inputs = tf.keras.layers.Input(shape=[10])
    sn_layer = tf.keras.layers.Dense(1, kernel_constraint=SpectralNorm(3))
    model = tf.keras.models.Sequential(layers=[inputs, sn_layer])

    optimizer = tf.keras.optimizers.Adam()
    model_input = np.random.uniform(size=(2, 10))
    with tf.GradientTape() as tape:
        loss = model(model_input)
    gradients = tape.gradient(loss, model.trainable_variables)
    with pytest.raises(tf.errors.InvalidArgumentError):
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_normalization(global_policy):
    norm_instance = SpectralNorm(2)
    w = np.array([[2.0, 2.0], [2.0, 2.0]]).T

    # This wrapper normalizes weights by the maximum eigen value
    eigen_val, _ = np.linalg.eig(w)
    w_normed_target = w / np.max(eigen_val)

    w_normed_actual = tf.keras.backend.eval(norm_instance(tf.keras.backend.variable(w)))

    np.testing.assert_allclose(w_normed_actual, w_normed_target, rtol=1e-05)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_statefulness(global_policy):
    norm_instance = SpectralNorm(2)
    u = tf.constant([[0.03, 0.01]])
    norm_instance.u.assign(u)

    w = np.array([[1.0, 0.0], [0.0, 2.0]]).T

    # compute target
    lambda_target = np.sqrt(u[0, 0] ** 2 + 16.0 * u[0, 1] ** 2) / np.sqrt(
        u[0, 0] ** 2 + 4.0 * u[0, 1] ** 2
    )
    w_normed_target = 1.0 / lambda_target * np.array([[1.0, 0.0], [0.0, 2.0]])
    updated_u_target = np.array(
        [
            [
                u[0, 0] / np.sqrt(u[0, 0] ** 2 + 16 * u[0, 1] ** 2),
                4.0 * u[0, 1] / np.sqrt(u[0, 0] ** 2 + 16 * u[0, 1] ** 2),
            ]
        ]
    )

    w_normed_actual = tf.keras.backend.eval(norm_instance(tf.keras.backend.variable(w)))

    np.testing.assert_allclose(w_normed_actual, w_normed_target, rtol=1e-05)
    np.testing.assert_allclose(norm_instance.u, updated_u_target, rtol=1e-05)


def test_save_load_model(tmpdir, global_policy):
    # build model
    spectral_norm = SpectralNormBuilder()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=[1]))
    model.add(tf.keras.layers.Dense(1, kernel_constraint=spectral_norm.build(1)))
    model.add(tf.keras.layers.Dense(2, kernel_constraint=spectral_norm.build(2)))
    model.layers[0].kernel_constraint.u.assign(tf.constant([[0.01]]))

    # save
    checkpoint = tf.train.Checkpoint(
        model=model, spectral_norm_variables=spectral_norm.variables
    )
    save_path = checkpoint.save(str(tmpdir / "test.h5"))

    # build restored_model
    restored_spectral_norm = SpectralNormBuilder()
    restored_model = tf.keras.models.Sequential()
    restored_model.add(tf.keras.layers.Input(shape=[1]))
    restored_model.add(
        tf.keras.layers.Dense(1, kernel_constraint=restored_spectral_norm.build(1))
    )
    restored_model.add(
        tf.keras.layers.Dense(2, kernel_constraint=restored_spectral_norm.build(2))
    )
    restored_model.layers[0].kernel_constraint.u.assign(tf.constant([[0.03]]))

    assert (
        model.layers[0].kernel_constraint.u
        != restored_model.layers[0].kernel_constraint.u
    )

    # restore from checkpoint
    restored_checkpoint = tf.train.Checkpoint(
        model=restored_model, spectral_norm_variables=restored_spectral_norm.variables
    )
    restored_checkpoint.restore(save_path)

    assert (
        model.layers[0].kernel_constraint.u
        == restored_model.layers[0].kernel_constraint.u
    )
