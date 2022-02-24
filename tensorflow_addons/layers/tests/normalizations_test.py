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
# =============================================================================


import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.normalizations import FilterResponseNormalization
from tensorflow_addons.layers.normalizations import GroupNormalization
from tensorflow_addons.layers.normalizations import InstanceNormalization


# ------------Tests to ensure proper inheritance. If these suceed you can
# test for Instance norm by setting Groupnorm groups = -1
def test_inheritance():
    assert issubclass(InstanceNormalization, GroupNormalization)
    assert InstanceNormalization.build == GroupNormalization.build
    assert InstanceNormalization.call == GroupNormalization.call


def test_groups_after_init():
    layers = InstanceNormalization()
    assert layers.groups == -1


def test_weights():
    # Check if weights get initialized correctly
    layer = GroupNormalization(groups=1, scale=False, center=False)
    layer.build((None, 3, 4))
    assert len(layer.trainable_weights) == 0
    assert len(layer.weights) == 0

    layer = InstanceNormalization()
    layer.build((None, 3, 4))
    assert len(layer.trainable_weights) == 2
    assert len(layer.weights) == 2


def test_apply_normalization():
    input_shape = (1, 4)
    reshaped_inputs = tf.constant([[[2.0, 2.0], [3.0, 3.0]]])
    layer = GroupNormalization(groups=2, axis=1, scale=False, center=False)
    normalized_input = layer._apply_normalization(reshaped_inputs, input_shape)
    np.testing.assert_equal(normalized_input, np.array([[[0.0, 0.0], [0.0, 0.0]]]))


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_reshape():
    def run_reshape_test(axis, group, input_shape, expected_shape):
        group_layer = GroupNormalization(groups=group, axis=axis)
        group_layer._set_number_of_groups_for_instance_norm(input_shape)

        inputs = np.ones(input_shape)
        tensor_input_shape = tf.convert_to_tensor(input_shape)
        reshaped_inputs, group_shape = group_layer._reshape_into_groups(
            inputs, (10, 10, 10), tensor_input_shape
        )
        for i in range(len(expected_shape)):
            assert group_shape[i] == expected_shape[i]

    input_shape = (10, 10, 10)
    expected_shape = [10, 10, 5, 2]
    run_reshape_test(2, 5, input_shape, expected_shape)

    input_shape = (10, 10, 10)
    expected_shape = [10, 2, 5, 10]
    run_reshape_test(1, 2, input_shape, expected_shape)

    input_shape = (10, 10, 10)
    expected_shape = [10, 10, 10]
    run_reshape_test(1, -1, input_shape, expected_shape)

    input_shape = (10, 10, 10)
    expected_shape = [10, 1, 10, 10]
    run_reshape_test(1, 1, input_shape, expected_shape)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("scale", [True, False])
def test_feature_input(center, scale):
    shape = (10, 100)
    for groups in [-1, 1, 2, 5]:
        _test_random_shape_on_all_axis_except_batch(shape, groups, center, scale)


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("scale", [True, False])
def test_picture_input(center, scale):
    shape = (10, 30, 30, 3)
    for groups in [-1, 1, 3]:
        _test_random_shape_on_all_axis_except_batch(shape, groups, center, scale)


def _test_random_shape_on_all_axis_except_batch(shape, groups, center, scale):
    inputs = tf.random.normal(shape)
    for axis in range(1, len(shape)):
        _test_specific_layer(inputs, axis, groups, center, scale)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def _test_specific_layer(inputs, axis, groups, center, scale):

    input_shape = inputs.shape

    # Get Output from Keras model
    layer = GroupNormalization(axis=axis, groups=groups, center=center, scale=scale)
    model = tf.keras.models.Sequential()
    model.add(layer)
    outputs = model.predict(inputs, steps=1)
    assert not np.isnan(outputs).any()

    is_instance_norm = False
    # Create shapes
    if groups == -1:
        groups = input_shape[axis]
    if (input_shape[axis] // groups) == 1:
        is_instance_norm = True
    np_inputs = inputs
    reshaped_dims = list(np_inputs.shape)
    if not is_instance_norm:
        reshaped_dims[axis] = reshaped_dims[axis] // groups
        reshaped_dims.insert(axis, groups)
        reshaped_inputs = np.reshape(np_inputs, tuple(reshaped_dims))
    else:
        reshaped_inputs = np_inputs

    group_reduction_axes = list(range(1, len(reshaped_dims)))
    if not is_instance_norm:
        axis = -2 if axis == -1 else axis - 1
    else:
        axis = -1 if axis == -1 else axis - 1
    group_reduction_axes.pop(axis)

    # Calculate mean and variance
    mean = np.mean(reshaped_inputs, axis=tuple(group_reduction_axes), keepdims=True)
    variance = np.var(reshaped_inputs, axis=tuple(group_reduction_axes), keepdims=True)

    # Get gamma and beta initalized by layer
    gamma, beta = layer._get_reshaped_weights(input_shape)
    if gamma is None:
        gamma = 1.0
    if beta is None:
        beta = 0.0

    # Get ouput from Numpy
    zeroed = reshaped_inputs - mean
    rsqrt = 1 / np.sqrt(variance + 1e-5)
    output_test = gamma * zeroed * rsqrt + beta

    # compare outputs
    output_test = tf.reshape(output_test, input_shape)
    np.testing.assert_almost_equal(tf.reduce_mean(output_test - outputs), 0, decimal=7)


def _create_and_fit_sequential_model(layer, shape):
    # Helperfunction for quick evaluation
    np.random.seed(0x2020)
    model = tf.keras.models.Sequential()
    model.add(layer)
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(0.01), loss="categorical_crossentropy"
    )
    layer_shape = (10,) + shape
    input_batch = np.random.rand(*layer_shape)
    output_batch = np.random.rand(*(10, 1))
    model.fit(x=input_batch, y=output_batch, epochs=1, batch_size=1)
    return model


def test_groupnorm_flat():
    # Check basic usage of groupnorm_flat
    # Testing for 1 == LayerNorm, 16 == GroupNorm, -1 == InstanceNorm

    groups = [-1, 16, 1]
    shape = (64,)
    for i in groups:
        model = _create_and_fit_sequential_model(GroupNormalization(groups=i), shape)
        assert hasattr(model.layers[0], "gamma")
        assert hasattr(model.layers[0], "beta")


def test_instancenorm_flat():
    # Check basic usage of instancenorm
    model = _create_and_fit_sequential_model(InstanceNormalization(), (64,))
    assert hasattr(model.layers[0], "gamma")
    assert hasattr(model.layers[0], "beta")


def test_initializer():
    # Check if the initializer for gamma and beta is working correctly
    layer = GroupNormalization(
        groups=32,
        beta_initializer="random_normal",
        beta_constraint="NonNeg",
        gamma_initializer="random_normal",
        gamma_constraint="NonNeg",
    )

    model = _create_and_fit_sequential_model(layer, (64,))

    weights = np.array(model.layers[0].get_weights())
    negativ = weights[weights < 0.0]
    assert len(negativ) == 0


def test_axis_error():
    with pytest.raises(ValueError):
        GroupNormalization(axis=0)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_groupnorm_conv():
    # Check if Axis is working for CONV nets
    # Testing for 1 == LayerNorm, 5 == GroupNorm, -1 == InstanceNorm
    np.random.seed(0x2020)
    groups = [-1, 5, 1]
    for i in groups:
        model = tf.keras.models.Sequential()
        model.add(GroupNormalization(axis=1, groups=i, input_shape=(20, 20, 3)))
        model.add(tf.keras.layers.Conv2D(5, (1, 1), padding="same"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, activation="softmax"))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(0.01), loss="mse")
        x = np.random.randint(1000, size=(10, 20, 20, 3))
        y = np.random.randint(1000, size=(10, 1))
        model.fit(x=x, y=y, epochs=1)
        assert hasattr(model.layers[0], "gamma")


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_regularizations():
    layer = GroupNormalization(
        gamma_regularizer="l1", beta_regularizer="l1", groups=4, axis=2
    )
    layer.build((None, 4, 4))
    assert len(layer.losses) == 2
    max_norm = tf.keras.constraints.max_norm
    layer = GroupNormalization(
        groups=2, gamma_constraint=max_norm, beta_constraint=max_norm
    )
    layer.build((None, 3, 4))
    assert layer.gamma.constraint == max_norm
    assert layer.beta.constraint == max_norm


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_groupnorm_correctness_1d():
    np.random.seed(0x2020)
    model = tf.keras.models.Sequential()
    norm = GroupNormalization(input_shape=(10,), groups=2)
    model.add(norm)
    model.compile(loss="mse", optimizer="rmsprop")

    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10))
    model.fit(x, x, epochs=5, verbose=0)
    out = model.predict(x)
    out -= norm.beta.numpy()
    out /= norm.gamma.numpy()

    np.testing.assert_allclose(out.mean(), 0.0, atol=1e-1)
    np.testing.assert_allclose(out.std(), 1.0, atol=1e-1)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_groupnorm_2d_different_groups():
    np.random.seed(0x2020)
    groups = [2, 1, 10]
    for i in groups:
        model = tf.keras.models.Sequential()
        norm = GroupNormalization(axis=1, groups=i, input_shape=(10, 3))
        model.add(norm)
        # centered and variance are 5.0 and 10.0, respectively
        model.compile(loss="mse", optimizer="rmsprop")
        x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10, 3))
        model.fit(x, x, epochs=5, verbose=0)
        out = model.predict(x)
        out -= np.reshape(norm.beta.numpy(), (1, 10, 1))
        out /= np.reshape(norm.gamma.numpy(), (1, 10, 1))

        np.testing.assert_allclose(
            out.mean(axis=(0, 1), dtype=np.float32), (0.0, 0.0, 0.0), atol=1e-1
        )
        np.testing.assert_allclose(
            out.std(axis=(0, 1), dtype=np.float32), (1.0, 1.0, 1.0), atol=1e-1
        )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_groupnorm_convnet():
    np.random.seed(0x2020)
    model = tf.keras.models.Sequential()
    norm = GroupNormalization(axis=1, input_shape=(3, 4, 4), groups=3)
    model.add(norm)
    model.compile(loss="mse", optimizer="sgd")

    # centered = 5.0, variance  = 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3, 4, 4))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)
    out -= np.reshape(norm.beta.numpy(), (1, 3, 1, 1))
    out /= np.reshape(norm.gamma.numpy(), (1, 3, 1, 1))

    np.testing.assert_allclose(
        np.mean(out, axis=(0, 2, 3), dtype=np.float32), (0.0, 0.0, 0.0), atol=1e-1
    )
    np.testing.assert_allclose(
        np.std(out, axis=(0, 2, 3), dtype=np.float32), (1.0, 1.0, 1.0), atol=1e-1
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_groupnorm_convnet_no_center_no_scale():
    np.random.seed(0x2020)
    model = tf.keras.models.Sequential()
    norm = GroupNormalization(
        axis=-1, groups=2, center=False, scale=False, input_shape=(3, 4, 4)
    )
    model.add(norm)
    model.compile(loss="mse", optimizer="sgd")
    # centered and variance are  5.0 and 10.0, respectively
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3, 4, 4))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)

    np.testing.assert_allclose(
        np.mean(out, axis=(0, 2, 3), dtype=np.float32), (0.0, 0.0, 0.0), atol=1e-1
    )
    np.testing.assert_allclose(
        np.std(out, axis=(0, 2, 3), dtype=np.float32), (1.0, 1.0, 1.0), atol=1e-1
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("scale", [True, False])
def test_group_norm_compute_output_shape(center, scale):

    target_variables_len = [center, scale].count(True)
    target_trainable_variables_len = [center, scale].count(True)
    layer1 = GroupNormalization(groups=2, center=center, scale=scale)
    layer1.build(input_shape=[8, 28, 28, 16])  # build()
    assert len(layer1.variables) == target_variables_len
    assert len(layer1.trainable_variables) == target_trainable_variables_len

    layer2 = GroupNormalization(groups=2, center=center, scale=scale)
    layer2.compute_output_shape(input_shape=[8, 28, 28, 16])  # compute_output_shape()
    assert len(layer2.variables) == target_variables_len
    assert len(layer2.trainable_variables) == target_trainable_variables_len

    layer3 = GroupNormalization(groups=2, center=center, scale=scale)
    layer3(tf.random.normal(shape=[8, 28, 28, 16]))  # call()
    assert len(layer3.variables) == target_variables_len
    assert len(layer3.trainable_variables) == target_trainable_variables_len


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("scale", [True, False])
def test_instance_norm_compute_output_shape(center, scale):

    target_variables_len = [center, scale].count(True)
    target_trainable_variables_len = [center, scale].count(True)
    layer1 = InstanceNormalization(groups=2, center=center, scale=scale)
    layer1.build(input_shape=[8, 28, 28, 16])  # build()
    assert len(layer1.variables) == target_variables_len
    assert len(layer1.trainable_variables) == target_trainable_variables_len

    layer2 = InstanceNormalization(groups=2, center=center, scale=scale)
    layer2.compute_output_shape(input_shape=[8, 28, 28, 16])  # compute_output_shape()
    assert len(layer2.variables) == target_variables_len
    assert len(layer2.trainable_variables) == target_trainable_variables_len

    layer3 = InstanceNormalization(groups=2, center=center, scale=scale)
    layer3(tf.random.normal(shape=[8, 28, 28, 16]))  # call()
    assert len(layer3.variables) == target_variables_len
    assert len(layer3.trainable_variables) == target_trainable_variables_len


def calculate_frn(
    x, beta=0.2, gamma=1, eps=1e-6, learned_epsilon=False, dtype=np.float32
):
    if learned_epsilon:
        eps = eps + 1e-4
    eps = tf.cast(eps, dtype=dtype)
    nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
    x = x * tf.math.rsqrt(nu2 + tf.abs(eps))
    return gamma * x + beta


def set_random_seed():
    seed = 0x2020
    np.random.seed(seed)
    tf.random.set_seed(seed)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_with_beta(dtype):
    set_random_seed()
    inputs = np.random.rand(28, 28, 1).astype(dtype)
    inputs = np.expand_dims(inputs, axis=0)
    frn = FilterResponseNormalization(
        beta_initializer="ones", gamma_initializer="ones", dtype=dtype
    )
    frn.build((None, 28, 28, 1))
    observed = frn(inputs)
    expected = calculate_frn(inputs, beta=1, gamma=1, dtype=dtype)
    np.testing.assert_allclose(expected[0], observed[0])


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_with_gamma(dtype):
    set_random_seed()
    inputs = np.random.rand(28, 28, 1).astype(dtype)
    inputs = np.expand_dims(inputs, axis=0)
    frn = FilterResponseNormalization(
        beta_initializer="zeros", gamma_initializer="ones", dtype=dtype
    )
    frn.build((None, 28, 28, 1))
    observed = frn(inputs)
    expected = calculate_frn(inputs, beta=0, gamma=1, dtype=dtype)
    np.testing.assert_allclose(expected[0], observed[0])


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_with_epsilon(dtype):
    set_random_seed()
    inputs = np.random.rand(28, 28, 1).astype(dtype)
    inputs = np.expand_dims(inputs, axis=0)
    frn = FilterResponseNormalization(
        beta_initializer=tf.keras.initializers.Constant(0.5),
        gamma_initializer="ones",
        learned_epsilon=True,
        dtype=dtype,
    )
    frn.build((None, 28, 28, 1))
    observed = frn(inputs)
    expected = calculate_frn(
        inputs, beta=0.5, gamma=1, learned_epsilon=True, dtype=dtype
    )
    np.testing.assert_allclose(expected[0], observed[0])


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_keras_model(dtype):
    set_random_seed()
    frn = FilterResponseNormalization(
        beta_initializer="ones", gamma_initializer="ones", dtype=dtype
    )
    random_inputs = np.random.rand(10, 32, 32, 3).astype(dtype)
    random_labels = np.random.randint(2, size=(10,)).astype(dtype)
    input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
    x = frn(input_layer)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.models.Model(input_layer, out)
    model.compile(loss="binary_crossentropy", optimizer="sgd")
    model.fit(random_inputs, random_labels, epochs=2)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_serialization(dtype):
    frn = FilterResponseNormalization(
        beta_initializer="ones", gamma_initializer="ones", dtype=dtype
    )
    serialized_frn = tf.keras.layers.serialize(frn)
    new_layer = tf.keras.layers.deserialize(serialized_frn)
    assert frn.get_config() == new_layer.get_config()


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_eps_gards(dtype):
    set_random_seed()
    random_inputs = np.random.rand(10, 32, 32, 3).astype(np.float32)
    random_labels = np.random.randint(2, size=(10,)).astype(np.float32)
    input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
    frn = FilterResponseNormalization(
        beta_initializer="ones", gamma_initializer="ones", learned_epsilon=True
    )
    initial_eps_value = frn.eps_learned.numpy()[0]
    x = frn(input_layer)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.models.Model(input_layer, out)
    model.compile(loss="binary_crossentropy", optimizer="sgd")
    model.fit(random_inputs, random_labels, epochs=1)
    final_eps_value = frn.eps_learned.numpy()[0]
    assert initial_eps_value != final_eps_value


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_filter_response_normalization_save(tmpdir):
    input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
    frn = FilterResponseNormalization()(input_layer)
    model = tf.keras.Model(input_layer, frn)
    filepath = str(tmpdir / "test.h5")
    model.save(filepath, save_format="h5")
    filepath = str(tmpdir / "test")
    model.save(filepath, save_format="tf")


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_filter_response_norm_compute_output_shape():
    target_variables_len = 2
    target_trainable_variables_len = 2
    layer1 = FilterResponseNormalization()
    layer1.build(input_shape=[8, 28, 28, 16])  # build()
    assert len(layer1.variables) == target_variables_len
    assert len(layer1.trainable_variables) == target_trainable_variables_len

    layer2 = FilterResponseNormalization()
    layer2.compute_output_shape(input_shape=[8, 28, 28, 16])  # compute_output_shape()
    assert len(layer2.variables) == target_variables_len
    assert len(layer2.trainable_variables) == target_trainable_variables_len

    layer3 = FilterResponseNormalization()
    layer3(tf.random.normal(shape=[8, 28, 28, 16]))  # call()
    assert len(layer3.variables) == target_variables_len
    assert len(layer3.trainable_variables) == target_trainable_variables_len
