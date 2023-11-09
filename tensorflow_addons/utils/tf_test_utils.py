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
"""Utilities for unit-testing TF-Keras."""


import threading

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow_addons.utils import tf_inspect


def string_test(actual, expected):
    np.testing.assert_array_equal(actual, expected)


def numeric_test(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)


def layer_test(
    layer_cls,
    kwargs=None,
    input_shape=None,
    input_dtype=None,
    input_data=None,
    expected_output=None,
    expected_output_dtype=None,
    expected_output_shape=None,
    validate_training=True,
    adapt_data=None,
    custom_objects=None,
    test_harness=None,
    supports_masking=None,
):
    """Test routine for a layer with a single input and single output.

    Args:
      layer_cls: Layer class object.
      kwargs: Optional dictionary of keyword arguments for instantiating the
        layer.
      input_shape: Input shape tuple.
      input_dtype: Data type of the input data.
      input_data: Numpy array of input data.
      expected_output: Numpy array of the expected output.
      expected_output_dtype: Data type expected for the output.
      expected_output_shape: Shape tuple for the expected shape of the output.
      validate_training: Whether to attempt to validate training on this layer.
        This might be set to False for non-differentiable layers that output
        string or integer values.
      adapt_data: Optional data for an 'adapt' call. If None, adapt() will not
        be tested for this layer. This is only relevant for PreprocessingLayers.
      custom_objects: Optional dictionary mapping name strings to custom objects
        in the layer class. This is helpful for testing custom layers.
      test_harness: The Tensorflow test, if any, that this function is being
        called in.
      supports_masking: Optional boolean to check the `supports_masking`
        property of the layer. If None, the check will not be performed.

    Returns:
      The output data (Numpy array) returned by the layer, for additional
      checks to be done by the calling code.

    Raises:
      ValueError: if `input_shape is None`.
    """
    if input_data is None:
        if input_shape is None:
            raise ValueError("input_shape is None")
        if not input_dtype:
            input_dtype = "float32"
        input_data_shape = list(input_shape)
        for i, e in enumerate(input_data_shape):
            if e is None:
                input_data_shape[i] = np.random.randint(1, 4)
        input_data = 10 * np.random.random(input_data_shape)
        if input_dtype[:5] == "float":
            input_data -= 0.5
        input_data = input_data.astype(input_dtype)
    elif input_shape is None:
        input_shape = input_data.shape
    if input_dtype is None:
        input_dtype = input_data.dtype
    if expected_output_dtype is None:
        expected_output_dtype = input_dtype

    if tf.as_dtype(expected_output_dtype) == tf.string:
        if test_harness:
            assert_equal = test_harness.assertAllEqual
        else:
            assert_equal = string_test
    else:
        if test_harness:
            assert_equal = test_harness.assertAllClose
        else:
            assert_equal = numeric_test

    # instantiation
    kwargs = kwargs or {}
    layer = layer_cls(**kwargs)

    if supports_masking is not None and layer.supports_masking != supports_masking:
        raise AssertionError(
            "When testing layer %s, the `supports_masking` property is %r"
            "but expected to be %r.\nFull kwargs: %s"
            % (
                layer_cls.__name__,
                layer.supports_masking,
                supports_masking,
                kwargs,
            )
        )

    # Test adapt, if data was passed.
    if adapt_data is not None:
        layer.adapt(adapt_data)

    # test get_weights , set_weights at layer level
    weights = layer.get_weights()
    layer.set_weights(weights)

    # test and instantiation from weights
    if "weights" in tf_inspect.getargspec(layer_cls.__init__):
        kwargs["weights"] = weights
        layer = layer_cls(**kwargs)

    # test in functional API
    x = layers.Input(shape=input_shape[1:], dtype=input_dtype)
    y = layer(x)
    if backend.dtype(y) != expected_output_dtype:
        raise AssertionError(
            "When testing layer %s, for input %s, found output "
            "dtype=%s but expected to find %s.\nFull kwargs: %s"
            % (
                layer_cls.__name__,
                x,
                backend.dtype(y),
                expected_output_dtype,
                kwargs,
            )
        )

    def assert_shapes_equal(expected, actual):
        """Asserts that the output shape from the layer matches the actual
        shape."""
        if len(expected) != len(actual):
            raise AssertionError(
                "When testing layer %s, for input %s, found output_shape="
                "%s but expected to find %s.\nFull kwargs: %s"
                % (layer_cls.__name__, x, actual, expected, kwargs)
            )

        for expected_dim, actual_dim in zip(expected, actual):
            if expected_dim is not None and expected_dim != actual_dim:
                raise AssertionError(
                    "When testing layer %s, for input %s, found output_shape="
                    "%s but expected to find %s.\nFull kwargs: %s"
                    % (layer_cls.__name__, x, actual, expected, kwargs)
                )

    if expected_output_shape is not None:
        assert_shapes_equal(tf.TensorShape(expected_output_shape), y.shape)

    # check shape inference
    model = models.Model(x, y)
    computed_output_shape = tuple(
        layer.compute_output_shape(tf.TensorShape(input_shape)).as_list()
    )
    computed_output_signature = layer.compute_output_signature(
        tf.TensorSpec(shape=input_shape, dtype=input_dtype)
    )
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    assert_shapes_equal(computed_output_shape, actual_output_shape)
    assert_shapes_equal(computed_output_signature.shape, actual_output_shape)
    if computed_output_signature.dtype != actual_output.dtype:
        raise AssertionError(
            "When testing layer %s, for input %s, found output_dtype="
            "%s but expected to find %s.\nFull kwargs: %s"
            % (
                layer_cls.__name__,
                x,
                actual_output.dtype,
                computed_output_signature.dtype,
                kwargs,
            )
        )
    if expected_output is not None:
        assert_equal(actual_output, expected_output)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = models.Model.from_config(model_config, custom_objects)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        output = recovered_model.predict(input_data)
        assert_equal(output, actual_output)

    # test training mode (e.g. useful for dropout tests)
    # Rebuild the model to avoid the graph being reused between predict() and
    # See b/120160788 for more details. This should be mitigated after 2.0.
    layer_weights = layer.get_weights()  # Get the layer weights BEFORE training.
    if validate_training:
        model = models.Model(x, layer(x))
        if _thread_local_data.run_eagerly is not None:
            model.compile(
                "rmsprop",
                "mse",
                weighted_metrics=["acc"],
                run_eagerly=should_run_eagerly(),
            )
        else:
            model.compile("rmsprop", "mse", weighted_metrics=["acc"])
        model.train_on_batch(input_data, actual_output)

    # test as first layer in Sequential API
    layer_config = layer.get_config()
    layer_config["batch_input_shape"] = input_shape
    layer = layer.__class__.from_config(layer_config)

    # Test adapt, if data was passed.
    if adapt_data is not None:
        layer.adapt(adapt_data)

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape[1:], dtype=input_dtype))
    model.add(layer)

    layer.set_weights(layer_weights)
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    for expected_dim, actual_dim in zip(computed_output_shape, actual_output_shape):
        if expected_dim is not None:
            if expected_dim != actual_dim:
                raise AssertionError(
                    "When testing layer %s **after deserialization**, "
                    "for input %s, found output_shape="
                    "%s but expected to find inferred shape %s.\n"
                    "Full kwargs: %s"
                    % (
                        layer_cls.__name__,
                        x,
                        actual_output_shape,
                        computed_output_shape,
                        kwargs,
                    )
                )
    if expected_output is not None:
        assert_equal(actual_output, expected_output)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = models.Sequential.from_config(model_config, custom_objects)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        output = recovered_model.predict(input_data)
        assert_equal(output, actual_output)

    # for further checks in the caller function
    return actual_output


_thread_local_data = threading.local()
_thread_local_data.model_type = None
_thread_local_data.run_eagerly = None
_thread_local_data.saved_model_format = None
_thread_local_data.save_kwargs = None


def should_run_eagerly():
    """Returns whether the models we are testing should be run eagerly."""
    return _thread_local_data.run_eagerly and tf.executing_eagerly()
