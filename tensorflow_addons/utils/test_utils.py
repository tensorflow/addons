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
"""Utilities for testing Addons."""

import os
import random
import inspect

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons import options
from tensorflow_addons.utils import resource_loader


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
):
    """Test routine for a layer with a single input and single output.
    Arguments:
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

    # instantiation
    kwargs = kwargs or {}
    layer = layer_cls(**kwargs)

    # Test adapt, if data was passed.
    if adapt_data is not None:
        layer.adapt(adapt_data)

    x = tf.keras.Input(shape=input_shape[1:], dtype=input_dtype)
    y = layer(x)
    model = tf.keras.Model(x, y)
    actual_output = model.predict(input_data)
    model_config = model.get_config()
    recovered_model = tf.keras.models.model_from_config(model_config, custom_objects)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        output = recovered_model.predict(input_data)
        np.testing.assert_allclose(output, actual_output, rtol=1e-3, atol=1e-6)

    output_tensor = layer(input_data).numpy()
    # model = tf.keras.Sequential()
    # model.add(tf.keras.Input(shape=input_shape[1:], dtype=input_dtype))
    # model.add(layer)
    # actual_output = model.predict(input_data)
    return output_tensor


NUMBER_OF_WORKERS = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
WORKER_ID = int(os.environ.get("PYTEST_XDIST_WORKER", "gw0")[2])
NUMBER_OF_GPUS = len(tf.config.list_physical_devices("GPU"))


def is_gpu_available():
    return NUMBER_OF_GPUS >= 1


# Some configuration before starting the tests.

# we only need one core per worker.
# This avoids context switching for speed, but it also prevents TensorFlow to go
# crazy on systems with many cores (kokoro has 30+ cores).
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

if is_gpu_available():
    # We use only the first gpu at the moment. That's enough for most use cases.
    # split the first gpu into chunks of 100MB per virtual device.
    # It's the user's job to limit the amount of pytest workers depending
    # on the available memory.
    # In practice, each process takes a bit more memory.
    # There must be some kind of overhead but it's not very big (~200MB more)
    # Each worker has two virtual devices.
    # When running on gpu, only the first device is used. The other one is used
    # in distributed strategies.
    first_gpu = tf.config.list_physical_devices("GPU")[0]
    virtual_gpus = [
        tf.config.LogicalDeviceConfiguration(memory_limit=100) for _ in range(2)
    ]

    tf.config.set_logical_device_configuration(first_gpu, virtual_gpus)


def finalizer():
    tf.config.run_functions_eagerly(False)


def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, tf.DType):
        return val.name
    if val is False:
        return "no_" + argname
    if val is True:
        return argname


@pytest.fixture(scope="function", params=["eager_mode", "tf_function"])
def maybe_run_functions_eagerly(request):
    if request.param == "eager_mode":
        tf.config.run_functions_eagerly(True)
    elif request.param == "tf_function":
        tf.config.run_functions_eagerly(False)

    request.addfinalizer(finalizer)


@pytest.fixture(scope="function")
def only_run_functions_eagerly(request):
    tf.config.run_functions_eagerly(True)
    request.addfinalizer(finalizer)


@pytest.fixture(scope="function", params=["custom_ops", "py_ops"])
def run_custom_and_py_ops(request):
    previous_is_custom_kernel_disabled = options.is_custom_kernel_disabled()
    if request.param == "custom_ops":
        options.enable_custom_kernel()
    elif request.param == "py_ops":
        options.disable_custom_kernel()

    def _restore_py_ops_value():
        if previous_is_custom_kernel_disabled:
            options.disable_custom_kernel()
        else:
            options.enable_custom_kernel()

    request.addfinalizer(_restore_py_ops_value)


@pytest.fixture(scope="function", params=["float32", "mixed_float16"])
def run_with_mixed_precision_policy(request):
    tf.keras.mixed_precision.experimental.set_policy(request.param)
    yield
    tf.keras.mixed_precision.experimental.set_policy("float32")


@pytest.fixture(scope="function", params=["channels_first", "channels_last"])
def data_format(request):
    return request.param


@pytest.fixture(scope="function", autouse=True)
def set_seeds():
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)


def pytest_addoption(parser):
    parser.addoption(
        "--skip-custom-ops",
        action="store_true",
        help="When a custom op is being loaded in a test, skip this test.",
    )


def gpus_for_testing():
    """For the moment it's very simple, but it might change in the future,
    with multiple physical gpus for example. So it's better if this function
    is called rather than hardcoding the gpu devices in the tests.
    """
    if not is_gpu_available():
        raise SystemError(
            "You are trying to get some gpus for testing but no gpu is available on "
            "your system. \nDid you forget to use `@pytest.mark.needs_gpu` on your test"
            " so that it's skipped automatically when no gpu is available?"
        )
    return ["gpu:0", "gpu:1"]


@pytest.fixture(scope="session", autouse=True)
def set_global_variables(request):
    if request.config.getoption("--skip-custom-ops"):
        resource_loader.SKIP_CUSTOM_OPS = True


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "with_device(devices): mark test to run on specific devices."
    )
    config.addinivalue_line("markers", "needs_gpu: mark test that needs a gpu.")


@pytest.fixture(autouse=True, scope="function")
def device(request):
    try:
        requested_device = request.param
    except Exception:
        # workaround for DocTestItem
        # https://github.com/pytest-dev/pytest/issues/5070
        requested_device = "no_device"
    if requested_device == "no_device":
        yield requested_device
    elif requested_device == tf.distribute.MirroredStrategy:
        strategy = requested_device(gpus_for_testing())
        with strategy.scope():
            yield strategy
    elif isinstance(requested_device, str):
        if requested_device in ["cpu", "gpu"]:
            # we use GPU:0 because the virtual device we created is the
            # only one in the first GPU (so first in the list of virtual devices).
            requested_device += ":0"
        else:
            raise KeyError("Invalid device: " + requested_device)
        with tf.device(requested_device):
            yield requested_device


def get_marks(device_name):
    if device_name == "gpu" or device_name == tf.distribute.MirroredStrategy:
        return [pytest.mark.needs_gpu]
    else:
        return []


def pytest_generate_tests(metafunc):
    marker = metafunc.definition.get_closest_marker("with_device")
    if marker is None:
        # tests which don't have the "with_device" mark are executed on CPU
        # to ensure reproducibility. We can't let TensorFlow decide
        # where to place the ops.
        devices = ["cpu"]
    else:
        devices = marker.args[0]

    parameters = [pytest.param(x, marks=get_marks(x)) for x in devices]
    metafunc.parametrize("device", parameters, indirect=True)


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("needs_gpu") is not None:
            if not is_gpu_available():
                item.add_marker(pytest.mark.skip("The gpu is not available."))


def assert_not_allclose(a, b, **kwargs):
    """Assert that two numpy arrays, do not have near values.

    Args:
      a: the first value to compare.
      b: the second value to compare.
      **kwargs: additional keyword arguments to be passed to the underlying
        `np.testing.assert_allclose` call.

    Raises:
      AssertionError: If `a` and `b` are unexpectedly close at all elements.
    """
    try:
        np.testing.assert_allclose(a, b, **kwargs)
    except AssertionError:
        return
    raise AssertionError("The two values are close at all elements")


def assert_allclose_according_to_type(
    a,
    b,
    rtol=1e-6,
    atol=1e-6,
    float_rtol=1e-6,
    float_atol=1e-6,
    half_rtol=1e-3,
    half_atol=1e-3,
    bfloat16_rtol=1e-2,
    bfloat16_atol=1e-2,
):
    """
    Similar to tf.test.TestCase.assertAllCloseAccordingToType()
    but this doesn't need a subclassing to run.
    """
    a = np.array(a)
    b = np.array(b)
    # types with lower tol are put later to overwrite previous ones.
    if (
        a.dtype == np.float32
        or b.dtype == np.float32
        or a.dtype == np.complex64
        or b.dtype == np.complex64
    ):
        rtol = max(rtol, float_rtol)
        atol = max(atol, float_atol)
    if a.dtype == np.float16 or b.dtype == np.float16:
        rtol = max(rtol, half_rtol)
        atol = max(atol, half_atol)
    if a.dtype == tf.bfloat16.as_numpy_dtype or b.dtype == tf.bfloat16.as_numpy_dtype:
        rtol = max(rtol, bfloat16_rtol)
        atol = max(atol, bfloat16_atol)

    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def discover_classes(module, parent, class_exceptions):
    """
    Args:
        module: a module in which to search for classes that inherit from the parent class
        parent: the parent class that identifies classes in the module that should be tested
        class_exceptions: a list of specific classes that should be excluded when discovering classes in a module

    Returns:
        a list of classes for testing using pytest for parameterized tests
    """

    classes = [
        class_info[1]
        for class_info in inspect.getmembers(module, inspect.isclass)
        if issubclass(class_info[1], parent) and not class_info[0] in class_exceptions
    ]

    return classes
