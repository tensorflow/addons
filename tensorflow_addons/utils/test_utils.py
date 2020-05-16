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

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.utils import resource_loader

# TODO: copy the layer_test implementation in Addons.
from tensorflow.python.keras.testing_utils import layer_test  # noqa: F401


NUMBER_OF_WORKERS = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
WORKER_ID = int(os.environ.get("PYTEST_XDIST_WORKER", "gw0")[2])
NUMBER_OF_GPUS = len(tf.config.list_physical_devices("GPU"))

# Some configuration before starting the tests.

# we only need one core per worker.
# This avoids context switching for speed, but it also prevents TensorFlow to go
# crazy on systems with many cores (kokoro has 30+ cores).
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

if NUMBER_OF_GPUS != 0:
    # We use only the first gpu at the moment. That's enough for most use cases.
    # split the first gpu into chunks of 100MB per pytest worker.
    # It's the user's job to limit the amount of pytest workers depending
    # on the available memory.
    # In practice, each process takes a bit more memory.
    # There must be some kind of overhead but it's not very big (~200MB more)
    first_gpu = tf.config.list_physical_devices("GPU")[0]

    tf.config.set_logical_device_configuration(
        first_gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=100)],
    )


def finalizer():
    tf.config.experimental_run_functions_eagerly(False)


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
        tf.config.experimental_run_functions_eagerly(True)
    elif request.param == "tf_function":
        tf.config.experimental_run_functions_eagerly(False)

    request.addfinalizer(finalizer)


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
    requested_device = request.param
    if requested_device == "no_device":
        yield requested_device
    else:
        if requested_device in ["cpu", "gpu"]:
            # we use GPU:0 because the virtual device we created is the
            # only one in the first GPU (so first in the list of virtual devices).
            requested_device += ":0"
        else:
            raise KeyError("Invalid device: " + requested_device)
        with tf.device(requested_device):
            yield requested_device


def get_marks(device_name):
    marks = []
    if device_name == "gpu":
        marks.append(pytest.mark.needs_gpu)
        if NUMBER_OF_GPUS == 0:
            skip_message = "The gpu is not available."
            marks.append(pytest.mark.skip(reason=skip_message))
    return marks


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
