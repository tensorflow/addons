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
"""Utilities for tf.test.TestCase."""

import contextlib
import inspect
import time
import unittest
import logging
import tensorflow as tf

# TODO: find public API alternative to these
from tensorflow.python.framework.test_util import (  # noqa: F401
    run_all_in_graph_and_eager_modes,
)
from tensorflow.python.framework.test_util import run_deprecated_v1  # noqa: F401
from tensorflow.python.framework.test_util import (  # noqa: F401
    run_in_graph_and_eager_modes,
)
from tensorflow.python.keras.testing_utils import layer_test  # noqa: F401
from tensorflow.python.keras import keras_parameterized  # noqa: F401


@contextlib.contextmanager
def device(use_gpu):
    """Uses gpu when requested and available."""
    if use_gpu and tf.test.is_gpu_available():
        dev = "/device:GPU:0"
    else:
        dev = "/device:CPU:0"
    with tf.device(dev):
        yield


@contextlib.contextmanager
def use_gpu():
    """Uses gpu when requested and available."""
    with device(use_gpu=True):
        yield


def create_virtual_devices(
    num_devices, force_device=None, memory_limit_per_device=1024
):
    """Virtualize a the physical device into logical devices.

    Args:
        num_devices: The number of virtual devices needed.
        force_device: 'CPU'/'GPU'. Defaults to None, where the
            devices is selected based on the system.
        memory_limit_per_device: Specify memory for each
            virtual GPU. Only for GPUs.

    Returns:
        virtual_devices: A list of virtual devices which can be passed to
            tf.distribute.MirroredStrategy()
    """
    if force_device is None:
        device_type = (
            "GPU" if len(tf.config.list_physical_devices("GPU")) > 0 else "CPU"
        )
    else:
        assert force_device in ["CPU", "GPU"]
        device_type = force_device

    physical_devices = tf.config.list_physical_devices(device_type)

    if device_type == "CPU":
        memory_limit_per_device = None

    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [
            tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_per_device)
            for _ in range(num_devices)
        ],
    )

    return tf.config.list_logical_devices(device_type)


def create_or_get_logical_devices(
    num_devices, force_device=None, memory_limit_per_device=1024
):
    if force_device is None:
        device_type = (
            "GPU" if len(tf.config.list_physical_devices("GPU")) > 0 else "CPU"
        )
    else:
        assert force_device in ["CPU", "GPU"]
        device_type = force_device

    logical_devices = tf.config.list_logical_devices(device_type)

    if len(logical_devices) == num_devices:
        # if we have x logical devices already, return logical devices.
        logical_devices_out = logical_devices
    elif len(logical_devices) > num_devices:
        # if we have x logical devices and we are requesting x - n devices, return fewer devices.
        logical_devices_out = logical_devices[:num_devices]
    elif len(logical_devices) < num_devices:
        # if we don't have enough logical devices, try create x devices.
        try:
            logical_devices_out = create_virtual_devices(
                num_devices, force_device, memory_limit_per_device
            )
        except RuntimeError as r:
            if "Virtual devices cannot be modified after being initialized" in str(r):
                logging.info(
                    """Tensorflow does not allow you to initialize devices again.
                Please make sure that your first test initializes x number of devices.
                Afterwards, this function will correctly allocate x - n number of devices, if you need fewer than x.
                Otherwise, this function will correctly allocate x devices.
                Finally, if you ask this function for x + n devices, after the first initialization, you will see this
                error reminding you to initialize x number of devices first, where x is maximum number needed for tests.
                """
                )
                raise r

    return logical_devices_out


def run_all_distributed(num_devices):
    base_decorator = run_distributed(num_devices)

    def decorator(cls):
        for name, method in cls.__dict__.copy().items():
            if (
                callable(method)
                and name.startswith(unittest.TestLoader.testMethodPrefix)
                and name != "test_session"
            ):
                setattr(cls, name, base_decorator(method))
        return cls

    return decorator


# TODO: Add support for other distribution strategies
def run_distributed(num_devices):
    def decorator(f):
        if inspect.isclass(f):
            raise TypeError(
                "`run_distributed` only supports test methods. "
                "Did you mean to use `run_all_distributed`?"
            )

        def decorated(self, *args, **kwargs):
            logical_devices = create_or_get_logical_devices(num_devices)
            strategy = tf.distribute.MirroredStrategy(logical_devices)
            with strategy.scope():
                f(self, *args, **kwargs)

        return decorated

    return decorator


def run_all_with_types(dtypes):
    """Execute all test methods in the given class with and without eager."""
    base_decorator = run_with_types(dtypes)

    def decorator(cls):
        for name, method in cls.__dict__.copy().items():
            if (
                callable(method)
                and name.startswith(unittest.TestLoader.testMethodPrefix)
                and name != "test_session"
            ):
                setattr(cls, name, base_decorator(method))
        return cls

    return decorator


def run_with_types(dtypes):
    def decorator(f):
        if inspect.isclass(f):
            raise TypeError(
                "`run_with_types` only supports test methods. "
                "Did you mean to use `run_all_with_types`?"
            )

        def decorated(self, *args, **kwargs):
            for t in dtypes:
                f(self, *args, dtype=t, **kwargs)

        return decorated

    return decorator


def time_function(f):
    def decorated(self, *args, **kwargs):
        start = time.time()
        f(self, *args, **kwargs)
        end = time.time()
        print(f.__name__, "took", (end - start), "seconds")

    return decorated


def time_all_functions(cls):
    for name, method in cls.__dict__.copy().items():
        if (
            callable(method)
            and name.startswith(unittest.TestLoader.testMethodPrefix)
            and name != "test_session"
        ):
            setattr(cls, name, time_function(method))
    return cls
