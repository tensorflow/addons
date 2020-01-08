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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import inspect
import unittest

import tensorflow as tf
# yapf: disable
# pylint: disable=unused-import
# TODO: find public API alternative to these
from tensorflow.python.framework.test_util import run_all_in_graph_and_eager_modes
from tensorflow.python.framework.test_util import run_deprecated_v1
from tensorflow.python.framework.test_util import run_in_graph_and_eager_modes
from tensorflow.python.keras.testing_utils import layer_test
from tensorflow.python.keras import keras_parameterized

# pylint: enable=unused-import
# yapf: enable


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


def create_virtual_devices(num_devices, memory_limit_per_device=1024):
    """Virtualize a the physical device into `num_devices` logical devices."""
    is_gpu_available = len(tf.config.list_physical_devices('GPU'))
    device_type = 'GPU' if is_gpu_available > 0 else 'CPU'
    physical_devices = tf.config.list_physical_devices(device_type)

    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0], [
            tf.config.experimental.VirtualDeviceConfiguration(
                memory_limit=memory_limit_per_device
                if is_gpu_available else None) for _ in range(num_devices)
        ])

    return tf.config.experimental.list_logical_devices(device_type)


def run_all_distributed(num_devices):
    base_decorator = run_distributed(num_devices)

    def decorator(cls):
        for name, method in cls.__dict__.copy().items():
            if (callable(method)
                    and name.startswith(unittest.TestLoader.testMethodPrefix)
                    and name != "test_session"):
                setattr(cls, name, base_decorator(method))
        return cls

    return decorator


def run_distributed(num_devices):
    def decorator(f):
        if inspect.isclass(f):
            raise TypeError("`run_distributed` only supports test methods. "
                            "Did you mean to use `run_all_distributed`?")

        def decorated(self, *args, **kwargs):
            logical_devices = create_virtual_devices(num_devices)
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
            if (callable(method)
                    and name.startswith(unittest.TestLoader.testMethodPrefix)
                    and name != "test_session"):
                setattr(cls, name, base_decorator(method))
        return cls

    return decorator


def run_with_types(dtypes):
    def decorator(f):
        if inspect.isclass(f):
            raise TypeError("`run_with_types` only supports test methods. "
                            "Did you mean to use `run_all_with_types`?")

        def decorated(self, *args, **kwargs):
            for t in dtypes:
                f(self, *args, dtype=t, **kwargs)

        return decorated

    return decorator
