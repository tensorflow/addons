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

import inspect
import unittest

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
            raise ValueError("`run_with_types` only supports test methods. "
                             "Did you mean to use `run_all_with_types`?")

        def decorated(self, *args, **kwargs):
            for t in dtypes:
                f(self, *args, dtype=t, **kwargs)

        return decorated

    return decorator
