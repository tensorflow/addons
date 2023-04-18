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
"""Types for typing functions signatures."""

from typing import Union, Callable, List

import importlib
import numpy as np
import tensorflow as tf

from packaging.version import Version

# TODO: Remove once https://github.com/tensorflow/tensorflow/issues/44613 is resolved
if Version(tf.__version__).release >= Version("2.13").release:
    # New versions of Keras require importing from `keras.src` when
    # importing internal symbols.
    from keras.src.engine import keras_tensor
elif Version(tf.__version__).release >= Version("2.5").release:
    from keras.engine import keras_tensor
else:
    from tensorflow.python.keras.engine import keras_tensor


Number = Union[
    float,
    int,
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

Initializer = Union[None, dict, str, Callable, tf.keras.initializers.Initializer]
Regularizer = Union[None, dict, str, Callable, tf.keras.regularizers.Regularizer]
Constraint = Union[None, dict, str, Callable, tf.keras.constraints.Constraint]
Activation = Union[None, str, Callable]
if importlib.util.find_spec("tensorflow.keras.optimizers.legacy") is not None:
    Optimizer = Union[
        tf.keras.optimizers.Optimizer, tf.keras.optimizers.legacy.Optimizer, str
    ]
else:
    Optimizer = Union[tf.keras.optimizers.Optimizer, str]

TensorLike = Union[
    List[Union[Number, list]],
    tuple,
    Number,
    np.ndarray,
    tf.Tensor,
    tf.SparseTensor,
    tf.Variable,
    keras_tensor.KerasTensor,
]
FloatTensorLike = Union[tf.Tensor, float, np.float16, np.float32, np.float64]
AcceptableDTypes = Union[tf.DType, np.dtype, type, int, str, None]
