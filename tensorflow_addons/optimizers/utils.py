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
"""Additional Utilities used for tfa.optimizers."""

import tensorflow as tf


def fit_bn(model, *args, **kwargs):
    """Resets batch normalization layers of model, and recalculates the
    statistics for each batchnorm layer by running a pass on the data.

    Args:
        model: An instance of tf.keras.Model
        *args, **kwargs: Params that'll be passed to `.fit` method of model
    """
    kwargs["epochs"] = 1
    if not isinstance(model, tf.keras.Model):
        raise TypeError("model must be an instance of tf.keras.Model")

    if not model.built:
        raise ValueError("Call `fit_bn` after the model is built and trained")

    assign_ops = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            assign_ops.extend(
                [
                    layer.moving_mean.assign(tf.zeros_like(layer.moving_mean)),
                    layer.moving_variance.assign(tf.ones_like(layer.moving_variance)),
                ]
            )

    _trainable = model.trainable
    _metrics = model._metrics
    model.trainable = False
    model._metrics = []

    model.fit(*args, **kwargs)

    model.trainable = _trainable
    model._metrics = _metrics
