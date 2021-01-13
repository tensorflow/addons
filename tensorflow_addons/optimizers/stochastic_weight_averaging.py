# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""An implementation of the Stochastic Weight Averaging optimizer.

The Stochastic Weight Averaging mechanism was proposed by Pavel Izmailov
et. al in the paper [Averaging Weights Leads to Wider Optima and Better
Generalization](https://arxiv.org/abs/1803.05407). The optimizer
implements averaging of multiple points along the trajectory of SGD.
This averaging has shown to improve model performance on validation/test
sets whilst possibly causing a small increase in loss on the training
set.
"""

import tensorflow as tf
from tensorflow_addons.optimizers.average_wrapper import AveragedOptimizerWrapper
from tensorflow_addons.utils import types

from typeguard import typechecked


@tf.keras.utils.register_keras_serializable(package="Addons")
class SWA(AveragedOptimizerWrapper):
    """This class extends optimizers with Stochastic Weight Averaging (SWA).

    The Stochastic Weight Averaging mechanism was proposed by Pavel Izmailov
    et. al in the paper [Averaging Weights Leads to Wider Optima and
    Better Generalization](https://arxiv.org/abs/1803.05407). The optimizer
    implements averaging of multiple points along the trajectory of SGD. The
    optimizer expects an inner optimizer which will be used to apply the
    gradients to the variables and itself computes a running average of the
    variables every `k` steps (which generally corresponds to the end
    of a cycle when a cyclic learning rate is employed).

    We also allow the specification of the number of steps averaging
    should first happen after. Let's say, we want averaging to happen every `k`
    steps after the first `m` steps. After step `m` we'd take a snapshot of the
    variables and then average the weights appropriately at step `m + k`,
    `m + 2k` and so on. The assign_average_vars function can be called at the
    end of training to obtain the averaged_weights from the optimizer.

    Note: If your model has batch-normalization layers you would need to run
    the final weights through the data to compute the running mean and
    variance corresponding to the activations for each layer of the network.
    From the paper: If the DNN uses batch normalization we run one
    additional pass over the data, to compute the running mean and standard
    deviation of the activations for each layer of the network with SWA
    weights after the training is finished, since these statistics are not
    collected during training. For most deep learning libraries, such as
    PyTorch or Tensorflow, one can typically collect these statistics by
    making a forward pass over the data in training mode
    ([Averaging Weights Leads to Wider Optima and Better
    Generalization](https://arxiv.org/abs/1803.05407))

    Example of usage:

    ```python
    opt = tf.keras.optimizers.SGD(learning_rate)
    opt = tfa.optimizers.SWA(opt, start_averaging=m, average_period=k)
    ```
    """

    @typechecked
    def __init__(
        self,
        optimizer: types.Optimizer,
        start_averaging: int = 0,
        average_period: int = 10,
        name: str = "SWA",
        **kwargs,
    ):
        r"""Wrap optimizer with the Stochastic Weight Averaging mechanism.

        Args:
            optimizer: The original optimizer that will be used to compute and
                apply the gradients.
            start_averaging: An integer. Threshold to start averaging using
                SWA. Averaging only occurs at `start_averaging` iters, must
                be >= 0. If start_averaging = m, the first snapshot will be
                taken after the mth application of gradients (where the first
                iteration is iteration 0).
            average_period: An integer. The synchronization period of SWA. The
                averaging occurs every average_period steps. Averaging period
                needs to be >= 1.
            name: Optional name for the operations created when applying
                gradients. Defaults to 'SWA'.
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(optimizer, name, **kwargs)

        if average_period < 1:
            raise ValueError("average_period must be >= 1")
        if start_averaging < 0:
            raise ValueError("start_averaging must be >= 0")

        self._set_hyper("average_period", average_period)
        self._set_hyper("start_averaging", start_averaging)

    @tf.function
    def average_op(self, var, average_var, local_apply_state):
        average_period = self._get_hyper("average_period", tf.dtypes.int64)
        start_averaging = self._get_hyper("start_averaging", tf.dtypes.int64)
        # number of times snapshots of weights have been taken (using max to
        # avoid negative values of num_snapshots).
        num_snapshots = tf.math.maximum(
            tf.cast(0, tf.int64),
            tf.math.floordiv(self.iterations - start_averaging, average_period),
        )

        # The average update should happen iff two conditions are met:
        # 1. A min number of iterations (start_averaging) have taken place.
        # 2. Iteration is one in which snapshot should be taken.
        checkpoint = start_averaging + num_snapshots * average_period
        if self.iterations >= start_averaging and self.iterations == checkpoint:
            num_snapshots = tf.cast(num_snapshots, tf.float32)
            average_value = (average_var * num_snapshots + var) / (num_snapshots + 1.0)
            return average_var.assign(average_value, use_locking=self._use_locking)

        return average_var

    def get_config(self):
        config = {
            "average_period": self._serialize_hyperparameter("average_period"),
            "start_averaging": self._serialize_hyperparameter("start_averaging"),
        }
        base_config = super().get_config()
        return {**base_config, **config}
