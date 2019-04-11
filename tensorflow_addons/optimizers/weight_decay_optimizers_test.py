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
"""Tests for optimizers with weight decay."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.utils import test_utils
from tensorflow_addons.optimizers.optimizer_test_base import OptimizerTestBase
from tensorflow_addons.optimizers import weight_decay_optimizers

WEIGHT_DECAY = 0.01


def adamw_update_numpy(param, grad_t, slot_vars, optimizer_params):
    """Numpy update function for AdamW."""
    opt_params = (
        optimizer_params[k] for k in
        ["learning_rate", "beta_1", "beta_2", "epsilon", "weight_decay"])
    lr, beta1, beta2, eps, wd = (v() if callable(v) else v for v in opt_params)
    t = slot_vars.get("t", 0) + 1
    lr_t = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
    slot_vars["m"] = beta1 * slot_vars.get("m", 0) + (1 - beta1) * grad_t
    slot_vars["v"] = beta2 * slot_vars.get("v", 0) + (1 - beta2) * grad_t**2
    param_t = (param * (1 - wd) -
               lr_t * slot_vars["m"] / (np.sqrt(slot_vars["v"]) + eps))
    slot_vars["t"] = t
    return param_t, slot_vars


def sgdw_update_numpy(param, grad_t, slot_vars, optimizer_params):
    """Numpy update function for SGDW."""
    m = slot_vars.get("m", 0)
    optimizer_params = {
        k: v() if callable(v) else v
        for k, v in optimizer_params.items()
    }
    slot_vars["m"] = optimizer_params["momentum"] * m + grad_t
    lr = optimizer_params["learning_rate"]
    wd = optimizer_params["weight_decay"]
    param_t = param * (1 - wd) - lr * slot_vars["m"]
    return param_t, slot_vars


class AdamWOptimizerTest(OptimizerTestBase):

    opt_params = {
        "learning_rate": 0.001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": WEIGHT_DECAY
    }
    callable_opt_params = {k: (lambda: v) for k, v in opt_params.items()}
    optimizer = weight_decay_optimizers.AdamWOptimizer

    @test_utils.run_in_graph_and_eager_modes(reset_test=True)
    def testSparse(self):
        self.doTest(
            self.optimizer,
            adamw_update_numpy,
            self.opt_params,
            do_sparse=True)

    @test_utils.run_in_graph_and_eager_modes
    def testSparseRepeatedIndices(self):
        self.doTestSparseRepeatedIndices(self.optimizer, self.opt_params)

    @test_utils.run_in_graph_and_eager_modes(reset_test=True)
    def testBasic(self):
        self.doTest(self.optimizer, adamw_update_numpy, self.opt_params)

    def testBasicCallableParams(self):
        self.doTest(self.optimizer, adamw_update_numpy,
                    self.callable_opt_params)


class SGDWOptimizerTest(OptimizerTestBase):

    opt_params = {
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": WEIGHT_DECAY
    }
    callable_opt_params = {k: (lambda: v) for k, v in opt_params.items()}
    optimizer = weight_decay_optimizers.SGDWOptimizer

    @test_utils.run_in_graph_and_eager_modes(reset_test=True)
    def testSparse(self):
        self.doTest(
            self.optimizer, sgdw_update_numpy, self.opt_params, do_sparse=True)

    @test_utils.run_in_graph_and_eager_modes
    def testSparseRepeatedIndices(self):
        self.doTestSparseRepeatedIndices(self.optimizer, self.opt_params)

    @test_utils.run_in_graph_and_eager_modes(reset_test=True)
    def testBasic(self):
        self.doTest(self.optimizer, sgdw_update_numpy, self.opt_params)

    def testBasicCallableParams(self):
        self.doTest(self.optimizer, sgdw_update_numpy,
                    self.callable_opt_params)


class ExtendWithWeightDecayTest(SGDWOptimizerTest):
    """Verify that the factory function SGDW is the same as SGDW."""

    optimizer = weight_decay_optimizers.extend_with_decoupled_weight_decay(
        tf.optimizers.SGD)


if __name__ == "__main__":
    tf.test.main()
