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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import numpy as np
import tensorflow as tf
from tensorflow_addons.activations import rrelu
from tensorflow_addons.utils import test_utils

SEED = 111111


@test_utils.run_all_in_graph_and_eager_modes
class RreluTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(("float16", np.float16),
                                    ("float32", np.float32),
                                    ("float64", np.float64))
    def test_rrelu(self, dtype):
        x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=dtype)
        lower = 0.1
        upper = 0.2

        training_results = {
            np.float16: [-0.288330078, -0.124206543, 0, 1, 2],
            np.float32: [-0.26851666, -0.116421416, 0, 1, 2],
            np.float64: [-0.3481333923206531, -0.17150176242558851, 0, 1, 2],
        }
        for training in [True, False]:
            with self.subTest(training=training):
                tf.random.set_seed(SEED)
                result = rrelu(x, lower, upper, training=training, seed=SEED)
                if training:
                    expect_result = training_results.get(dtype)
                else:
                    expect_result = [
                        -0.30000001192092896, -0.15000000596046448, 0, 1, 2
                    ]
                self.assertAllCloseAccordingToType(result, expect_result)

    @parameterized.named_parameters(("float32", np.float32),
                                    ("float64", np.float64))
    def test_theoretical_gradients(self, dtype):
        if tf.executing_eagerly():

            def rrelu_wrapper(lower, upper, training):
                def inner(x):
                    tf.random.set_seed(SEED)
                    return rrelu(x, lower, upper, training=training, seed=SEED)

                return inner

            x = tf.constant([-2.0, -1.0, -0.1, 0.1, 1.0, 2.0], dtype=dtype)
            lower = 0.1
            upper = 0.2

            for training in [True, False]:
                with self.subTest(training=training):
                    theoretical, numerical = tf.test.compute_gradient(
                        rrelu_wrapper(lower, upper, training), [x])
                    self.assertAllCloseAccordingToType(
                        theoretical, numerical, rtol=5e-4, atol=5e-4)


# TODO: Benchmark fails for windows builds #839
# class RreluBenchmarks(tf.test.Benchmark):
#     def benchmarkRreluOp(self):
#         with tf.compat.v1.Session(config=tf.test.benchmark_config()) as sess:
#             x = tf.constant([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
#             lower = 0.1
#             upper = 0.2
#             result = rrelu(x, lower, upper, training=True)
#             self.run_op_benchmark(sess, result.op, min_iters=25)


if __name__ == "__main__":
    tf.test.main()
