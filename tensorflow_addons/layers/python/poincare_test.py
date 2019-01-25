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
"""Tests for PoincareNormalize layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow_addons.layers.python.poincare import PoincareNormalize


class PoincareNormalizeTest(test.TestCase):
    def _PoincareNormalize(self, x, dim, epsilon=1e-5):
        if isinstance(dim, list):
            norm = np.linalg.norm(x, axis=tuple(dim))
            for d in dim:
                norm = np.expand_dims(norm, d)
            norm_x = ((1. - epsilon) * x) / norm
        else:
            norm = np.expand_dims(
                np.apply_along_axis(np.linalg.norm, dim, x), dim)
            norm_x = ((1. - epsilon) * x) / norm
        return np.where(norm > 1.0 - epsilon, norm_x, x)

    def testPoincareNormalize(self):
        x_shape = [20, 7, 3]
        epsilon = 1e-5
        tol = 1e-6
        np.random.seed(1)
        inputs = np.random.random_sample(x_shape).astype(np.float32)

        for dim in range(len(x_shape)):
            outputs_expected = self._PoincareNormalize(inputs, dim, epsilon)

            outputs = testing_utils.layer_test(
                PoincareNormalize,
                kwargs={
                    'axis': dim,
                    'epsilon': epsilon
                },
                input_data=inputs,
                expected_output=outputs_expected)
            for y in outputs_expected, outputs:
                norm = np.linalg.norm(y, axis=dim)
                self.assertLessEqual(norm.max(), 1. - epsilon + tol)

    def testPoincareNormalizeDimArray(self):
        x_shape = [20, 7, 3]
        epsilon = 1e-5
        tol = 1e-6
        np.random.seed(1)
        inputs = np.random.random_sample(x_shape).astype(np.float32)
        dim = [1, 2]

        outputs_expected = self._PoincareNormalize(inputs, dim, epsilon)

        outputs = testing_utils.layer_test(
            PoincareNormalize,
            kwargs={
                'axis': dim,
                'epsilon': epsilon
            },
            input_data=inputs,
            expected_output=outputs_expected)
        for y in outputs_expected, outputs:
            norm = np.linalg.norm(y, axis=tuple(dim))
            self.assertLessEqual(norm.max(), 1. - epsilon + tol)


if __name__ == '__main__':
    test.main()
