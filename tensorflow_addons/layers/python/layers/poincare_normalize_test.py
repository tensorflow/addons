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

import numpy as np

from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.framework import constant_op

from tensorflow_addons.layers.python.layers.poincare_normalize import poincare_normalize


# TODO: Is this the prefered way to run tests in TF2?
@test_util.run_all_in_graph_and_eager_modes
class PoincareNormalizeTest(test.TestCase):

    def _PoincareNormalize(self, x, dim, epsilon=1e-5):
        if isinstance(dim, list):
            norm = np.linalg.norm(x, axis=tuple(dim))
            for d in dim:
                norm = np.expand_dims(norm, d)
            norm_x = ((1. - epsilon) * x) / norm
        else:
            norm = np.expand_dims(np.apply_along_axis(np.linalg.norm, dim, x), dim)
            norm_x = ((1. - epsilon) * x) / norm
        return np.where(norm > 1.0 - epsilon, norm_x, x)

    def testPoincareNormalize(self):
        x_shape = [20, 7, 3]
        epsilon = 1e-5
        tol = 1e-6
        np.random.seed(1)
        x_np = np.random.random_sample(x_shape).astype(np.float32)
        for dim in range(len(x_shape)):
            y_np = self._PoincareNormalize(x_np, dim, epsilon)
            with self.cached_session():
                x_tf = constant_op.constant(x_np, name='x')
                y_tf = poincare_normalize(x_tf, dim, epsilon)
                y_tf_eval = y_tf.numpy()
                norm = np.linalg.norm(y_np, axis=dim)
                self.assertLessEqual(norm.max(), 1. - epsilon + tol)
                norm = np.linalg.norm(y_tf_eval, axis=dim)
                self.assertLessEqual(norm.max(), 1. - epsilon + tol)
                self.assertAllClose(y_np, y_tf_eval)

    def testPoincareNormalizeDimArray(self):
        x_shape = [20, 7, 3]
        epsilon = 1e-5
        tol = 1e-6
        np.random.seed(1)
        x_np = np.random.random_sample(x_shape).astype(np.float32)
        dim = [1, 2]
        y_np = self._PoincareNormalize(x_np, dim, epsilon)
        with self.cached_session():
            x_tf = constant_op.constant(x_np, name='x')
            y_tf = poincare_normalize(x_tf, dim, epsilon)
            y_tf_eval = y_tf.numpy()
            norm = np.linalg.norm(y_np, axis=tuple(dim))
            self.assertLess(norm.max(), 1. - epsilon + tol)
            norm = np.linalg.norm(y_tf_eval, axis=tuple(dim))
            self.assertLess(norm.max(), 1. - epsilon + tol)
            self.assertAllClose(y_np, y_tf_eval, rtol=1e-6, atol=1e-6)

    def testPoincareNormalizeGradient(self):
        x_shape = [20, 7, 3]
        np.random.seed(1)
        x_np = np.random.random_sample(x_shape).astype(np.float64)
        for dim in range(len(x_shape)):
            with self.cached_session():
                x_tf = constant_op.constant(x_np, name='x')
                y_tf = poincare_normalize(x_tf, dim)
                err = gradient_checker.compute_gradient_error(x_tf,
                                                              x_shape,
                                                              y_tf,
                                                              x_shape)
            print('PoinCareNormalize gradient err = %g ' % err)
            self.assertLess(err, 1e-4)


if __name__ == '__main__':
    test.main()
