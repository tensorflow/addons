# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may noa use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import median_filter_2d as md
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class Median2DTest(tf.test.TestCase):
    def _validateMedian_2d(self, inputs, expected_values, filter_shape=(3, 3)):

        values_op = md.median_filter_2D(inputs)
        with self.test_session(use_gpu=False) as sess:
            if tf.executing_eagerly():
                expected_values = expected_values.numpy()
                values = values_op.numpy()
            else:
                expected_values = expected_values.eval()
                values = values_op.eval()
            self.assertShapeEqual(values, inputs)
            self.assertShapeEqual(expected_values, values_op)
            self.assertAllClose(expected_values, values)

    def testfiltertuple(self):
        tf_img = tf.zeros([3, 4, 3], tf.int32)

        with self.assertRaisesRegexp(TypeError,
                                     'Filter shape must be a tuple'):
            md.median_filter_2D(tf_img, 3)
            md.median_filter_2D(tf_img, 3.5)
            md.median_filter_2D(tf_img, 'dt')
            md.median_filter_2D(tf_img, None)

        filter_shape = (3, 3, 3)
        msg = 'Filter shape must be a tuple of 2 integers. ' \
              'Got %s values in tuple' % len(filter_shape)
        with self.assertRaisesRegexp(ValueError, msg):
            md.median_filter_2D(tf_img, filter_shape)

        with self.assertRaisesRegexp(TypeError,
                                     'Size of the filter must be Integers'):
            md.median_filter_2D(tf_img, (3.5, 3))
            md.median_filter_2D(tf_img, (None, 3))

    def testfiltervalue(self):
        tf_img = tf.zeros([3, 4, 3], tf.int32)

        with self.assertRaises(ValueError):
            md.median_filter_2D(tf_img, (4, 3))

    def testDimension(self):
        tf.compat.v1.disable_eager_execution()
        tf_img = tf.compat.v1.placeholder(tf.int32, shape=[3, 4, None])
        tf_img1 = tf.compat.v1.placeholder(tf.int32, shape=[3, None, 4])
        tf_img2 = tf.compat.v1.placeholder(tf.int32, shape=[None, 3, 4])

        with self.assertRaises(TypeError):
            md.median_filter_2D(tf_img)
            md.median_filter_2D(tf_img1)
            md.median_filter_2D(tf_img2)

    def test_imagevsfilter(self):
        tf_img = tf.zeros([3, 4, 3], tf.int32)
        m = tf_img.shape[0]
        no = tf_img.shape[1]
        ch = tf_img.shape[2]
        filter_shape = (3, 5)
        with self.assertRaises(ValueError):
            md.median_filter_2D(tf_img, filter_shape)

    def testcase(self):
        tf_img = [[[0.32801723, 0.08863795, 0.79119259],
                   [0.35526001, 0.79388736, 0.55435993],
                   [0.11607035, 0.55673079, 0.99473371]],
                  [[0.53240645, 0.74684819, 0.33700031],
                   [0.01760473, 0.28181609, 0.9751476],
                   [0.01605137, 0.8292904, 0.56405609]],
                  [[0.57215374, 0.10155051, 0.64836128],
                   [0.36533048, 0.91401874, 0.02524159],
                   [0.56379134, 0.9028874, 0.19505117]]]

        tf_img = tf.convert_to_tensor(value=tf_img)
        expt = [[[0, 0, 0], [4, 71, 141], [0, 0, 0]],
                [[83, 25, 85], [90, 190, 143], [4, 141, 49]],
                [[0, 0, 0], [4, 71, 49], [0, 0, 0]]]
        expt = tf.convert_to_tensor(value=expt)
        self._validateMedian_2d(tf_img, expt)


if __name__ == "__main__":
    tf.test.main()
