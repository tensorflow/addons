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
from tensorflow_addons.image import mean_filter2d
from tensorflow_addons.image import median_filter2d
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class MeanFilter2dTest(tf.test.TestCase):
    def test_invalid_image(self):
        msg = "image should be either 3 or 4-dimensional."

        for image_shape in [(28, 28), (16, 28, 28, 1, 1)]:
            with self.subTest(dim=len(image_shape)):
                with self.assertRaisesRegexp(ValueError, msg):
                    mean_filter2d(tf.ones(shape=image_shape))

    def test_invalid_filter_shape(self):
        msg = ("The `filter_shape` argument must be a tuple of " "2 integers.")
        image = tf.ones(shape=(1, 28, 28, 1))

        for filter_shape in [(3, 3, 3), (3, None, 3), None]:
            with self.subTest(filter_shape=filter_shape):
                with self.assertRaisesRegexp(ValueError, msg):
                    mean_filter2d(image, filter_shape=filter_shape)

    def test_invalid_padding(self):
        msg = ("padding should be one of \"REFLECT\", \"CONSTANT\", "
               "or \"SYMMETRIC\".")
        image = tf.ones(shape=(1, 28, 28, 1))

        with self.assertRaisesRegexp(ValueError, msg):
            mean_filter2d(image, padding="TEST")

    def test_3d_image(self):
        # Test shape (3, 3, 1)
        image = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                            shape=(3, 3, 1))

        # (3, 3) filter shape:
        # reflected padding with 1 pixel to each direction
        #   5 4 5 6 5
        #   2 1 2 3 2
        #   5 4 5 6 5
        #   8 7 8 9 8
        #   5 4 5 6 5

        expected = tf.constant(
            [[33. / 9., 36. / 9., 39. / 9.], [42. / 9., 45. / 9., 48. / 9.],
             [51. / 9., 54. / 9., 57. / 9.]],
            shape=(3, 3, 1))

        output = mean_filter2d(image)
        self.assertAllClose(output, expected)

        # Test shape (3, 3, 3)
        image = tf.stack([image, 2. * image, 3. * image], axis=-1)
        expected = tf.stack([expected, 2. * expected, 3. * expected], axis=-1)
        # Squeeze shape from (3, 3, 1, 3) to (3, 3, 3)
        image = tf.squeeze(image, axis=-2)
        expected = tf.squeeze(expected, axis=-2)

        output = mean_filter2d(image)
        self.assertAllClose(output, expected)

    def test_4d_image(self):
        # Test shape (2, 3, 3, 1)
        image = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                            shape=(3, 3, 1))

        expected = tf.constant(
            [[33. / 9., 36. / 9., 39. / 9.], [42. / 9., 45. / 9., 48. / 9.],
             [51. / 9., 54. / 9., 57. / 9.]],
            shape=(3, 3, 1))

        # Batch size = 2, shape = (2, 3, 3, 1)
        image = tf.stack([image, 2. * image], axis=0)
        expected = tf.stack([expected, 2. * expected], axis=0)

        output = mean_filter2d(image)
        self.assertAllClose(output, expected)

        # Test shape (2, 3, 3, 3)
        image = tf.stack([image, image, image], axis=-1)
        expected = tf.stack([expected, expected, expected], axis=-1)
        image = tf.squeeze(image, axis=-2)
        expected = tf.squeeze(expected, axis=-2)

        output = mean_filter2d(image)
        self.assertAllClose(output, expected)

    def test_zero_padding(self):
        image = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                            shape=(3, 3, 1))

        # (3, 3) filter shape:
        # zero padding with 1 pixel to each direction
        #   0 0 0 0 0
        #   0 1 2 3 0
        #   0 4 5 6 0
        #   0 7 8 9 0
        #   0 0 0 0 0

        expected = tf.constant(
            [[12. / 9., 21. / 9., 16. / 9.], [27. / 9., 45. / 9., 33. / 9.],
             [24. / 9., 39. / 9., 28. / 9.]],
            shape=(3, 3, 1))

        output = mean_filter2d(image, padding="CONSTANT", constant_values=0)
        self.assertAllClose(output, expected)

    def test_symmetric_padding(self):
        image = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                            shape=(3, 3, 1))

        # (3, 3) filter shape:
        # symmetric padding with 1 pixel to each direction
        #   1 1 2 3 3
        #   1 1 2 3 3
        #   4 4 5 6 6
        #   7 7 8 9 9
        #   7 7 8 9 9

        expected = tf.constant(
            [[21. / 9., 27. / 9., 33. / 9.], [39. / 9., 45. / 9., 51. / 9.],
             [57. / 9., 63. / 9., 69. / 9.]],
            shape=(3, 3, 1))

        output = mean_filter2d(image, padding="SYMMETRIC")
        self.assertAllClose(output, expected)


class MedianFilter2dTest(tf.test.TestCase):
    def _validate_median_filter2d(self,
                                  inputs,
                                  expected_values,
                                  filter_shape=(3, 3)):
        output = median_filter2d(inputs, filter_shape)
        self.assertAllClose(output, expected_values)

    @test_utils.run_in_graph_and_eager_modes
    def test_filter_tuple(self):
        tf_img = tf.zeros([3, 4, 3], tf.int32)

        for filter_shape in [3, 3.5, 'dt', None]:
            with self.assertRaisesRegexp(TypeError,
                                         'Filter shape must be a tuple'):
                median_filter2d(tf_img, filter_shape)

        filter_shape = (3, 3, 3)
        msg = ('Filter shape must be a tuple of 2 integers. '
               'Got %s values in tuple' % len(filter_shape))
        with self.assertRaisesRegexp(ValueError, msg):
            median_filter2d(tf_img, filter_shape)

        msg = 'Size of the filter must be Integers'
        for filter_shape in [(3.5, 3), (None, 3)]:
            with self.assertRaisesRegexp(TypeError, msg):
                median_filter2d(tf_img, filter_shape)

    @test_utils.run_in_graph_and_eager_modes
    def test_filter_value(self):
        tf_img = tf.zeros([3, 4, 3], tf.int32)

        with self.assertRaises(ValueError):
            median_filter2d(tf_img, (4, 3))

    @test_utils.run_deprecated_v1
    def test_dimension(self):
        for image_shape in [(3, 4, None), (3, None, 4), (None, 3, 4)]:
            with self.assertRaises(TypeError):
                tf_img = tf.compat.v1.placeholder(tf.int32, shape=image_shape)
                median_filter2d(tf_img)

    @test_utils.run_in_graph_and_eager_modes
    def test_image_vs_filter(self):
        tf_img = tf.zeros([3, 4, 3], tf.int32)
        filter_shape = (3, 5)
        with self.assertRaises(ValueError):
            median_filter2d(tf_img, filter_shape)

    @test_utils.run_in_graph_and_eager_modes
    def test_three_channels(self):
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
        self._validate_median_filter2d(tf_img, expt)


if __name__ == "__main__":
    tf.test.main()
