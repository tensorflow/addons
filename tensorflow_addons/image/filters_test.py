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


class _Filter2dTest(tf.test.TestCase):
    def setUp(self):
        self._dtypes_to_test = [
            tf.dtypes.uint8, tf.dtypes.int32, tf.dtypes.float16,
            tf.dtypes.float32, tf.dtypes.float64
        ]
        self._image_shapes_to_test = [(3, 3, 1), (3, 3, 3), (1, 3, 3, 1),
                                      (1, 3, 3, 3), (2, 3, 3, 1), (2, 3, 3, 3)]
        super(_Filter2dTest, self).setUp()

    def _tile_image(self, plane, image_shape):
        """Tile a 2-D image `plane` into 3-D or 4-D as per `image_shape`."""
        assert 3 <= len(image_shape) <= 4
        plane = tf.convert_to_tensor(plane)
        plane = tf.expand_dims(plane, -1)
        channels = image_shape[-1]
        image = tf.tile(plane, (1, 1, channels))

        if len(image_shape) == 4:
            batch_size = image_shape[0]
            image = tf.expand_dims(image, 0)
            image = tf.tile(image, (batch_size, 1, 1, 1))

        return image

    def _setup_values(self, image_shape, filter_shape, padding,
                      constant_values, dtype):
        assert 3 <= len(image_shape) <= 4
        height, width = image_shape[-3], image_shape[-2]
        plane = tf.constant([x for x in range(1, height * width + 1)],
                            shape=(height, width),
                            dtype=dtype)
        image = self._tile_image(plane, image_shape=image_shape)

        result = self._filter2d_fn(
            image,
            filter_shape=filter_shape,
            padding=padding,
            constant_values=constant_values)

        return result

    def _verify_values(self, image_shape, filter_shape, padding,
                       constant_values, expected_plane):
        expected_output = self._tile_image(expected_plane, image_shape)
        for dtype in self._dtypes_to_test:
            result = self._setup_values(image_shape, filter_shape, padding,
                                        constant_values, dtype)
            self.assertAllCloseAccordingToType(
                result, tf.dtypes.cast(expected_output, dtype))


@test_utils.run_all_in_graph_and_eager_modes
class MeanFilter2dTest(_Filter2dTest):
    def setUp(self):
        self._filter2d_fn = mean_filter2d
        super(MeanFilter2dTest, self).setUp()

    def test_invalid_image(self):
        msg = "`image` must be 2/3/4D tensor"
        errors = (ValueError, tf.errors.InvalidArgumentError)
        for image_shape in [(1,), (16, 28, 28, 1, 1)]:
            with self.subTest(dim=len(image_shape)):
                with self.assertRaisesRegexp(errors, msg):
                    image = tf.ones(shape=image_shape)
                    self.evaluate(mean_filter2d(image))

    def test_invalid_filter_shape(self):
        msg = ("The `filter_shape` argument must be a tuple of 2 integers.")
        image = tf.ones(shape=(1, 28, 28, 1))

        for filter_shape in [(3, 3, 3), (3, None, 3)]:
            with self.subTest(filter_shape=filter_shape):
                with self.assertRaisesRegexp(ValueError, msg):
                    mean_filter2d(image, filter_shape=filter_shape)

        filter_shape = None
        with self.subTest(filter_shape=filter_shape):
            with self.assertRaisesRegexp(TypeError, msg):
                mean_filter2d(image, filter_shape=filter_shape)                    

    def test_invalid_padding(self):
        msg = ("padding should be one of \"REFLECT\", \"CONSTANT\", "
               "or \"SYMMETRIC\".")
        image = tf.ones(shape=(1, 28, 28, 1))

        with self.assertRaisesRegexp(ValueError, msg):
            mean_filter2d(image, padding="TEST")

    def test_none_channels(self):
        # 3-D image
        fn = mean_filter2d.get_concrete_function(
            tf.TensorSpec(dtype=tf.dtypes.float32, shape=(3, 3, None)))
        fn(tf.ones(shape=(3, 3, 1)))
        fn(tf.ones(shape=(3, 3, 3)))

        # 4-D image
        fn = mean_filter2d.get_concrete_function(
            tf.TensorSpec(dtype=tf.dtypes.float32, shape=(1, 3, 3, None)))
        fn(tf.ones(shape=(1, 3, 3, 1)))
        fn(tf.ones(shape=(1, 3, 3, 3)))

    def test_unknown_shape(self):
        fn = mean_filter2d.get_concrete_function(
            tf.TensorSpec(shape=None, dtype=tf.dtypes.float32),
            padding="CONSTANT",
            constant_values=1.)

        for shape in [(3, 3), (3, 3, 3), (1, 3, 3, 3)]:
            image = tf.ones(shape=shape)
            self.assertAllEqual(self.evaluate(image), self.evaluate(fn(image)))

    def test_reflect_padding_with_3x3_filter(self):
        expected_plane = tf.constant([[33. / 9., 36. / 9., 39. / 9.],
                                      [42. / 9., 45. / 9., 48. / 9.],
                                      [51. / 9., 54. / 9., 57. / 9.]])

        for image_shape in self._image_shapes_to_test:
            self._verify_values(
                image_shape=image_shape,
                filter_shape=(3, 3),
                padding="REFLECT",
                constant_values=0,
                expected_plane=expected_plane)

    def test_reflect_padding_with_4x4_filter(self):
        expected_plane = tf.constant([[80. / 16., 80. / 16., 80. / 16.],
                                      [80. / 16., 80. / 16., 80. / 16.],
                                      [80. / 16., 80. / 16., 80. / 16.]])

        for image_shape in self._image_shapes_to_test:
            self._verify_values(
                image_shape=image_shape,
                filter_shape=(4, 4),
                padding="REFLECT",
                constant_values=0,
                expected_plane=expected_plane)

    def test_constant_padding_with_3x3_filter(self):
        expected_plane = tf.constant([[12. / 9., 21. / 9., 16. / 9.],
                                      [27. / 9., 45. / 9., 33. / 9.],
                                      [24. / 9., 39. / 9., 28. / 9.]])

        for image_shape in self._image_shapes_to_test:
            self._verify_values(
                image_shape=image_shape,
                filter_shape=(3, 3),
                padding="CONSTANT",
                constant_values=0,
                expected_plane=expected_plane)

        expected_plane = tf.constant([[17. / 9., 24. / 9., 21. / 9.],
                                      [30. / 9., 45. / 9., 36. / 9.],
                                      [29. / 9., 42. / 9., 33. / 9.]])

        for image_shape in self._image_shapes_to_test:
            self._verify_values(
                image_shape=image_shape,
                filter_shape=(3, 3),
                padding="CONSTANT",
                constant_values=1,
                expected_plane=expected_plane)

    def test_symmetric_padding_with_3x3_filter(self):
        expected_plane = tf.constant([[21. / 9., 27. / 9., 33. / 9.],
                                      [39. / 9., 45. / 9., 51. / 9.],
                                      [57. / 9., 63. / 9., 69. / 9.]])

        for image_shape in self._image_shapes_to_test:
            self._verify_values(
                image_shape=image_shape,
                filter_shape=(3, 3),
                padding="SYMMETRIC",
                constant_values=0,
                expected_plane=expected_plane)


@test_utils.run_all_in_graph_and_eager_modes
class MedianFilter2dTest(_Filter2dTest):
    def setUp(self):
        self._filter2d_fn = median_filter2d
        super(MedianFilter2dTest, self).setUp()

    def test_invalid_image(self):
        msg = "`image` must be 2/3/4D tensor"
        errors = (ValueError, tf.errors.InvalidArgumentError)
        for image_shape in [(1,), (16, 28, 28, 1, 1)]:
            with self.subTest(dim=len(image_shape)):
                with self.assertRaisesRegexp(errors, msg):
                    image = tf.ones(shape=image_shape)
                    self.evaluate(median_filter2d(image))

    def test_invalid_filter_shape(self):
        msg = ("The `filter_shape` argument must be a tuple of 2 integers.")
        image = tf.ones(shape=(1, 28, 28, 1))

        for filter_shape in [(3, 3, 3), (3, None, 3)]:
            with self.subTest(filter_shape=filter_shape):
                with self.assertRaisesRegexp(ValueError, msg):
                    median_filter2d(image, filter_shape=filter_shape)
                    
        filter_shape = None
        with self.subTest(filter_shape=filter_shape):
            with self.assertRaisesRegexp(TypeError, msg):
                mean_filter2d(image, filter_shape=filter_shape)                                        

    def test_invalid_padding(self):
        msg = ("padding should be one of \"REFLECT\", \"CONSTANT\", "
               "or \"SYMMETRIC\".")
        image = tf.ones(shape=(1, 28, 28, 1))

        with self.assertRaisesRegexp(ValueError, msg):
            median_filter2d(image, padding="TEST")

    def test_none_channels(self):
        # 3-D image
        fn = median_filter2d.get_concrete_function(
            tf.TensorSpec(dtype=tf.dtypes.float32, shape=(3, 3, None)))
        fn(tf.ones(shape=(3, 3, 1)))
        fn(tf.ones(shape=(3, 3, 3)))

        # 4-D image
        fn = median_filter2d.get_concrete_function(
            tf.TensorSpec(dtype=tf.dtypes.float32, shape=(1, 3, 3, None)))
        fn(tf.ones(shape=(1, 3, 3, 1)))
        fn(tf.ones(shape=(1, 3, 3, 3)))

    def test_unknown_shape(self):
        fn = median_filter2d.get_concrete_function(
            tf.TensorSpec(shape=None, dtype=tf.dtypes.float32),
            padding="CONSTANT",
            constant_values=1.)

        for shape in [(3, 3), (3, 3, 3), (1, 3, 3, 3)]:
            image = tf.ones(shape=shape)
            self.assertAllEqual(self.evaluate(image), self.evaluate(fn(image)))

    def test_reflect_padding_with_3x3_filter(self):
        expected_plane = tf.constant([[4, 4, 5], [5, 5, 5], [5, 6, 6]])

        for image_shape in self._image_shapes_to_test:
            self._verify_values(
                image_shape=image_shape,
                filter_shape=(3, 3),
                padding="REFLECT",
                constant_values=0,
                expected_plane=expected_plane)

    def test_reflect_padding_with_4x4_filter(self):
        expected_plane = tf.constant([[5, 5, 5], [5, 5, 5], [5, 5, 5]])

        for image_shape in self._image_shapes_to_test:
            self._verify_values(
                image_shape=image_shape,
                filter_shape=(4, 4),
                padding="REFLECT",
                constant_values=0,
                expected_plane=expected_plane)

    def test_constant_padding_with_3x3_filter(self):
        expected_plane = tf.constant([[0, 2, 0], [2, 5, 3], [0, 5, 0]])

        for image_shape in self._image_shapes_to_test:
            self._verify_values(
                image_shape=image_shape,
                filter_shape=(3, 3),
                padding="CONSTANT",
                constant_values=0,
                expected_plane=expected_plane)

        expected_plane = tf.constant([[1, 2, 1], [2, 5, 3], [1, 5, 1]])

        for image_shape in self._image_shapes_to_test:
            self._verify_values(
                image_shape=image_shape,
                filter_shape=(3, 3),
                padding="CONSTANT",
                constant_values=1,
                expected_plane=expected_plane)

    def test_symmetric_padding_with_3x3_filter(self):
        expected_plane = tf.constant([[2, 3, 3], [4, 5, 6], [7, 7, 8]])

        for image_shape in self._image_shapes_to_test:
            self._verify_values(
                image_shape=image_shape,
                filter_shape=(3, 3),
                padding="SYMMETRIC",
                constant_values=0,
                expected_plane=expected_plane)


if __name__ == "__main__":
    tf.test.main()
