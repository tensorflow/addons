from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.image import build_sprite_image
from tensorflow_addons.utils import test_utils


class SpriteImageCreatorTest(tf.test.TestCase):
    """Test for sprite image creator."""
    def setUp(self):
        np.random.seed(0)

    @test_utils.run_all_in_graph_and_eager_modes
    def test_grayscale_image_no_resize(self):
        # yapf: disable
        # pylint: disable=bad-whitespace
        # pylint: disable=bad-continuation
        images =[[[[0], [1], [1], [1]],
                [[1], [0], [0], [1]],
                [[0], [1], [1], [1]],
                [[1], [0], [0], [0]]],
               [[[1], [1], [0], [1]],
                [[0], [1], [0], [0]],
                [[0], [0], [0], [1]],
                [[1], [1], [1], [0]]],
               [[[0], [1], [1], [1]],
                [[0], [0], [0], [0]],
                [[1], [1], [0], [1]],
                [[1], [0], [1], [0]]]]

        end_result = [[1., 0., 0., 0., 1., 0., 1., 0.],
                      [0., 0., 1., 1., 0., 0., 1., 0.],
                      [1., 1., 1., 0., 1., 1., 0., 0.],
                      [1., 0., 0., 0., 0., 1., 0., 0.],
                      [0., 0., 1., 1., 0., 0., 0., 0.],
                      [0., 0., 1., 1., 0., 0., 0., 0.],
                      [1., 1., 1., 1., 0., 0., 0., 0.],
                      [1., 1., 0., 1., 0., 0., 0., 0.]]

        # yapf: enable
        res = build_sprite_image(images,[4,4])
        sess = self._cached_session()
        result = sess.run(res)
        self.assertAllEqual(end_result,result)


    def test_grayscale_image_shape(self):

        # yapf: disable
        # pylint: disable=bad-whitespace
        # pylint: disable=bad-continuation
        images =[[[[0], [1], [1], [1]],
                [[1], [0], [0], [1]],
                [[0], [1], [1], [1]],
                [[1], [0], [0], [0]]],
                [[[1], [1], [0], [1]],
                [[0], [1], [0], [0]],
                [[0], [0], [0], [1]],
                [[1], [1], [1], [0]]],
                [[[0], [1], [1], [1]],
                [[0], [0], [0], [0]],
                [[1], [1], [0], [1]],
                [[1], [0], [1], [0]]]]

        end_result_shape = (4, 4)
        # yapf: enable
        res = build_sprite_image(images, [2, 2])
        sess = self._cached_session()
        result = sess.run(res)
        result = result.shape
        self.assertAllEqual(end_result_shape, result)

    def test_multiple_images_shape(self):
        images = np.random.randn(100, 20, 20, 3)
        res = build_sprite_image(images, [10, 10])
        sess = self._cached_session()
        result = sess.run(res)
        result = result.shape
        self.assertAllEqual(images.shape, result)

    def test_size_shape(self):
        """With a size with 4 dimension"""
        images = np.random.randn(4, 4, 4, 3)
        size = [2, 2, 2, 3]
        with self.assertRaisesRegex(ValueError,
                                    '\'size\' must be a list of size 2, [height,width].'):
            _ = build_sprite_image(images, size)

    def test_size_type(self):
        """Shape has a non-integer value"""
        images = np.random.randn(4, 4, 4, 3)
        size = [0.5, 2]
        with self.assertRaisesRegex(TypeError,
                                    '\'size\' must be a list of two integers.'):
            _ = build_sprite_image(images, size)

    def test_image_dimensions(self):
        """With a faulty image dimension i.e not equal to 4-D"""
        images = np.random.rand(10, 10, 10, 10, 10)
        with self.assertRaisesRegex(ValueError, '\'image\' must have 4 dimensions.'):
            _ = build_sprite_image(images)


if __name__ == "__main__":
    tf.test.main()
