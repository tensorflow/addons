"""Test of solarize_ops"""
import tensorflow as tf
import sys
import pytest
from tensorflow_addons.image import solarize_ops
from tensorflow_addons.utils import test_utils
from absl.testing import parameterized


@test_utils.run_all_in_graph_and_eager_modes
class SolarizeOPSTest(tf.test.TestCase, parameterized.TestCase):
    def test_solarize(self):
        if tf.executing_eagerly():
            test_image_file = tf.io.read_file("image/test_data/Yellow_Smiley_Face.png")
            test_image_file = tf.io.decode_image(test_image_file, dtype=tf.uint8)
            threshold = 10
            solarize_img = solarize_ops.solarize(test_image_file, threshold)
            self.assertAllEqual(tf.shape(solarize_img), tf.shape(test_image_file))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
