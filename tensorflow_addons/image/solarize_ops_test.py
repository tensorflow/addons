"""Test of solarize_ops"""

import sys
import pytest
import tensorflow as tf
from tensorflow_addons.image import solarize_ops
from tensorflow_addons.utils import test_utils
from absl.testing import parameterized


@test_utils.run_all_in_graph_and_eager_modes
class SolarizeOPSTest(tf.test.TestCase, parameterized.TestCase):
    """SolarizeOPSTest class to test the solarize images"""

    def test_solarize(self):
        if tf.executing_eagerly():
            image2 = tf.constant(
                [
                    [255, 255, 255, 255],
                    [255, 255, 255, 255],
                    [255, 255, 255, 255],
                    [255, 255, 255, 255],
                ],
                dtype=tf.uint8,
            )
            threshold = 10
            solarize_img = solarize_ops.solarize(image2, threshold)
            self.assertAllEqual(tf.shape(solarize_img), tf.shape(image2))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
