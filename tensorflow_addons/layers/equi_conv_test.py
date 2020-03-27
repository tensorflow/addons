import tensorflow as tf
import pytest
from tensorflow_addons.layers.equi_conv import EquiConv


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
class EquiConvTest(tf.test.TestCase):
    def testKerasNHWC(self):
        channel = 32
        input = tf.ones(shape=[1, 10, 10, channel])
        layer = EquiConv(
            channel, (3, 3), 1, 1, (1, 1), 1, False, "same", "channels_last"
        )
        res = layer(input)
        self.assertAllEqual(tf.shape(input), tf.shape(res))

    def testKerasNCHW(self):
        channel = 32
        input = tf.ones(shape=[1, channel, 10, 10])
        layer = EquiConv(
            channel, (3, 3), 1, 1, (1, 1), 1, False, "same", "channels_first"
        )
        res = layer(input)
        self.assertAllEqual(tf.shape(input), tf.shape(res))


if __name__ == "__main__":
    tf.test.main()
