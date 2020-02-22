import tensorflow as tf
from tensorflow_addons.layers.equi_conv import EquiConv

from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class EquiConvTest(tf.test.TestCase):
    def testKerasNHWC(self):
        input = tf.ones(shape=[16, 400, 400, 3])
        layer = EquiConv(64, (3, 3), 1, 1, (1, 1), 1, False, "same", "channels_last")
        res = layer(input)
        self.assertAllEqual(tf.shape(input), tf.shape(res))

    def testKerasNCHW(self):
        input = tf.ones(shape=[16, 3, 400, 400])
        layer = EquiConv(64, (3, 3), 1, 1, (1, 1), 1, False, "same", "channels_first")
        res = layer(input)
        self.assertAllEqual(tf.shape(input), tf.shape(res))


if __name__ == "__main__":
    tf.test.main()
