from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_addons.utils import test_utils
from tensorflow_addons.losses import giou_loss, GIOULoss


@test_utils.run_all_in_graph_and_eager_modes
class GIOULossTest(tf.test.TestCase):
    def test_giou_loss(self):
        pred_data = [[0, 0, 1, 1], [0, 0, 2, 2]]
        true_data = [[0, 0, 1, 1], [0, 0, 1, 2]]
        loss = giou_loss(true_data, pred_data)
        self.assertAllClose(loss, tf.constant([0, 0.5], tf.float64))


if __name__ == '__main__':
    tf.test.main()
