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
"""Tests for TCN layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_addons.utils import test_utils
from tensorflow_addons.layers import TCN
from tensorflow_addons.layers.tcn import ResidualBlock


@test_utils.run_all_in_graph_and_eager_modes
class TCNTest(tf.test.TestCase):
    def test_tcn(self):
        test_utils.layer_test(TCN, input_shape=(2, 4, 4))

    def test_config_tcn(self):

        # test default config
        tcn = TCN()
        self.assertEqual(tcn.filters, 64)
        self.assertEqual(tcn.kernel_size, 2)
        self.assertEqual(tcn.stacks, 1)
        self.assertEqual(tcn.dilations, [1, 2, 4, 8, 16, 32, 64])
        self.assertEqual(tcn.padding, 'causal')
        self.assertEqual(tcn.use_skip_connections, True)
        self.assertEqual(tcn.dropout_rate, 0.0)
        self.assertEqual(tcn.return_sequences, False)
        self.assertEqual(tcn.activation, 'linear')
        self.assertEqual(tcn.kernel_initializer, 'he_normal')
        self.assertEqual(tcn.use_batch_norm, False)

        # Check save and restore config
        tcn_2 = TCN.from_config(tcn.get_config())
        self.assertEqual(tcn_2.filters, 64)
        self.assertEqual(tcn_2.kernel_size, 2)
        self.assertEqual(tcn_2.stacks, 1)
        self.assertEqual(tcn_2.dilations, [1, 2, 4, 8, 16, 32, 64])
        self.assertEqual(tcn_2.padding, 'causal')
        self.assertEqual(tcn_2.use_skip_connections, True)
        self.assertEqual(tcn_2.dropout_rate, 0.0)
        self.assertEqual(tcn_2.return_sequences, False)
        self.assertEqual(tcn_2.activation, 'linear')
        self.assertEqual(tcn_2.kernel_initializer, 'he_normal')
        self.assertEqual(tcn_2.use_batch_norm, False)

    def test_config_residual_block(self):

        # test default config
        residual_block = ResidualBlock()
        self.assertEqual(residual_block.dilation_rate, 1)
        self.assertEqual(residual_block.filters, 64)
        self.assertEqual(residual_block.kernel_size, 2)
        self.assertEqual(residual_block.padding, 'same')
        self.assertEqual(residual_block.activation, 'relu')
        self.assertEqual(residual_block.dropout_rate, 0.0)
        self.assertEqual(residual_block.kernel_initializer, 'he_normal')
        self.assertEqual(residual_block.last_block, False)
        self.assertEqual(residual_block.use_batch_norm, False)

        # Check save and restore config
        residual_block_2 = ResidualBlock.from_config(
            residual_block.get_config())
        self.assertEqual(residual_block_2.dilation_rate, 1)
        self.assertEqual(residual_block_2.filters, 64)
        self.assertEqual(residual_block_2.kernel_size, 2)
        self.assertEqual(residual_block_2.padding, 'same')
        self.assertEqual(residual_block_2.activation, 'relu')
        self.assertEqual(residual_block_2.dropout_rate, 0.0)
        self.assertEqual(residual_block_2.kernel_initializer, 'he_normal')
        self.assertEqual(residual_block_2.last_block, False)
        self.assertEqual(residual_block_2.use_batch_norm, False)


if __name__ == "__main__":
    tf.test.main()
