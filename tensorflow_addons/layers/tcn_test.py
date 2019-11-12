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


@test_utils.run_all_in_graph_and_eager_modes
class TCNTest(tf.test.TestCase):
    def test_tcn(self):
        test_utils.layer_test(TCN, input_shape=(2, 4, 4))


if __name__ == "__main__":
    tf.test.main()
