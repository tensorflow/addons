# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for NetVLAD layer."""

from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v2 as tf

from tensorflow_addons.layers.netvlad import NetVLAD
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class NetVLADTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for NetVLAD."""

  @parameterized.parameters(
      {"num_clusters": 1},
      {"num_clusters": 4},
  )
  def test_simple(self, num_clusters):
    test_utils.layer_test(
        NetVLAD,
        kwargs={"num_clusters": num_clusters},
        input_shape=(5, 4, 100))

  def test_unknown(self):
    inputs = np.random.random((5, 4, 100)).astype("float32")
    test_utils.layer_test(
        NetVLAD,
        kwargs={"num_clusters": 3},
        input_shape=(None, None, 100),
        input_data=inputs,
    )

  def test_invalid_shape(self):
    with self.assertRaisesRegexp(
        ValueError, r"`num_clusters` must be greater than 1"):
      test_utils.layer_test(
          NetVLAD, kwargs={"num_clusters": 0}, input_shape=(5, 4, 20))

    with self.assertRaisesRegexp(
        ValueError, r"must have rank 3"):
      test_utils.layer_test(
          NetVLAD, kwargs={"num_clusters": 2}, input_shape=(5, 4, 4, 20))


if __name__ == "__main__":
  tf.test.main()
