# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_addons.layers import Sparsemax
from tensorflow_addons.utils.python import test_utils

test_obs = 17


def _np_sparsemax(z):
    z = z - np.mean(z, axis=1)[:, np.newaxis]

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    # use argmax to get the index by row as .nonzero() doesn't
    # take an axis argument. np.argmax return the first index, but the last
    # index is required here, use np.flip to get the last index and
    # `z.shape[axis]` to compensate for np.flip afterwards.
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    # calculate p
    return np.maximum(0, z - tau_z)


@test_utils.run_all_with_types(['float32', 'float64'])
@test_utils.run_all_in_graph_and_eager_modes
class SparsemaxTest(tf.test.TestCase):
    def test_sparsemax_layer_against_numpy(self, dtype=None):
        """check sparsemax kernel against numpy."""
        random = np.random.RandomState(1)

        z = random.uniform(low=-3, high=3, size=(test_obs, 10)).astype(dtype)

        test_utils.layer_test(
            Sparsemax,
            input_data=z,
            expected_output=_np_sparsemax(z).astype(dtype))


if __name__ == '__main__':
    tf.test.main()
