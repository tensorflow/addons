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
"""Tests for GeLU activation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow_addons.layers.gelu import GeLU
from tensorflow_addons.utils import test_utils
from absl.testing import parameterized


@parameterized.parameters([np.float16, np.float32, np.float64])
@test_utils.run_all_in_graph_and_eager_modes
class GELUTest(tf.test.TestCase):
    def random_test(self, dtype):
        x = tf.constant([2.5, 0.02, -0.001], shape=(3,1))
        val = np.array([ 2.4849157e+00, 
                        1.0159566e-02, 
                        -4.9960107e-04], 
                        dtype=dtype).reshape(3,1)
     
        test_utils.layer_test(
            GeLU,
            kwargs={'dtype': dtype},
            input_data=x,
            expected_output=val) 

    def random_test_with_numpy(self, dtype):
        x = np.array([[0.5, 1.2, -0.3]]).astype(dtype)
        val = np.array([[0.345714, 1.0617027, -0.11462909]]).astype(dtype)
     
        test_utils.layer_test(
            GeLU,
            kwargs={'dtype': dtype},
            input_data=x,
            expected_output=val)

if __name__ == '__main__':
    tf.test.main()


