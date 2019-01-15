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
# =============================================================================

from tensorflow_addons.layers.python.normalizations import GroupNorm,LayerNorm,InstanceNorm
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as keras
from tensorflow.python.training.rmsprop import RMSPropOptimizer

from tensorflow.python.platform import test
from tensorflow.python.framework import test_util as tf_test_util


class NormTest(test.TestCase):

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_groupnorm_flat(self):
        # Testing for 1 == LayerNorm, 5 == GroupNorm, -1 == InstanceNorm
        groups=[-1,16,1]
        for i in groups:

            model = keras.models.Sequential()
            model.add(GroupNorm(
                keras.layers.Dense(32), input_shape=(32,),groups=i))

            model.compile(optimizer=RMSPropOptimizer(0.01), loss='mse')
            model.fit(
                    np.random.random((10,32)),
                    np.random.random((10,32)),
                    epochs=1,
                    batch_size=10)
            self.assertTrue(hasattr(model.layers[0], 'gamma'))
            self.assertTrue(hasattr(model.layers[0], 'beta'))

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_groupnorm_conv(self):
        # Testing for 1 == LayerNorm, 5 == GroupNorm, -1 == InstanceNorm
        groups=[1,5,-1]
        for i in groups:

            model = keras.models.Sequential()
            model.add(GroupNorm(
                keras.layers.Conv2D(5, (3, 10), padding='same'),
                input_shape=(3,10),groups=i))

            model.compile(optimizer=RMSPropOptimizer(0.01), loss='mse')
            model.fit(
                    np.random.random((10, 3, 10)),
                    np.random.random((10, 3, 10)),
                    epochs=1,
                    batch_size=10)
            self.assertTrue(hasattr(model.layers[0], 'gamma'))
            self.assertTrue(hasattr(model.layers[0], 'beta'))


if __name__ == "__main__":
    test.main()
