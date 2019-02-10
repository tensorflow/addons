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

from tensorflow_addons.layers.python.normalizations import GroupNormalization,LayerNormalization,InstanceNormalization
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as keras
from tensorflow.python.training.rmsprop import RMSPropOptimizer

from tensorflow.python.platform import test
from tensorflow.python.framework import test_util as tf_test_util


def create_and_fit_Sequential_model(layer):
    model=keras.models.Sequential()
    model.add(layer)
    model.add(keras.layers.Dense(32))

    model.compile(optimizer=RMSPropOptimizer(0.01),loss="mse")
    layer_shape=(10,)+layer.input_shape[1:]
    print(type(layer_shape))
    input_batch=np.random.random_sample(size=layer_shape)
    model.fit(input_batch,
              epochs=1,
              batch_size=5)
    return model
class normalization_test(test.TestCase):

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_weights(self):
        layer = GroupNormalization(groups=1,scale=False, center=False)
        layer.build((None, 3, 4))
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.weights), 0)

        layer = LayerNormalization()
        layer.build((None, 3, 4))
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.weights), 2)

        layer = InstanceNormalization()
        layer.build((None, 3, 4))
        self.assertEqual(len(layer.trainable_weights),2)
        self.assertEqual(len(layer.weights),2)

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_groupnorm_flat(self):
        # Testing for 1 == LayerNorm, 16 == GroupNorm, -1 == InstanceNorm
        groups=[-1,16,1]
        for i in groups:

            model=create_and_fit_Sequential_model(GroupNormalization(input_shape=(64,),groups=i))
            self.assertTrue(hasattr(model.layers[0], 'gamma'))
            self.assertTrue(hasattr(model.layers[0], 'beta'))

    @tf_test_util.run_all_in_graph_and_eager_modes
    def test_groupnorm_conv(self):
        # Testing for 1 == LayerNorm, 5 == GroupNorm, -1 == InstanceNorm
        #groups=[1,5,-1]
        groups=[1]
        for i in groups:

            model = keras.models.Sequential()
            model.add(GroupNormalization(
                 input_shape=(20,20,3,),groups=i))

            model.add(keras.layers.Conv2D(5, (1, 1), padding='same'))

            model.compile(optimizer=RMSPropOptimizer(0.01), loss='mse')
            model.fit(np.random.random((10,20, 20, 3)))
            self.assertTrue(hasattr(model.layers[0], 'gamma'))


if __name__ == "__main__":
    test.main()
