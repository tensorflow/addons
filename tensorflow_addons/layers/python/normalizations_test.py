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
import scipy as scipy
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util as tf_test_util


def create_and_fit_Sequential_model(layer,shape):
    #Helperfunction for quick evaluation
    model=keras.models.Sequential()
    model.add(layer)
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=RMSPropOptimizer(0.01),loss="categorical_crossentropy")
    layer_shape=(10,)+shape
    input_batch=np.random.rand(*layer_shape)
    output_batch=np.random.rand(*(10,1))
    model.fit(x=input_batch,y=output_batch, epochs=1, batch_size=1)
    return model


class normalization_test(test.TestCase):

    def test_weights(self):
        #Check if weights get initialized
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


    def test_groupnorm_flat(self):
        #Check basic usage of groupnorm_flat
        # Testing for 1 == LayerNorm, 16 == GroupNorm, -1 == InstanceNorm
        groups=[-1,16,1]
        shape=(64,)
        for i in groups:
            model=create_and_fit_Sequential_model(GroupNormalization(groups=i),shape)
            self.assertTrue(hasattr(model.layers[0], 'gamma'))
            self.assertTrue(hasattr(model.layers[0], 'beta'))


    def test_layernorm_flat(self):
        # Check basic usage of layernorm
        model=create_and_fit_Sequential_model(LayerNormalization(),(64,))
        self.assertTrue(hasattr(model.layers[0],'gamma'))
        self.assertTrue(hasattr(model.layers[0],'beta'))


    def test_instancenorm_flat(self):
        # Check basic usage of instancenorm
        model=create_and_fit_Sequential_model(InstanceNormalization(),(64,))
        self.assertTrue(hasattr(model.layers[0],'gamma'))
        self.assertTrue(hasattr(model.layers[0],'beta'))


    def test_initializer(self):
        # Check if the initializer for gamma and beta is working correctly

        model=create_and_fit_Sequential_model(GroupNormalization(groups=32,
                                                                 beta_initializer='random_normal',
                                                                 beta_constraint='NonNeg',
                                                                 gamma_initializer='random_normal',
                                                                 gamma_constraint='NonNeg'),
                                              (64,))
        
        weights=np.array(model.layers[0].get_weights())
        negativ=weights[weights<0.0]

        self.assertTrue(len(weights)==0)


    def test_groupnorm_conv(self):
        # Check if Axis is working for CONV nets
        # Testing for 1 == LayerNorm, 5 == GroupNorm, -1 == InstanceNorm
        groups=[-1,5,1]
        for i in groups:
            model = keras.models.Sequential()
            model.add(GroupNormalization(axis=1,groups=i,input_shape=(20,20,3)))
            model.add(keras.layers.Conv2D(5, (1, 1), padding='same'))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(1,activation='softmax'))
            model.compile(optimizer=RMSPropOptimizer(0.01), loss='mse')
            x=np.random.randint(1000,size=(10,20, 20, 3))
            y=np.random.randint(1000,size=(10,1))
            a=model.fit(x=x,y=y,epochs=1)
            self.assertTrue(hasattr(model.layers[0], 'gamma'))


if __name__ == "__main__":
    test.main()
