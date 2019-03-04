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

import numpy as np
import scipy as scipy
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow_addons.layers.python.normalizations import GroupNormalization
from tensorflow_addons.layers.python.normalizations import InstanceNormalization
from tensorflow_addons.layers.python.normalizations import LayerNormalization
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.layers import normalization
from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


def create_and_fit_Sequential_model(layer, shape):
    # Helperfunction for quick evaluation
    model = keras.models.Sequential()
    model.add(layer)
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=RMSPropOptimizer(0.01),
                  loss="categorical_crossentropy")
    layer_shape = (10,) + shape
    input_batch = np.random.rand(*layer_shape)
    output_batch = np.random.rand(*(10, 1))
    model.fit(x=input_batch, y=output_batch, epochs=1, batch_size=1)
    return model


class normalization_test(test.TestCase):

# ------------Tests to ensure proper inheritance. If these suceed you can test for Instance norm and Layernorm by setting Groupnorm groups = -1 or 1
    def test_inheritance(self):
        self.assertTrue(issubclass(LayerNormalization, GroupNormalization))
        self.assertTrue(issubclass(InstanceNormalization, GroupNormalization))
        self.assertTrue(LayerNormalization.build==GroupNormalization.build)
        self.assertTrue(InstanceNormalization.build==GroupNormalization.build)
        self.assertTrue(LayerNormalization.call==GroupNormalization.call)
        self.assertTrue(InstanceNormalization.call==GroupNormalization.call)


    def test_groups_after_init(self):
        layers=InstanceNormalization()
        self.assertTrue(layers.groups==-1)
        layers=LayerNormalization()
        self.assertTrue(layers.groups==1)
# -----------------------------------------------------------------------------------------------------------------------------------------

    def test_reshape(self):
        def run_reshape_test(axis, group, input_shape, expected_shape):

            group_layer=GroupNormalization(groups=group,axis=axis)
            group_layer._set_number_of_groups_for_instance_norm(input_shape)

            inputs=np.ones(input_shape)
            tensor_input_shape=tf.convert_to_tensor(input_shape)
            reshaped_inputs, group_shape=group_layer._reshape_into_groups(inputs,(10,10,10),tensor_input_shape)
            for i in range(len(expected_shape)):
                self.assertEqual(int(group_shape[i]),expected_shape[i])

        input_shape=(10,10,10)
        expected_shape=[10,5,10,2]
        run_reshape_test(2,5,input_shape,expected_shape)

        input_shape=(10,10,10)
        expected_shape=[10,2,5,10]
        run_reshape_test(1,2,input_shape,expected_shape)

        input_shape=(10,10,10)
        expected_shape=[10,10,1,10]
        run_reshape_test(1,-1,input_shape,expected_shape)

        input_shape=(10,10,10)
        expected_shape=[10,1,10,10]
        run_reshape_test(1,1,input_shape,expected_shape)

    @tf_test_util.run_in_graph_and_eager_modes
    def test_weights(self):
        # Check if weights get initialized
        layer = GroupNormalization(groups=1, scale=False, center=False)
        layer.build((None, 3, 4))
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.weights), 0)

        layer = LayerNormalization()
        layer.build((None, 3, 4))
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.weights), 2)

        layer = InstanceNormalization()
        layer.build((None, 3, 4))
        self.assertEqual(len(layer.trainable_weights), 2)
        self.assertEqual(len(layer.weights), 2)
    
    def test_apply_normalization(self):
        
        input_shape = (1,4)
        expected_shape= (1,2,2)
        reshaped_inputs= tf.constant([[[2.0,2.0],[3.0,3.0]]])
        layer=GroupNormalization(groups=2,axis=1,scale=False, center= False)
        normalized_input=layer._apply_normalization(reshaped_inputs, input_shape)
        self.assertTrue(tf.reduce_all(tf.equal(normalized_input,tf.constant([[[0.0,0.0],[0.0,0.0]]]))))


        

    @tf_test_util.run_in_graph_and_eager_modes
    def test_groupnorm_flat(self):
        # Check basic usage of groupnorm_flat
        # Testing for 1 == LayerNorm, 16 == GroupNorm, -1 == InstanceNorm

        groups = [-1, 16, 1]
        shape = (64,)
        for i in groups:
            model = create_and_fit_Sequential_model(
                GroupNormalization(groups=i), shape)
            self.assertTrue(hasattr(model.layers[0], 'gamma'))
            self.assertTrue(hasattr(model.layers[0], 'beta'))


    @tf_test_util.run_in_graph_and_eager_modes
    def test_layernorm_flat(self):
        # Check basic usage of layernorm

        model = create_and_fit_Sequential_model(LayerNormalization(), (64,))
        self.assertTrue(hasattr(model.layers[0], 'gamma'))
        self.assertTrue(hasattr(model.layers[0], 'beta'))


    @tf_test_util.run_in_graph_and_eager_modes
    def test_instancenorm_flat(self):
        # Check basic usage of instancenorm

        model = create_and_fit_Sequential_model(InstanceNormalization(), (64,))
        self.assertTrue(hasattr(model.layers[0], 'gamma'))
        self.assertTrue(hasattr(model.layers[0], 'beta'))


    @tf_test_util.run_in_graph_and_eager_modes
    def test_initializer(self):
        # Check if the initializer for gamma and beta is working correctly

        layer=GroupNormalization(groups=32,
                                 beta_initializer='random_normal',
                                 beta_constraint='NonNeg',
                                 gamma_initializer='random_normal',
                                 gamma_constraint='NonNeg')

        model = create_and_fit_Sequential_model(layer,(64,))

        weights = np.array(model.layers[0].get_weights())
        negativ = weights[weights < 0.0]
        self.assertTrue(len(negativ) == 0)


    @tf_test_util.run_in_graph_and_eager_modes
    def test_regularizations(self):

        layer = GroupNormalization(
                            gamma_regularizer='l1', 
                            beta_regularizer='l1',
                            groups=4,
                            axis=2)
        layer.build((None, 4, 4))
        self.assertEqual(len(layer.losses), 2)
        max_norm = keras.constraints.max_norm
        layer = GroupNormalization(
                    gamma_constraint=max_norm, 
                    beta_constraint=max_norm)
        layer.build((None, 3, 4))
        self.assertEqual(layer.gamma.constraint, max_norm)
        self.assertEqual(layer.beta.constraint, max_norm)
 

    @tf_test_util.run_in_graph_and_eager_modes
    def test_groupnorm_conv(self):
        # Check if Axis is working for CONV nets
        # Testing for 1 == LayerNorm, 5 == GroupNorm, -1 == InstanceNorm

        groups = [-1, 5, 1]
        for i in groups:
            model = keras.models.Sequential()
            model.add(GroupNormalization(
                axis=1, groups=i, input_shape=(20, 20, 3)))
            model.add(keras.layers.Conv2D(5, (1, 1), padding='same'))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(1, activation='softmax'))
            model.compile(optimizer=RMSPropOptimizer(0.01), loss='mse')
            x = np.random.randint(1000, size=(10, 20, 20, 3))
            y = np.random.randint(1000, size=(10, 1))
            a = model.fit(x=x, y=y, epochs=1)
            self.assertTrue(hasattr(model.layers[0], 'gamma'))


if __name__ == "__main__":
    test.main()
