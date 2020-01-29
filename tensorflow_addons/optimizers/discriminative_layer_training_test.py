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
"""Tests for Discriminative Layer Training Manager for TensorFlow."""

# python -m black tensorflow_addons/optimizers/discriminative_layer_training_test.py


import tensorflow as tf
from tensorflow_addons.utils import test_utils
import numpy as np
from tensorflow_addons.optimizers.discriminative_layer_training import DiscriminativeLearning

#:TODO create tests
def toy_cnn():
    '''Consistently create model with same random weights
    skip head activation to allow both bce with logits and cce with logits
    intended to work with
    x = np.ones(shape = (None, 32, 32, 3), dtype = np.float32)
    y = np.zeros(shape = (None, 5), dtype = np.float32)
    y[:, 0] = 1.
    '''

    tf.random.set_seed(1)

    bignet = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False
                                                            , weights=None
                                                            , input_shape=(32, 32, 3)
                                                            , pooling='avg')

    net = tf.keras.models.Model(inputs=bignet.input, outputs=bignet.get_layer('block_2_add').output)

    model = tf.keras.Sequential([net
                                    , tf.keras.layers.GlobalAveragePooling2D()
                                    , tf.keras.layers.Dropout(0.5)
                                    , tf.keras.layers.Dense(5, name='head')])

    return model

def toy_rnn():
    '''Consistently create model with same random weights
    skip head activation to allow both bce with logits and cce with logits
    intended to work with

    x = np.ones(shape = (None, 32, 32, 3), dtype = np.float32)
    y = np.zeros(shape = (None, 5), dtype = np.float32)
    y[:, 0] = 1.
    '''

    tf.random.set_seed(1)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))
    model.add(tf.keras.layers.Reshape(target_shape=(32, 96)))
    model.add(tf.keras.layers.Cropping1D(cropping=(0, 24)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(5))

    return model

def get_train_results(model, loss, opt):
    '''Run a traininng loop and return the results for analysis
    Accepts loss classes and optimizer classes as defined in tf.keras.losses and tf.keras.optimizers
    '''
    tf.random.set_seed(1)

    model.compile(loss=loss(), optimizer=opt)

    x = np.ones(shape=(32, 32, 32, 3), dtype=np.float32)
    y = np.zeros(shape=(32, 5), dtype=np.float32)
    y[:, 0] = 1.

    return model.fit(x, y, epochs=10, batch_size=16, verbose=0)


def opt_list():
    return [tf.keras.optimizers.Adam, tf.keras.optimizers.SGD]

def loss_list():
    return [tf.keras.losses.BinaryCrossentropy, tf.keras.losses.CategoricalCrossentropy, tf.keras.losses.MSE]

@test_utils.run_all_in_graph_and_eager_modes
class DiscriminativeLearningTest(tf.test.TestCase):

    def test_same_results_when_no_lr_mult_specified(self):

        model_fns = [toy_cnn, toy_rnn]

        for model_fn in model_fns:
            for loss in loss_list():
                for opt in opt_list():
                    model = model_fn()
                    hist = get_train_results(model, loss, opt)

                    model_lr = model_fn()
                    DiscriminativeLearning(model_lr)
                    hist_lr = get_train_results(model_lr, loss, opt)

                    print(hist)
                    print(hist_lr)
                    break
                break
            break

        return


if __name__ == '__main__':
    d = DiscriminativeLearningTest()
    d.test_same_results_when_no_lr_mult_specified()

    pass




