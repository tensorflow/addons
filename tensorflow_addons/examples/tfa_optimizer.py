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
"""MNIST example utilizing an optimizer from TensorFlow Addons."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_addons as tfa


def build_mnist_model():
    """Build a simple dense network for processing MNIST data.

    :return: Keras `Model`
    """
    inputs = tf.keras.Input(shape=(784,), name='digits')
    net = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
    net = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(net)
    net = tf.keras.layers.Dense(
        10, activation='softmax', name='predictions')(net)

    model = tf.keras.Model(inputs=inputs, outputs=net)
    return model


def generate_data():
    """Download and preprocess the MNIST dataset.

    :return: Dictionary of data split into train/test/val
    """
    dataset = {}

    # Load MNIST dataset as NumPy arrays
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    # Reserve 10,000 samples for validation
    dataset['x_val'] = x_train[-10000:]
    dataset['y_val'] = y_train[-10000:]
    dataset['x_train'] = x_train[:-10000]
    dataset['y_train'] = y_train[:-10000]

    dataset['x_test'] = x_test
    dataset['y_test'] = y_test

    return dataset


if __name__ == "__main__":
    data = generate_data()
    dense_net = build_mnist_model()
    dense_net.compile(
        optimizer=tfa.optimizers.LazyAdamOptimizer(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    # Train the network
    history = dense_net.fit(
        data['x_train'],
        data['y_train'],
        batch_size=64,
        epochs=10,
        validation_data=(data['x_val'], data['y_val']))

    # Evaluate the network
    print('\n# Evaluate on test data')
    results = dense_net.evaluate(
        data['x_test'], data['y_test'], batch_size=128)
    print('Test loss, Test acc:', results)
