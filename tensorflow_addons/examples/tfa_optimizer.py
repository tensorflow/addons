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

VALIDATION_SAMPLES = 10000


def build_mnist_model():
    """Build a simple dense network for processing MNIST data.

    :return: Keras `Model`
    """
    inputs = tf.keras.Input(shape=(784,), name='digits')
    net = tf.keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
    net = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(net)
    net = tf.keras.layers.Dense(
        10, activation='softmax', name='predictions')(net)

    return tf.keras.Model(inputs=inputs, outputs=net)


def generate_data(num_validation):
    """Download and preprocess the MNIST dataset.

    :num_validaton: Number of samples to use in validation set
    :return: Dictionary of data split into train/test/val
    """
    dataset = {}

    # Load MNIST dataset as NumPy arrays
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess the data
    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    x_test = x_test.reshape(-1, 784).astype('float32') / 255

    # Subset validation set
    dataset['x_train'] = x_train[:-num_validation]
    dataset['y_train'] = y_train[:-num_validation]
    dataset['x_val'] = x_train[-num_validation:]
    dataset['y_val'] = y_train[-num_validation:]

    dataset['x_test'] = x_test
    dataset['y_test'] = y_test

    return dataset


def train_and_eval():
    """Train and evalute simple MNIST model using LazyAdam."""
    data = generate_data(num_validation=VALIDATION_SAMPLES)
    dense_net = build_mnist_model()
    dense_net.compile(
        optimizer=tfa.optimizers.LazyAdam(0.001),
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
    print('Evaluate on test data:')
    results = dense_net.evaluate(
        data['x_test'], data['y_test'], batch_size=128)
    print('Test loss, Test acc:', results)


if __name__ == "__main__":
    train_and_eval()
