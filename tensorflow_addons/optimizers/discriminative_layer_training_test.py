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

import tensorflow as tf
from tensorflow_addons.utils import test_utils
import numpy as np
from tensorflow_addons.optimizers.discriminative_layer_training import (
    DiscriminativeLearning,
)
import itertools
import os
from tensorflow.python.eager import context


def toy_cnn():
    """Consistently create model with same random weights
    skip head activation to allow both bce with logits and cce with logits

    The model returned by this function should have identical weights to all
    other models returned by this function, for the duration of that
    continuous integration run

    model is intended to work with
    x = np.ones(shape = (None, 32, 32, 3), dtype = np.float32)
    y = np.zeros(shape = (None, 5), dtype = np.float32)
    y[:, 0] = 1.
    """

    cnn_model_path = "cnn.h5"

    if not os.path.exists(cnn_model_path):
        # force eager mode for simple initialization of vars
        with context.eager_mode():
            tf.random.set_seed(1)
            bignet = tf.keras.applications.mobilenet_v2.MobileNetV2(
                include_top=False, weights=None, input_shape=(32, 32, 3), pooling="avg"
            )

            # take the first few layers so we cover BN, Conv, Pooling ops for testing
            net = tf.keras.models.Model(
                inputs=bignet.input, outputs=bignet.get_layer("block_2_add").output
            )
            model = tf.keras.Sequential(
                [
                    net,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(5, name="head"),
                ]
            )
            # always save and never return initialized model from memory
            # it seems you cannot pass variables from a nested eager context to its parent graph context

            model.save(cnn_model_path)

    # load the initialized model from the disk
    return tf.keras.models.load_model(cnn_model_path)


# TODO: get toy_run to work
def toy_rnn():
    """

    Consistently create model with same random weights
    skip head activation to allow both bce with logits and cce with logits
    intended to work with

    x = np.ones(shape = (None, 32, 32, 3), dtype = np.float32)
    y = np.zeros(shape = (None, 5), dtype = np.float32)
    y[:, 0] = 1.
    """

    tf.random.set_seed(1)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(32, 32, 3)))
    model.add(tf.keras.layers.Reshape(target_shape=(32, 96)))
    model.add(tf.keras.layers.Cropping1D(cropping=(0, 24)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(5))

    return model


def get_train_results(model):
    """Run a traininng loop and return the results for analysis
    model must be compiled first
    """
    tf.random.set_seed(1)
    x = np.ones(shape=(32, 32, 32, 3), dtype=np.float32)
    y = np.zeros(shape=(32, 5), dtype=np.float32)
    y[:, 0] = 1.0

    return model.fit(x, y, epochs=10, batch_size=16, verbose=0, shuffle=False)


def zipped_permutes():
    model_fns = [toy_cnn]
    losses = [
        tf.keras.losses.BinaryCrossentropy(from_logits=True),
        tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        tf.keras.losses.MeanSquaredError(),
    ]
    optimzers = [
        tf.keras.optimizers.SGD
        # , tf.keras.optimizers.Adam
    ]
    return list(itertools.product(model_fns, losses, optimzers))


def get_losses(hist):
    return np.array(hist.__dict__["history"]["loss"])


# @test_utils.run_all_distributed
@test_utils.run_all_in_graph_and_eager_modes
class DiscriminativeLearningTest(tf.test.TestCase):
    def _assert_losses_are_close(self, hist, hist_lr):
        """higher tolerance for graph due to non determinism"""
        if tf.executing_eagerly():
            rtol, atol = 1e-6, 1e-6
        else:
            # atol isn't important.
            rtol, atol = 0.05, 1.00
        rtol, atol = 0.01, 0.01
        return self.assertAllClose(
            get_losses(hist), get_losses(hist_lr), rtol=rtol, atol=atol
        )

    def _assert_training_losses_are_close(self, model, model_lr):
        hist = get_train_results(model)
        hist_lr = get_train_results(model_lr)
        self._assert_losses_are_close(hist, hist_lr)

    def _test_equal_with_no_layer_lr(self, model_fn, loss, opt):
        model = model_fn()
        model.compile(loss=loss, optimizer=opt())

        model_lr = model_fn()
        model_lr.compile(loss=loss, optimizer=opt())
        DiscriminativeLearning(model_lr)

        self._assert_training_losses_are_close(model, model_lr)

    def _test_equal_0_layer_lr_to_trainable_false(self, model_fn, loss, opt):
        model = model_fn()
        model.trainable = False
        model.compile(loss=loss, optimizer=opt())

        model_lr = model_fn()
        model_lr.lr_mult = 0.
        model_lr.compile(loss=loss, optimizer=opt())
        DiscriminativeLearning(model_lr)

        self._assert_training_losses_are_close(model, model_lr)

    def _test_equal_layer_lr_to_opt_lr(self, model_fn, loss, opt):
        lr = 0.001
        model = model_fn()
        model.compile(loss=loss, optimizer=opt(learning_rate=lr * 0.5))

        model_lr = model_fn()
        model_lr.lr_mult = 0.5
        model_lr.compile(loss=loss, optimizer=opt(learning_rate=lr))
        DiscriminativeLearning(model_lr)

        self._assert_training_losses_are_close(model, model_lr)

    def _run_tests_in_notebook(self):
        for name, method in DiscriminativeLearningTest.__dict__.items():
            if callable(method) and name[:4] == "test":
                print("running test %s" % name)
                method(self)


def test_wrap(method, **kwargs):
    #     @test_utils.run_in_graph_and_eager_modes
    def test(self):
        return method(self, **kwargs)

    return test


def generate_tests():
    for name, method in DiscriminativeLearningTest.__dict__.copy().items():
        if callable(method) and name[:5] == "_test":
            for model_fn, loss, opt in zipped_permutes()[:2]:
                testmethodname = name[1:] + "_%s_%s_%s" % (
                    model_fn.__name__,
                    loss.name,
                    opt.__name__,
                )
                testmethod = test_wrap(
                    method=method, model_fn=model_fn, loss=loss, opt=opt
                )
                setattr(DiscriminativeLearningTest, testmethodname, testmethod)


if __name__ == "__main__":
    generate_tests()

    # DiscriminativeLearningTest()._run_tests_in_notebook()
    # print("done")
    tf.test.main()
