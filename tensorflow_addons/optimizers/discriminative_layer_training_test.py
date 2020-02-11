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
    DiscriminativeLayerOptimizer,
)
import itertools
import os
import tempfile


def toy_cnn(first_run=False):
    """Consistently create model with same random weights
    skip head activation to allow both bce with logits and cce with logits

    The model returned by this function should have identical weights to all
    other models returned by this function, for the duration of that
    continuous integration run

    Run this function before running the tests and set first run to true

    model is intended to work with
    x = np.ones(shape = (None, 32, 32, 3), dtype = np.float32)
    y = np.zeros(shape = (None, 5), dtype = np.float32)
    y[:, 0] = 1.
    """

    cnn_model_path = os.path.join(tempfile.gettempdir(), "cnn.h5")

    if first_run:
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

        model.save(cnn_model_path)
        # this creates a model with set weights for testing purposes
        # most tests will assert equivalency between a model with discriminative training and a model without
        return None
    else:
        assert os.path.exists((cnn_model_path)), (
            "Could not find h5 file at path %s " % cnn_model_path
        )
        # load the variable initialized model from the disk
        return tf.keras.models.load_model(cnn_model_path)


def toy_rnn(first_run=False):
    """
    Consistently create model with same random weights
    skip head activation to allow both bce with logits and cce with logits
    intended to work with

    The model returned by this function should have identical weights to all
    other models returned by this function, for the duration of that
    continuous integration run

    Run this function before running the tests and set first run to true

    x = np.ones(shape = (None, 32, 32, 3), dtype = np.float32)
    y = np.zeros(shape = (None, 5), dtype = np.float32)
    y[:, 0] = 1.
    """
    rnn_model_path = os.path.join(tempfile.gettempdir(), "rnn.h5")

    if first_run:

        # pretend that net is a pretrained lstm of some sort
        net = tf.keras.Sequential(name="pretrained lstm")

        net.add(tf.keras.layers.Input(input_shape=(32, 32, 3)))
        net.add(tf.keras.layers.Reshape(target_shape=(32, 96)))
        # reduce the length of the time series
        net.add(tf.keras.layers.Cropping1D(cropping=(0, 16)))
        # we are primarily interested in the bidir lstm layer and its behavior
        net.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8)))

        model = tf.keras.Sequential(
            [net, tf.keras.layers.Dropout(0.5), tf.keras.layers.Dense(5, name="head"),]
        )

        model.save(rnn_model_path)
        # this creates a model with set weights for testing purposes
        # most tests will assert equivalency between a model with discriminative training and a model without
        return None

    else:
        assert os.path.exists((rnn_model_path)), (
            "Could not find h5 file at path %s " % rnn_model_path
        )
        # load the variable initialized model from the disk
        return tf.keras.models.load_model(rnn_model_path)


def _get_train_results(model, verbose=False):
    """Run a training loop and return the results for analysis
    model must be compiled first
    """
    tf.random.set_seed(1)
    x = np.ones(shape=(32, 32, 32, 3), dtype=np.float32)
    y = np.zeros(shape=(32, 5), dtype=np.float32)
    y[:, 0] = 1.0

    return model.fit(x, y, epochs=10, batch_size=16, verbose=verbose, shuffle=False)


def _zipped_permutes():
    model_fns = [
        # generally, we want to test that common layers function correctly with discriminative layer training
        # dense, conv2d, batch norm, lstm, pooling, should cover the majority of layer types
        # we also assume that if it works for conv2d, it should work for conv3d by extension
        # apply the same extension logic for all layers tested and it should cover maybe 90% of layers in use?
        toy_cnn,
        toy_rnn,
    ]
    losses = [
        # additional loss types do not need to be tested
        # this is because losses affect the gradient tape, which is computed before
        # the apply_gradients step. This means that the some gradient value is passed on to each opt
        # and the gradient calculation is unaffected by which optimizer you are using
        tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    ]
    optimzers = [
        # additional optimizers can be added for testing
        # seems to be timing out. will add SGD back later
        # tf.keras.optimizers.SGD,
        tf.keras.optimizers.Adam,
    ]
    return list(itertools.product(model_fns, losses, optimzers))


def get_losses(hist):
    return np.array(hist.__dict__["history"]["loss"])


class DiscriminativeLearningTest(tf.test.TestCase):
    def _assert_losses_are_close(self, hist, hist_lr):
        """higher tolerance for graph and distributed bc unable to run deterministically"""
        if not tf.executing_eagerly() or tf.distribute.has_strategy():
            rtol, atol = 0.05, 1.00
            # print('graph or dist')
        else:
            rtol, atol = 0.01, 0.01

        return self.assertAllClose(
            get_losses(hist), get_losses(hist_lr), rtol=rtol, atol=atol
        )

    def _assert_training_losses_are_close(self, model, model_lr):
        hist = _get_train_results(model, verbose=False)
        hist_lr = _get_train_results(model_lr, verbose=False)
        self._assert_losses_are_close(hist, hist_lr)

    def _test_equal_with_no_layer_lr(self, model_fn, loss, opt):
        """confirm that discriminative learning is almost the same as regular learning"""
        learning_rate = 0.01
        model = model_fn()
        model.compile(loss=loss, optimizer=opt(learning_rate))

        model_lr = model_fn()
        d_opt = DiscriminativeLayerOptimizer(
            opt, model_lr, verbose=False, learning_rate=learning_rate
        )
        model_lr.compile(loss=loss, optimizer=d_opt)

        self._assert_training_losses_are_close(model, model_lr)

    def _test_equal_0_layer_lr_to_trainable_false(self, model_fn, loss, opt):
        """confirm 0 lr_mult for the model is the same as model not trainable"""
        learning_rate = 0.01
        model = model_fn()
        model.trainable = False
        model.compile(loss=loss, optimizer=opt(learning_rate))

        model_lr = model_fn()
        model_lr.lr_mult = 0.0
        d_opt = DiscriminativeLayerOptimizer(
            opt, model_lr, verbose=False, learning_rate=learning_rate
        )
        model_lr.compile(loss=loss, optimizer=d_opt)

        self._assert_training_losses_are_close(model, model_lr)

    def _test_equal_half_layer_lr_to_half_lr_of_opt(self, model_fn, loss, opt):
        """confirm 0.5 lr_mult for the model is the same as optim with 0.5 lr"""

        mult = 0.5
        learning_rate = 0.01
        model = model_fn()
        model.compile(loss=loss, optimizer=opt(learning_rate * mult))

        model_lr = model_fn()
        model_lr.lr_mult = mult
        d_opt = DiscriminativeLayerOptimizer(
            opt, model_lr, verbose=False, learning_rate=learning_rate
        )
        model_lr.compile(loss=loss, optimizer=d_opt)

        self._assert_training_losses_are_close(model, model_lr)

    def _test_loss_changes_over_time(self, model_fn, loss, opt):
        """confirm that model trains with lower lr on specific layer"""

        learning_rate = 0.01
        model_lr = model_fn()
        model_lr.layers[0].lr_mult = 0.01
        d_opt = DiscriminativeLayerOptimizer(
            opt, model_lr, verbose=False, learning_rate=learning_rate
        )
        model_lr.compile(loss=loss, optimizer=d_opt)

        loss_values = get_losses(_get_train_results(model_lr))
        self.assertLess(loss_values[-1], loss_values[0])

    def _run_tests_in_notebook(self):
        for name, method in DiscriminativeLearningTest.__dict__.items():
            if callable(method) and name[:4] == "test":
                print("running test %s" % name)
                method(self)


def run_distributed(devices):
    def decorator(f):
        def decorated(self, *args, **kwargs):
            logical_devices = devices
            strategy = tf.distribute.MirroredStrategy(logical_devices)
            with strategy.scope():
                f(self, *args, **kwargs)

        return decorated

    return decorator


def test_wrap(method, devices, **kwargs):
    @test_utils.run_in_graph_and_eager_modes
    def single(self):
        return method(self, **kwargs)

    @test_utils.run_in_graph_and_eager_modes
    @run_distributed(devices)
    def distributed(self):
        return method(self, **kwargs)

    return single, distributed


def generate_tests(devices):
    for name, method in DiscriminativeLearningTest.__dict__.copy().items():
        if callable(method) and name[:5] == "_test":
            for model_fn, loss, opt in _zipped_permutes():
                testmethodname = name[1:] + "_%s_%s_%s" % (
                    model_fn.__name__,
                    loss.name,
                    opt.__name__,
                )
                testmethod, testmethod_dist = test_wrap(
                    method=method,
                    devices=devices,
                    model_fn=model_fn,
                    loss=loss,
                    opt=opt,
                )

                #                 setattr(DiscriminativeLearningTest, testmethodname, testmethod)
                setattr(
                    DiscriminativeLearningTest,
                    testmethodname + "_distributed",
                    testmethod_dist,
                )


if __name__ == "__main__":
    # save models so weights are always the same
    toy_cnn(first_run=True)
    toy_rnn(first_run=True)

    devices = test_utils.create_virtual_devices(2)
    generate_tests(devices)
    tf.test.main()
