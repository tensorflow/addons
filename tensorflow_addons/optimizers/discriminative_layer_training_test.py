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
                tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
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
        net = tf.keras.Sequential()

        # crop the input shape so the lstm runs faster
        # pretrained need inputshape for weights to be initialized
        net.add(
            tf.keras.layers.Cropping2D(
                cropping=((8, 8), (12, 12)), input_shape=(32, 32, 3)
            )
        )

        # reshape into a timeseries
        net.add(tf.keras.layers.Reshape(target_shape=(16, 8 * 3)))

        # reduce the length of the time series
        net.add(tf.keras.layers.Cropping1D(cropping=(0, 5)))
        # reduce dimensions

        # we are primarily interested in the bidir lstm layer and its behavior
        net.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(4)))

        model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
                net,
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(5, name="head"),
            ]
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


def _get_train_results(model, verbose=False, epochs=10):
    """Run a training loop and return the results for analysis
    model must be compiled first
    """
    tf.random.set_seed(1)
    x = np.ones(shape=(32, 32, 32, 3), dtype=np.float32)
    y = np.zeros(shape=(32, 5), dtype=np.float32)
    y[:, 0] = 1.0

    return model.fit(x, y, epochs=epochs, batch_size=16, verbose=verbose, shuffle=False)


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

    def _assert_training_losses_are_close(self, model, model_lr, epochs=10):
        """easy way to check if two models train in almost the same way
        epochs set to 10 by default to allow momentum methods to pick up momentum and diverge
        if the disc training is not working
        """
        hist = _get_train_results(model, verbose=False, epochs=epochs)
        hist_lr = _get_train_results(model_lr, verbose=False, epochs=epochs)
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

    def _test_equal_0_sub_layer_lr_to_sub_layer_trainable_false(
        self, model_fn, loss, opt
    ):
        """confirm 0 lr_mult for the a specific layer is the same as setting layer to not trainable
        this also confirms that lr_mult propagates into that layer's trainable variables
        this also confirms that lr_mult does not propagate to the rest of the layers unintentionally
        """
        learning_rate = 0.01
        model = model_fn()

        # we use layer 1 instead of 0 bc layer 0 is just an input layer
        model.layers[0].trainable = False
        model.compile(loss=loss, optimizer=opt(learning_rate))

        model_lr = model_fn()
        model_lr.layers[0].lr_mult = 0.0
        d_opt = DiscriminativeLayerOptimizer(
            opt, model_lr, verbose=False, learning_rate=learning_rate
        )
        model_lr.compile(loss=loss, optimizer=d_opt)

        self._assert_training_losses_are_close(model, model_lr)

    def _test_equal_0_layer_lr_to_trainable_false(self, model_fn, loss, opt):
        """confirm 0 lr_mult for the model is the same as model not trainable
        this also confirms that lr_mult on the model level is propagated to all sublayers and their variables
        """
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

        # only two epochs because we expect no training to occur, thus losses shouldn't change anyways
        self._assert_training_losses_are_close(model, model_lr, epochs=2)

    def _test_equal_half_layer_lr_to_half_lr_of_opt(self, model_fn, loss, opt):
        """confirm 0.5 lr_mult for the model is the same as optim with 0.5 lr
        this also confirms that lr_mult on the model level is propagated to all sublayers and their variables
        """

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

    def _test_sub_layers_keep_lr_mult(self, model_fn, loss, opt):
        """confirm that model trains with lower lr on specific layer
        while a different lr_mult is applied everywhere else
        also confirms that sub layers with an lr mult do not get overridden
        """

        learning_rate = 0.01
        model_lr = model_fn()

        # we set model to lrmult 0 and layer one to lrmult 5
        # if layer one is trainable, then the loss should decrease
        model_lr.lr_mult = 0.00
        model_lr.layers[-1].lr_mult = 3

        d_opt = DiscriminativeLayerOptimizer(
            opt, model_lr, verbose=False, learning_rate=learning_rate
        )
        model_lr.compile(loss=loss, optimizer=d_opt)

        loss_values = get_losses(_get_train_results(model_lr, epochs=5))
        self.assertLess(loss_values[-1], loss_values[0])

    def _test_variables_get_assigned(self, model_fn, loss, opt):
        """confirm that variables do get an lr_mult attribute and that they get the correct one
        """
        learning_rate = 0.01
        model_lr = model_fn()

        # set lr mults
        model_lr.layers[0].lr_mult = 0.3
        model_lr.layers[0].layers[-1].lr_mult = 0.1
        model_lr.layers[-1].lr_mult = 0.5

        d_opt = DiscriminativeLayerOptimizer(
            opt, model_lr, verbose=False, learning_rate=learning_rate
        )
        model_lr.compile(loss=loss, optimizer=d_opt)

        # we expect trainable vars at 0.3 to be reduced by the amount at 0.1
        # this tests that the 0.3 lr mult does not override the 0.1 lr mult
        self.assertEqual(
            len(model_lr.layers[0].trainable_variables)
            - len(model_lr.layers[0].layers[-1].trainable_variables),
            len([var for var in model_lr.trainable_variables if var.lr_mult == 0.3]),
        )

        # we expect trainable vars of model with lr_mult 0.1 to equal trainable vars of that layer
        self.assertEqual(
            len(model_lr.layers[0].layers[-1].trainable_variables),
            len([var for var in model_lr.trainable_variables if var.lr_mult == 0.1]),
        )

        # same logic as above
        self.assertEqual(
            len(model_lr.layers[-1].trainable_variables),
            len([var for var in model_lr.trainable_variables if var.lr_mult == 0.5]),
        )

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

    devices = devices

    @test_utils.run_distributed(2)
    @test_utils.run_in_graph_and_eager_modes
    # @run_distributed(devices)
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
    # create devices to avoid cannot create devices error
    # devices = test_utils.create_virtual_devices(2)

    # save models so weights are always the same
    toy_cnn(first_run=True)
    toy_rnn(first_run=True)

    generate_tests(devices=None)
    tf.test.main()
