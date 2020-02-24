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


def toy_cnn():
    """Consistently create model with same random weights.
    Skip head activation to allow both bce with logits and cce with logits.

    The model returned by this function should have identical weights to all
    other models returned by this function, for the duration of that
    continuous integration run.

    Run this function within a test, but make sure it runs before other tests.

    Model is intended to work with
    x = np.ones(shape = (None, 32, 32, 3), dtype = np.float32)
    y = np.zeros(shape = (None, 5), dtype = np.float32)
    y[:, 0] = 1.
    """

    cnn_model_path = os.path.join(tempfile.gettempdir(), "cnn.h5")

    if not os.path.exists(cnn_model_path):
        bignet = tf.keras.applications.mobilenet_v2.MobileNetV2(
            include_top=False, weights=None, input_shape=(32, 32, 3), pooling="avg"
        )

        # Take the first few layers so we cover BN, Conv, Pooling ops for testing.
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
        # This creates a model with set weights for testing purposes.
        # Most tests will assert equivalency between a model with discriminative training and a model without.
        return tf.keras.models.load_model(cnn_model_path)
    else:
        assert os.path.exists((cnn_model_path)), (
            "Could not find h5 file at path %s " % cnn_model_path
        )
        # Load the variable initialized model from the disk.
        return tf.keras.models.load_model(cnn_model_path)


def toy_rnn():
    """Consistently create model with same random weights.
    Skip head activation to allow both bce with logits and cce with logits.

    The model returned by this function should have identical weights to all
    other models returned by this function, for the duration of that
    continuous integration run.

    Run this function within a test, but make sure it runs before other tests.

    Model is intended to work with
    x = np.ones(shape = (None, 32, 32, 3), dtype = np.float32)
    y = np.zeros(shape = (None, 5), dtype = np.float32)
    y[:, 0] = 1.
    """
    rnn_model_path = os.path.join(tempfile.gettempdir(), "rnn.h5")

    if not os.path.exists(rnn_model_path):

        # Pretend this net is a pretrained lstm of some sort.
        net = tf.keras.Sequential()

        # Crop the input shape so the lstm runs faster.
        # Pretrained need inputshape for weights to be initialized.
        net.add(
            tf.keras.layers.Cropping2D(
                cropping=((8, 8), (12, 12)), input_shape=(32, 32, 3)
            )
        )

        # Reshape into a timeseries.
        net.add(tf.keras.layers.Reshape(target_shape=(16, 8 * 3)))

        # Reduce the length of the time series.
        net.add(tf.keras.layers.Cropping1D(cropping=(0, 5)))

        # We are primarily interested in the bidir lstm layer and its behavior.
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
        # This creates a model with set weights for testing purposes.
        # Most tests will assert equivalency between a model with discriminative training and a model without.
        return tf.keras.models.load_model(rnn_model_path)

    else:
        assert os.path.exists((rnn_model_path)), (
            "Could not find h5 file at path %s " % rnn_model_path
        )
        # Load the variable initialized model from the disk
        return tf.keras.models.load_model(rnn_model_path)


def _get_train_results(model, verbose=False, epochs=10):
    """Run a training loop and return the results for analysis.
    Model must be compiled first.
    Training data sizes reduced.
    """
    tf.random.set_seed(1)
    x = np.ones(shape=(8, 32, 32, 3), dtype=np.float32)
    y = np.zeros(shape=(8, 5), dtype=np.float32)
    y[:, 0] = 1.0

    return model.fit(x, y, epochs=epochs, batch_size=4, verbose=verbose, shuffle=False)


def _zipped_permutes():
    model_fns = [
        # Generally, we want to test that common layers function correctly with discriminative layer training.
        # Dense, conv2d, batch norm, lstm, pooling, should cover the majority of layer types.
        # We also assume that if it works for conv2d, it should work for conv3d by extension.
        # Apply the same extension logic for all layers tested and it should cover maybe 90% of layers in use?
        toy_cnn,
        toy_rnn,
    ]
    losses = [
        # Additional loss types do not need to be tested.
        # This is because losses affect the gradient tape, which is computed before
        # the apply_gradients step. This means that the some gradient value is passed on to each opt
        # and the gradient calculation is unaffected by which optimizer you are using.
        tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    ]
    optimzers = [
        # Additional optimizers can be added for testing.
        # However, testing adam should cover most optimizer behaviours because it uses momentum.
        tf.keras.optimizers.Adam,
    ]
    return list(itertools.product(model_fns, losses, optimzers))


def get_losses(hist):
    return np.array(hist.__dict__["history"]["loss"])


class DiscriminativeLearningTest(tf.test.TestCase):
    def _assert_losses_are_close(self, hist, hist_lr):
        """Higher tolerance for graph and distributed bc unable to run deterministically."""
        if not tf.executing_eagerly() or tf.distribute.has_strategy():
            rtol, atol = 0.05, 1.00
            # print('graph or dist')
        else:
            rtol, atol = 0.01, 0.01

        return self.assertAllClose(
            get_losses(hist), get_losses(hist_lr), rtol=rtol, atol=atol
        )

    def _assert_training_losses_are_close(self, model, model_lr, epochs=10):
        """Easy way to check if two models train in almost the same way.
        Epochs set to 10 by default to allow momentum methods to pick up momentum and diverge,
        if the disc training is not working.
        """
        hist = _get_train_results(model, verbose=False, epochs=epochs)
        hist_lr = _get_train_results(model_lr, verbose=False, epochs=epochs)
        self._assert_losses_are_close(hist, hist_lr)

    @test_utils.run_distributed(2)
    def test_a_initialize_model_weights(self):
        """This test should run first to initialize the model weights.
        There seem to be major issues in initializing model weights on the fly when testing,
        so we initialize them and save them to an h5 file and reload them each time.
        This ensures that when comparing two runs, they start at the same place.
        This is not actually testing anything, so it does not need to run in eager and graph.
        This needs to run distributed or else it will cause the cannot modify virtual devices error."""
        toy_cnn()
        toy_rnn()

    @test_utils.run_in_graph_and_eager_modes
    @test_utils.run_distributed(2)
    def _test_equal_with_no_layer_lr(self, model_fn, loss, opt):
        """Confirm that discriminative learning is almost the same as regular learning."""
        learning_rate = 0.01
        model = model_fn()
        model.compile(loss=loss, optimizer=opt(learning_rate))

        model_lr = model_fn()
        d_opt = DiscriminativeLayerOptimizer(
            opt, model_lr, verbose=False, learning_rate=learning_rate
        )
        model_lr.compile(loss=loss, optimizer=d_opt)

        self._assert_training_losses_are_close(model, model_lr)

    @test_utils.run_in_graph_and_eager_modes
    @test_utils.run_distributed(2)
    def _test_equal_0_sub_layer_lr_to_sub_layer_trainable_false(
        self, model_fn, loss, opt
    ):
        """Confirm 0 lr_mult for the a specific layer is the same as setting layer to not trainable.
        This also confirms that lr_mult propagates into that layer's trainable variables.
        This also confirms that lr_mult does not propagate to the rest of the layers unintentionally.
        """
        learning_rate = 0.01
        model = model_fn()

        # Layers 0 represents the pretrained network
        model.layers[0].trainable = False
        model.compile(loss=loss, optimizer=opt(learning_rate))

        model_lr = model_fn()
        model_lr.layers[0].lr_mult = 0.0
        d_opt = DiscriminativeLayerOptimizer(
            opt, model_lr, verbose=False, learning_rate=learning_rate
        )
        model_lr.compile(loss=loss, optimizer=d_opt)

        self._assert_training_losses_are_close(model, model_lr)

    @test_utils.run_in_graph_and_eager_modes
    @test_utils.run_distributed(2)
    def _test_equal_0_layer_lr_to_trainable_false(self, model_fn, loss, opt):
        """Confirm 0 lr_mult for the model is the same as model not trainable.
        This also confirms that lr_mult on the model level is propagated to all sublayers and their variables.
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

        # Only two epochs because we expect no training to occur, thus losses shouldn't change anyways.
        self._assert_training_losses_are_close(model, model_lr, epochs=2)

    @test_utils.run_in_graph_and_eager_modes
    @test_utils.run_distributed(2)
    def _test_equal_half_layer_lr_to_half_lr_of_opt(self, model_fn, loss, opt):
        """Confirm 0.5 lr_mult for the model is the same as optim with 0.5 lr.
        This also confirms that lr_mult on the model level is propagated to all sublayers and their variables.
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

    @test_utils.run_in_graph_and_eager_modes
    @test_utils.run_distributed(2)
    def _test_sub_layers_keep_lr_mult(self, model_fn, loss, opt):
        """Confirm that model trains with lower lr on specific layer,
        while a different lr_mult is applied everywhere else.
        Also confirms that sub layers with an lr mult do not get overridden.
        """

        learning_rate = 0.01
        model_lr = model_fn()

        # We set model to lrmult 0 and layer one to lrmult 5.
        # If layer one is trainable, then the loss should decrease.
        model_lr.lr_mult = 0.00
        model_lr.layers[-1].lr_mult = 3

        d_opt = DiscriminativeLayerOptimizer(
            opt, model_lr, verbose=False, learning_rate=learning_rate
        )
        model_lr.compile(loss=loss, optimizer=d_opt)

        loss_values = get_losses(_get_train_results(model_lr, epochs=5))
        self.assertLess(loss_values[-1], loss_values[0])

    @test_utils.run_in_graph_and_eager_modes
    @test_utils.run_distributed(2)
    def _test_variables_get_assigned(self, model_fn, loss, opt):
        """Confirm that variables do get an lr_mult attribute and that they get the correct one.
        """
        learning_rate = 0.01
        model_lr = model_fn()

        # set lr mults.
        model_lr.layers[0].lr_mult = 0.3
        model_lr.layers[0].layers[-1].lr_mult = 0.1
        model_lr.layers[-1].lr_mult = 0.5

        d_opt = DiscriminativeLayerOptimizer(
            opt, model_lr, verbose=False, learning_rate=learning_rate
        )
        model_lr.compile(loss=loss, optimizer=d_opt)

        # We expect trainable vars at 0.3 to be reduced by the amount at 0.1.
        # This tests that the 0.3 lr mult does not override the 0.1 lr mult.
        self.assertEqual(
            len(model_lr.layers[0].trainable_variables)
            - len(model_lr.layers[0].layers[-1].trainable_variables),
            len([var for var in model_lr.trainable_variables if var.lr_mult == 0.3]),
        )

        # We expect trainable vars of model with lr_mult 0.1 to equal trainable vars of that layer.
        self.assertEqual(
            len(model_lr.layers[0].layers[-1].trainable_variables),
            len([var for var in model_lr.trainable_variables if var.lr_mult == 0.1]),
        )

        # Same logic as above.
        self.assertEqual(
            len(model_lr.layers[-1].trainable_variables),
            len([var for var in model_lr.trainable_variables if var.lr_mult == 0.5]),
        )

    @test_utils.run_in_graph_and_eager_modes
    @test_utils.run_distributed(2)
    def _test_model_checkpoint(self, model_fn, loss, opt):
        """Confirm that model does save checkpoints and can load them properly"""

        learning_rate = 0.01
        model_lr = model_fn()
        model_lr.layers[0].lr_mult = 0.3
        model_lr.layers[0].layers[-1].lr_mult = 0.1
        model_lr.layers[-1].lr_mult = 0.5

        d_opt = DiscriminativeLayerOptimizer(
            opt, model_lr, verbose=False, learning_rate=learning_rate
        )
        model_lr.compile(loss=loss, optimizer=d_opt)

        x = np.ones(shape=(8, 32, 32, 3), dtype=np.float32)
        y = np.zeros(shape=(8, 5), dtype=np.float32)
        y[:, 0] = 1.0

        filepath = os.path.join(tempfile.gettempdir(), model_fn.__name__ + "_{epoch}")

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=filepath, save_weights_only=True, verbose=1
            )
        ]

        model_lr.fit(
            x,
            y,
            epochs=2,
            batch_size=4,
            verbose=False,
            shuffle=False,
            callbacks=callbacks,
        )

        # If this doesn't error out, then loading and checkpointing should be fine.
        model_lr.load_weights(
            filepath=os.path.join(tempfile.gettempdir(), model_fn.__name__ + "1")
        )

    def _run_tests_in_notebook(self):
        for name, method in DiscriminativeLearningTest.__dict__.items():
            if callable(method) and name[:4] == "test":
                print("running test %s" % name)
                method(self)


def test_wrap(method, **kwargs):
    """Wrap the test method so that it has pre assigned kwargs."""

    def test(self):
        return method(self, **kwargs)

    return test


def generate_tests():
    # Generate tests for each permutation in the zipped permutes.
    # This separates tests for each permuatation of model, optimizer, and loss.
    for name, method in DiscriminativeLearningTest.__dict__.copy().items():
        if callable(method) and name[:5] == "_test":
            for model_fn, loss, opt in _zipped_permutes():

                # Name the test as test_testname_model_loss_optimizer.
                testmethodname = name[1:] + "_%s_%s_%s" % (
                    model_fn.__name__,
                    loss.name,
                    opt.__name__,
                )

                # Create test functions that use kwargs mentioned above.
                testmethod_dist = test_wrap(
                    method=method, model_fn=model_fn, loss=loss, opt=opt,
                )

                # Set class attributes so we get multiple nicely named tests.
                # Also all tests are set to run distributed, so append distributed to the end.
                setattr(
                    DiscriminativeLearningTest,
                    testmethodname + "_distributed",
                    testmethod_dist,
                )


if __name__ == "__main__":
    generate_tests()
    tf.test.main()
