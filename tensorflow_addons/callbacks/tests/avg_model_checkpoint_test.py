import os
import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.callbacks import AverageModelCheckpoint
from tensorflow_addons.optimizers import MovingAverage

TRAIN_SAMPLES = 10
NUM_CLASSES = 2
INPUT_DIM = 3
NUM_HIDDEN = 5
BATCH_SIZE = 5
EPOCHS = 5


def get_data_and_model(optimizer="moving_avg"):
    x = tf.random.normal([TRAIN_SAMPLES, INPUT_DIM])
    y = tf.random.normal([TRAIN_SAMPLES, NUM_CLASSES])
    moving_avg = MovingAverage(
        tf.keras.optimizers.SGD(lr=2.0), sequential_update=True, average_decay=0.5
    )
    if optimizer == "moving_avg":
        optimizer = moving_avg
    inputs = keras.layers.Input(INPUT_DIM)
    hidden_layer = keras.layers.Dense(
        NUM_HIDDEN, input_dim=INPUT_DIM, activation="relu"
    )(inputs)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(hidden_layer)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])
    return x, y, model


def test_compatibility_with_some_opts_only(tmp_path):
    test_model_filepath = str(tmp_path / "test_model.h5")
    x, y, model = get_data_and_model(optimizer="rmsprop")
    avg_model_ckpt = AverageModelCheckpoint(
        update_weights=True, filepath=test_model_filepath
    )
    with pytest.raises(
        TypeError,
        match="AverageModelCheckpoint is only used when trainingwith"
        " MovingAverage or StochasticAverage",
    ):
        model.fit(
            x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[avg_model_ckpt]
        )


def test_model_file_creation(tmp_path):
    test_model_filepath = str(tmp_path / "test_model.h5")
    x, y, model = get_data_and_model()
    avg_model_ckpt = AverageModelCheckpoint(
        update_weights=True, filepath=test_model_filepath
    )
    model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[avg_model_ckpt])
    assert os.path.exists(test_model_filepath)


def test_mode_auto(tmp_path):
    test_model_filepath = str(tmp_path / "test_model.h5")
    x, y, model = get_data_and_model()
    monitor = "val_loss"
    save_best_only = False
    mode = "auto"
    avg_model_ckpt = AverageModelCheckpoint(
        update_weights=True,
        filepath=test_model_filepath,
        monitor=monitor,
        save_best_only=save_best_only,
        mode=mode,
    )
    model.fit(
        x,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x, y),
        callbacks=[avg_model_ckpt],
    )
    assert os.path.exists(test_model_filepath)


def test_mode_min(tmp_path):
    test_model_filepath = str(tmp_path / "test_model.h5")
    x, y, model = get_data_and_model()
    monitor = "val_loss"
    save_best_only = False
    mode = "min"
    avg_model_ckpt = AverageModelCheckpoint(
        update_weights=True,
        filepath=test_model_filepath,
        monitor=monitor,
        save_best_only=save_best_only,
        mode=mode,
    )
    model.fit(
        x,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x, y),
        callbacks=[avg_model_ckpt],
    )
    assert os.path.exists(test_model_filepath)


def test_mode_max(tmp_path):
    test_model_filepath = str(tmp_path / "test_model.h5")
    x, y, model = get_data_and_model()
    mode = "max"
    monitor = "val_acc"
    save_best_only = False
    avg_model_ckpt = AverageModelCheckpoint(
        update_weights=True,
        filepath=test_model_filepath,
        monitor=monitor,
        save_best_only=save_best_only,
        mode=mode,
    )
    model.fit(
        x,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x, y),
        callbacks=[avg_model_ckpt],
    )
    assert os.path.exists(test_model_filepath)


def test_save_best_only(tmp_path):
    test_model_filepath = str(tmp_path / "test_model.h5")
    x, y, model = get_data_and_model()
    save_best_only = True
    avg_model_ckpt = AverageModelCheckpoint(
        update_weights=True, filepath=test_model_filepath, save_best_only=save_best_only
    )
    model.fit(
        x,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x, y),
        callbacks=[avg_model_ckpt],
    )
    assert os.path.exists(test_model_filepath)


def test_metric_unavailable(tmp_path):
    test_model_filepath = str(tmp_path / "test_model.h5")
    x, y, model = get_data_and_model()
    monitor = "unknown"
    avg_model_ckpt = AverageModelCheckpoint(
        update_weights=False,
        filepath=test_model_filepath,
        monitor=monitor,
        save_best_only=True,
    )
    model.fit(
        x,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x, y),
        callbacks=[avg_model_ckpt],
    )
    assert not os.path.exists(test_model_filepath)


def test_save_freq(tmp_path):
    test_filepath = str(tmp_path / "test_model.{epoch:02d}.h5")
    x, y, model = get_data_and_model()
    save_freq = "epoch"
    avg_model_ckpt = AverageModelCheckpoint(
        update_weights=False, filepath=test_filepath, save_freq=save_freq
    )
    model.fit(
        x,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x, y),
        callbacks=[avg_model_ckpt],
    )
    assert os.path.exists(test_filepath.format(epoch=1))
    assert os.path.exists(test_filepath.format(epoch=2))
    assert os.path.exists(test_filepath.format(epoch=3))
    assert os.path.exists(test_filepath.format(epoch=4))
    assert os.path.exists(test_filepath.format(epoch=5))


def test_invalid_save_freq(tmp_path):
    test_model_filepath = str(tmp_path / "test_model.h5")
    save_freq = "invalid_save_freq"
    with pytest.raises(ValueError, match="Unrecognized save_freq"):
        AverageModelCheckpoint(
            update_weights=True, filepath=test_model_filepath, save_freq=save_freq
        )


def _get_dummy_resource_for_checkpoint_testing(tmp_path):
    def get_input_datasets():
        # Simple training input.
        train_input = [[1.0]] * 16
        train_label = [[0.0]] * 16
        ds = tf.data.Dataset.from_tensor_slices((train_input, train_label))
        return ds.batch(8, drop_remainder=True)

    # Very simple model to eliminate randomness.
    optimizer = MovingAverage(
        tf.keras.optimizers.SGD(lr=2.0), sequential_update=True, average_decay=0.5
    )
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=1, input_shape=(1,)))
    model.compile(loss="mae", optimizer=optimizer, metrics=["mae"])
    train_ds = get_input_datasets()
    filepath = str(tmp_path / "test_model.{epoch:02d}.h5")
    callback = AverageModelCheckpoint(
        update_weights=False, filepath=filepath, save_weights_only=True
    )
    return model, train_ds, callback, filepath


def _run_load_weights_on_restart_test_common_iterations():
    (model, train_ds, callback, filepath) = _get_dummy_resource_for_checkpoint_testing()
    initial_epochs = 3
    model.fit(train_ds, epochs=initial_epochs, callbacks=[callback])
    # The files should exist after fitting with callback.
    for epoch in range(initial_epochs):
        assert os.path.exists(filepath.format(epoch=epoch + 1))
    model.fit(train_ds, epochs=1)
    weights_after_one_more_epoch = model.get_weights()
    # The filepath should continue to exist after fitting without callback.
    for epoch in range(initial_epochs):
        assert os.path.exists(filepath.format(epoch=epoch + 1))
    return model, train_ds, filepath, weights_after_one_more_epoch


def test_checkpoint_load_weights():
    (
        model,
        train_ds,
        filepath,
        weights_after_one_more_epoch,
    ) = _run_load_weights_on_restart_test_common_iterations()
    callback = AverageModelCheckpoint(
        update_weights=False,
        filepath=filepath,
        save_weights_only=True,
        load_weights_on_restart=True,
    )
    model.fit(train_ds, epochs=1, callbacks=[callback])
    weights_after_model_restoring_and_one_more_epoch = model.get_weights()
    model.fit(train_ds, epochs=1, callbacks=[callback])
    weights_with_one_final_extra_epoch = model.get_weights()
    np.testing.assert_almost_equal(
        weights_after_one_more_epoch, weights_after_model_restoring_and_one_more_epoch
    )
    np.testing.assert_almost_equal(
        weights_after_one_more_epoch, weights_with_one_final_extra_epoch
    )
