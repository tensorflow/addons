import os
import pytest
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


def get_legacy_sgd(learning_rate):
    if hasattr(tf.keras.optimizers, "legacy"):
        return tf.keras.optimizers.legacy.SGD(learning_rate)
    return tf.keras.optimizers.SGD(learning_rate)


def get_data_and_model(optimizer="moving_avg"):
    x = tf.random.normal([TRAIN_SAMPLES, INPUT_DIM])
    y = tf.random.normal([TRAIN_SAMPLES, NUM_CLASSES])
    moving_avg = MovingAverage(get_legacy_sgd(2.0), average_decay=0.5)
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


def test_loss_scale_optimizer(tmp_path):
    test_model_filepath = str(tmp_path / "test_model.{epoch:02d}.h5")
    moving_avg = MovingAverage(get_legacy_sgd(2.0), average_decay=0.5)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(moving_avg)
    x, y, model = get_data_and_model(optimizer)
    save_freq = "epoch"
    avg_model_ckpt = AverageModelCheckpoint(
        update_weights=False, filepath=test_model_filepath, save_freq=save_freq
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
