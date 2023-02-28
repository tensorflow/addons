import re

import numpy as np
import pytest
import tensorflow as tf

import tensorflow_addons as tfa


def get_data_and_model():
    x = np.random.random((12, 1))
    y = np.random.randint(0, 2, (12, 1), dtype=np.int32)

    inputs = tf.keras.layers.Input(shape=(1,))
    outputs = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="sgd", loss="mse", metrics=["acc"])
    return x, y, model


def test_tqdm_progress_bar(capsys):

    x, y, model = get_data_and_model()

    capsys.readouterr()  # flush the buffer
    model.fit(x, y, epochs=1, verbose=0, callbacks=[tfa.callbacks.TQDMProgressBar()])
    fit_stderr = capsys.readouterr().err
    assert "loss:" in fit_stderr
    assert "acc:" in fit_stderr


def test_tqdm_progress_bar_overall_bar_format(capsys):

    x, y, model = get_data_and_model()
    overall_bar_format = (
        "{l_bar}{bar} {n_fmt}/{total_fmt} ETA: dodo"
        "{remaining}s,  {rate_fmt}{postfix}"
    )
    pb = tfa.callbacks.TQDMProgressBar(
        overall_bar_format=overall_bar_format, show_epoch_progress=False
    )
    capsys.readouterr()  # flush the buffer
    model.fit(x, y, epochs=1, verbose=0, callbacks=[pb])
    fit_stderr = capsys.readouterr().err
    assert "ETA: dodo" in fit_stderr


def test_tqdm_progress_bar_epoch_bar_format(capsys):

    x, y, model = get_data_and_model()
    epoch_bar_format = "{n_fmt}/{total_fmt}{bar} ETA: dodo {remaining}s - {desc}"
    pb = tfa.callbacks.TQDMProgressBar(
        epoch_bar_format=epoch_bar_format, show_overall_progress=False
    )
    capsys.readouterr()  # flush the buffer
    model.fit(x, y, epochs=1, verbose=0, callbacks=[pb])
    fit_stderr = capsys.readouterr().err
    assert "ETA: dodo" in fit_stderr


def test_tqdm_progress_bar_epoch_bar_format_missing_parameter(capsys):

    x, y, model = get_data_and_model()
    epoch_bar_format = "{n_fmt} {bar} ETA: dodo {remaining}s - {desc}"
    pb = tfa.callbacks.TQDMProgressBar(
        epoch_bar_format=epoch_bar_format, show_overall_progress=False
    )
    capsys.readouterr()  # flush the buffer
    model.fit(x, y, batch_size=4, epochs=2, verbose=0, callbacks=[pb])
    fit_stderr = capsys.readouterr().err
    assert "/3" not in fit_stderr


def test_tqdm_progress_bar_metrics_format(capsys):

    x, y, model = get_data_and_model()

    pb = tfa.callbacks.TQDMProgressBar(
        metrics_format="{name}: dodo {value:0.6f}", show_overall_progress=False
    )
    capsys.readouterr()  # flush the buffer
    model.fit(x, y, epochs=1, verbose=0, callbacks=[pb])
    fit_stderr = capsys.readouterr().err
    assert "acc: dodo" in fit_stderr
    assert re.search(r"acc: dodo [0-9]\.[0-9][0-9][0-9][0-9][0-9][0-9]", fit_stderr)


@pytest.mark.parametrize("show_epoch_progress", [True, False])
@pytest.mark.parametrize("show_overall_progress", [True, False])
def test_tqdm_progress_bar_show(capsys, show_epoch_progress, show_overall_progress):

    x, y, model = get_data_and_model()

    pb = tfa.callbacks.TQDMProgressBar(
        show_epoch_progress=show_epoch_progress,
        show_overall_progress=show_overall_progress,
    )
    capsys.readouterr()  # flush the buffer
    model.fit(x, y, batch_size=4, epochs=2, verbose=0, callbacks=[pb])
    fit_stderr = capsys.readouterr().err

    assert ("/3" in fit_stderr) is show_epoch_progress
    assert ("epochs/s" in fit_stderr) is show_overall_progress

    if show_epoch_progress and not show_overall_progress:
        assert "size" not in fit_stderr
        assert "batch" not in fit_stderr


def test_tqdm_progress_bar_validation(capsys):
    x, y, model = get_data_and_model()

    pb = tfa.callbacks.TQDMProgressBar()
    capsys.readouterr()  # flush the buffer
    model.fit(x, y, epochs=1, verbose=0, callbacks=[pb], validation_data=(x, y))
    fit_stderr = capsys.readouterr().err
    assert re.search(r"val_loss: [0-9]\.[0-9][0-9][0-9][0-9]", fit_stderr)
    assert re.search(r"val_acc: [0-9]\.[0-9][0-9][0-9][0-9]", fit_stderr)


def test_tqdm_progress_bar_evaluate(capsys):
    x, y, model = get_data_and_model()

    pb = tfa.callbacks.TQDMProgressBar()
    capsys.readouterr()  # flush the buffer
    model.evaluate(x, y, callbacks=[pb], verbose=0)
    evaluate_stderr = capsys.readouterr().err
    assert re.search(r"loss: [0-9]\.[0-9][0-9][0-9][0-9]", evaluate_stderr)
    assert re.search(r"acc: [0-9]\.[0-9][0-9][0-9][0-9]", evaluate_stderr)
