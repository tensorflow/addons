import time

import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow_addons.callbacks.time_stopping import TimeStopping


class SleepLayer(tf.keras.layers.Layer):
    def __init__(self, secs):
        self.secs = secs
        super().__init__(dynamic=True)

    def call(self, inputs):
        time.sleep(self.secs)
        return inputs


def get_data_and_model(secs):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    model = Sequential()
    model.add(SleepLayer(secs))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error")

    # In case there is some initialization going on.
    model.fit(X, y, epochs=1, verbose=0)
    return X, y, model


def test_stop_at_the_right_time():
    X, y, model = get_data_and_model(0.1)

    time_stopping = TimeStopping(2, verbose=0)
    history = model.fit(X, y, epochs=30, verbose=0, callbacks=[time_stopping])

    assert len(history.epoch) <= 20


def test_default_value():
    X, y, model = get_data_and_model(0.1)

    time_stopping = TimeStopping()
    history = model.fit(X, y, epochs=15, verbose=0, callbacks=[time_stopping])

    assert len(history.epoch) == 15


@pytest.mark.parametrize("verbose", [0, 1])
def test_time_stopping_verbose(capsys, verbose):
    X, y, model = get_data_and_model(0.25)

    time_stopping = TimeStopping(1, verbose=verbose)

    capsys.readouterr()  # flush the stdout/stderr buffer.
    history = model.fit(X, y, epochs=10, verbose=0, callbacks=[time_stopping])
    fit_stdout = capsys.readouterr().out
    nb_epochs_run = len(history.epoch)
    message = "Timed stopping at epoch " + str(nb_epochs_run)
    if verbose:
        assert message in fit_stdout
    else:
        assert message not in fit_stdout
    assert len(history.epoch) <= 4
