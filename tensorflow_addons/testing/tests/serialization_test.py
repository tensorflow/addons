import numpy as np
import pytest
import tensorflow as tf

from tensorflow.keras.metrics import MeanAbsoluteError, TrueNegatives, Metric
from tensorflow_addons.testing.serialization import check_metric_serialization


def test_check_metric_serialization_mae():
    check_metric_serialization(MeanAbsoluteError(), (2, 2), (2, 2))
    check_metric_serialization(MeanAbsoluteError(name="hello"), (2, 2), (2, 2))
    check_metric_serialization(MeanAbsoluteError(), (2, 2, 2), (2, 2, 2))
    check_metric_serialization(MeanAbsoluteError(), (2, 2, 2), (2, 2, 2), (2, 2, 1))


def get_random_booleans():
    return np.random.uniform(0, 2, size=(2, 2))


def test_check_metric_serialization_true_negative():
    check_metric_serialization(
        TrueNegatives(0.8),
        np.random.uniform(0, 2, size=(2, 2)).astype(bool),
        np.random.uniform(0, 1, size=(2, 2)).astype(np.float32),
    )


class MyDummyMetric(Metric):
    def __init__(self, stuff, name):
        super().__init__(name)
        self.stuff = stuff

    def update_state(self, y_true, y_pred, sample_weights):
        pass

    def get_config(self):
        return super().get_config()

    def result(self):
        return 3


def test_missing_arg():
    with pytest.raises(KeyError) as exception_info:
        check_metric_serialization(MyDummyMetric("dodo", "dada"), (2,), (2,))

    assert "stuff" in str(exception_info.value)


class MyOtherDummyMetric(Metric):
    def __init__(self, to_add, name=None, dtype=None):
        super().__init__(name, dtype)
        self.to_add = to_add
        self.sum_of_y_pred = self.add_weight(name="my_sum", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weights=None):
        self.sum_of_y_pred.assign_add(tf.math.reduce_sum(y_pred) + self.to_add)

    def get_config(self):
        config = {"to_add": self.to_add + 1}
        config.update(super().get_config())
        return config

    def result(self):
        return self.sum_of_y_pred


def test_wrong_serialization():
    with pytest.raises(AssertionError):
        check_metric_serialization(MyOtherDummyMetric(5), (2,), (2,))
