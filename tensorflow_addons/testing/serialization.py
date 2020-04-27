from typing import Union
import inspect

import numpy as np
from tensorflow.keras.metrics import Metric
import typeguard


@typeguard.typechecked
def check_metric_serialization(
    metric: Metric,
    y_true: Union[tuple, np.ndarray],
    y_pred: Union[tuple, np.ndarray],
    sample_weight: Union[tuple, np.ndarray, None] = None,
    strict: bool = True,
):
    config = metric.get_config()
    class_ = metric.__class__

    check_config(config, class_, strict)

    metric_copy = class_(**config)
    metric_copy.set_weights(metric.get_weights())

    if isinstance(y_true, tuple):
        y_true = get_random_array(y_true)
    if isinstance(y_pred, tuple):
        y_pred = get_random_array(y_pred)
    if isinstance(sample_weight, tuple) and sample_weight is not None:
        sample_weight = get_random_array(sample_weight)

    # the behavior should be the same for the original and the copy
    if sample_weight is None:
        metric.update_state(y_true, y_pred)
        metric_copy.update_state(y_true, y_pred)
    else:
        metric.update_state(y_true, y_pred, sample_weight)
        metric_copy.update_state(y_true, y_pred, sample_weight)

    assert_all_arrays_close(metric.get_weights(), metric_copy.get_weights())
    metric_result = metric.result().numpy()
    metric_copy_result = metric_copy.result().numpy()

    try:
        np.testing.assert_allclose(metric_result, metric_copy_result)
    except AssertionError as e:
        raise ValueError(
            "The original and the copy of the metric give different results after "
            "the same `.update_states()` call."
        ) from e


def check_config(config, class_, strict):
    init_signature = inspect.signature(class_.__init__)

    for parameter_name in init_signature.parameters:
        if parameter_name == "self":
            continue
        elif parameter_name == "args" and strict:
            raise KeyError(
                "Please do not use args in the class constructor of {}, "
                "as it hides the real signature "
                "and degrades the user experience. "
                "If you have no alternative to *args, "
                "use `strict=False` in check_metric_serialization.".format(
                    class_.__name__
                )
            )
        elif parameter_name == "kwargs" and strict:
            raise KeyError(
                "Please do not use kwargs in the class constructor of {}, "
                "as it hides the real signature "
                "and degrades the user experience. "
                "If you have no alternative to **kwargs, "
                "use `strict=False` in check_metric_serialization.".format(
                    class_.__name__
                )
            )
        if parameter_name not in config:
            raise KeyError(
                "The constructor parameter {} is not present in the config dict "
                "obtained with `.get_config()` of {}. All parameters should be set to "
                "ensure a perfect copy of the keras object can be obtained when "
                "serialized.".format(parameter_name, class_.__name__)
            )


def assert_all_arrays_close(list1, list2):
    for array1, array2 in zip(list1, list2):
        np.testing.assert_allclose(array1, array2)


def get_random_array(shape):
    return np.random.uniform(size=shape).astype(np.float32)
