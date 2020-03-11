import tensorflow as tf
import pytest


@pytest.fixture(scope="function", params=["eager_mode", "tf_function"])
def maybe_run_functions_eagerly(request):
    if request.param == "eager_mode":
        tf.config.experimental_run_functions_eagerly(True)
    elif request.param == "tf_function":
        tf.config.experimental_run_functions_eagerly(False)

    def finalizer():
        tf.config.experimental_run_functions_eagerly(False)

    request.addfinalizer(finalizer)
