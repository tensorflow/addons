from tensorflow_addons.utils.test_utils import maybe_run_functions_eagerly  # noqa: F401
from tensorflow_addons.utils.test_utils import cpu_and_gpu  # noqa: F401
from tensorflow_addons.utils.test_utils import data_format  # noqa: F401
from tensorflow_addons.utils.test_utils import set_seeds  # noqa: F401
from tensorflow_addons.utils.test_utils import pytest_addoption  # noqa: F401
from tensorflow_addons.utils.test_utils import set_global_variables  # noqa: F401

import numpy as np
import pytest

import tensorflow as tf
import tensorflow_addons as tfa


# fixtures present in this file will be available
# when running tests and can be referenced with strings
# https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["np"] = np
    doctest_namespace["tf"] = tf
    doctest_namespace["tfa"] = tfa
