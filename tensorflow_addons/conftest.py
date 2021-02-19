import numpy as np
import pytest
import tensorflow as tf

import tensorflow_addons as tfa

from tensorflow_addons.utils.test_utils import (  # noqa: F401
    maybe_run_functions_eagerly,
    only_run_functions_eagerly,
    run_custom_and_py_ops,
    run_with_mixed_precision_policy,
    pytest_make_parametrize_id,
    data_format,
    set_seeds,
    pytest_addoption,
    set_global_variables,
    pytest_configure,
    device,
    pytest_generate_tests,
    pytest_collection_modifyitems,
)

# fixtures present in this file will be available
# when running tests and can be referenced with strings
# https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions


@pytest.fixture(autouse=True)
def add_doctest_namespace(doctest_namespace):
    doctest_namespace["np"] = np
    doctest_namespace["tf"] = tf
    doctest_namespace["tfa"] = tfa
