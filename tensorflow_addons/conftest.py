from tensorflow_addons.utils.test_utils import (  # noqa: F401
    maybe_run_functions_eagerly,
    data_format,
    set_seeds,
    pytest_addoption,
    set_global_variables,
    pytest_configure,
    _device_placement,
    pytest_generate_tests,
)

# fixtures present in this file will be available
# when running tests and can be referenced with strings
# https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions
