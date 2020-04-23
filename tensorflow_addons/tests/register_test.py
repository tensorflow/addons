import sys

import pytest
import tensorflow as tf
from tensorflow_addons.register import register_all, _get_all_shared_objects
from tensorflow_addons.utils import resource_loader


def test_multiple_register():
    if resource_loader.SKIP_CUSTOM_OPS:
        pytest.skip(
            "Skipping the test because a custom ops "
            "was being loaded while --skip-custom-ops was set."
        )
    register_all()
    register_all()


def test_get_all_shared_objects():
    if resource_loader.SKIP_CUSTOM_OPS:
        pytest.skip(
            "Skipping the test because a custom ops "
            "was being loaded while --skip-custom-ops was set."
        )
    all_shared_objects = _get_all_shared_objects()
    assert len(all_shared_objects) >= 4

    for file in all_shared_objects:
        tf.load_op_library(file)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
