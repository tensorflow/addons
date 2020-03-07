import sys

import pytest
import tensorflow as tf
from tensorflow_addons.register import register_all, _get_all_shared_objects


def test_multiple_register():
    register_all()
    register_all()


def test_get_all_shared_objects():
    all_shared_objects = _get_all_shared_objects()
    assert len(all_shared_objects) >= 4

    for file in all_shared_objects:
        tf.load_op_library(file)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
