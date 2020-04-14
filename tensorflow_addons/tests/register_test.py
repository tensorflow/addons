import sys

import pytest
from tensorflow_addons.register import register_all, _get_all_shared_objects
from tensorflow_addons.utils.resource_loader import load_op_library


def test_multiple_register():
    register_all()
    register_all()


def test_get_all_shared_objects():
    all_shared_objects = _get_all_shared_objects()
    assert len(all_shared_objects) >= 4

    for file in all_shared_objects:
        load_op_library(file)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
