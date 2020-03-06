import unittest
import tensorflow as tf
from tensorflow_addons.register import register_all, _get_all_shared_objects


class AssertRNNCellTest(unittest.TestCase):

    def test_multiple_register(self):
        register_all()
        register_all()

    def test_get_all_shared_objects(self):
        all_shared_objects = _get_all_shared_objects()
        self.assertTrue(len(all_shared_objects) >= 4)

        for file in all_shared_objects:
            tf.load_op_library(file)


if __name__ == "__main__":
    unittest.main()
