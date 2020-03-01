import unittest
from tensorflow_addons.register import register_all


class AssertRNNCellTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_multiple_register(self):
        register_all()
        register_all()


if __name__ == "__main__":
    unittest.main()
