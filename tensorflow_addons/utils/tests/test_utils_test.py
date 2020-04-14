import random

import numpy as np
import tensorflow as tf


def test_seed_is_set():
    assert random.randint(0, 10000) == 6311
    assert np.random.randint(0, 10000) == 2732
    assert tf.random.uniform([], 0, 10000, dtype=tf.int64).numpy() == 9457
