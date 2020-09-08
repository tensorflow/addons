import pytest

import numpy as np
import tensorflow as tf
from tensorflow_addons.layers.stochastic_depth import StochasticDepth
from tensorflow_addons.utils import test_utils

KEEP_SEED = 1111
DROP_SEED = 2222

@pytest.mark.parametrize("seed", [KEEP_SEED, DROP_SEED])
@pytest.mark.parametrize("training", [True, False])
def stochastic_depth_test(seed, training):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    p_l = 0.5

    shortcut = np.asarray([[0.2, 0.1, 0.4]]).astype(np.float32)
    residual = np.asarray([[0.2, 0.4, 0.5]]).astype(np.float32)

    if training:
        if seed == KEEP_SEED:
            # shortcut + residual
            expected_output = np.asarray([[0.4, 0.5, 0.9]]).astype(np.float32)
        elif seed == DROP_SEED:
            # shortcut
            expected_output = np.asarray([[0.2, 0.1, 0.4]]).astype(np.float32)
    else:
        # shortcut + p_l * residual
        expected_output = np.asarray([[0.3, 0.3, 0.65]]).astype(np.float32)

    test_utils.layer_test(
        StochasticDepth, kwargs  = {"p_l": p_l}, input_data = [shortcut, residual], expected_output = expected_output
    )

def test_serialization():
    stoch_depth = StochasticDepth(
        p_l = 0.5
    )
    serialized_stoch_depth = tf.keras.layers.serialize(stoch_depth)
    new_layer = tf.keras.layers.deserialize(serialized_stoch_depth)
    assert serialized_stoch_depth.get_config() == new_layer.get_config()