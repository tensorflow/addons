import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.layers.stochastic_depth import StochasticDepth
from tensorflow_addons.utils import test_utils

_KEEP_SEED = 1111
_DROP_SEED = 2222


@pytest.mark.parametrize("seed", [_KEEP_SEED, _DROP_SEED])
@pytest.mark.parametrize("training", [True, False])
def stochastic_depth_test(seed, training):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    survival_probability = 0.5

    shortcut = np.asarray([[0.2, 0.1, 0.4]]).astype(np.float32)
    residual = np.asarray([[0.2, 0.4, 0.5]]).astype(np.float32)

    if training:
        if seed == _KEEP_SEED:
            # shortcut + residual
            expected_output = np.asarray([[0.4, 0.5, 0.9]]).astype(np.float32)
        elif seed == _DROP_SEED:
            # shortcut
            expected_output = np.asarray([[0.2, 0.1, 0.4]]).astype(np.float32)
    else:
        # shortcut + p_l * residual
        expected_output = np.asarray([[0.3, 0.3, 0.65]]).astype(np.float32)

    test_utils.layer_test(
        StochasticDepth,
        kwargs={"survival_probability": survival_probability},
        input_data=[shortcut, residual],
        expected_output=expected_output,
    )


@pytest.mark.usefixtures("run_with_mixed_precision_policy")
def test_with_mixed_precision_policy():
    policy = tf.keras.mixed_precision.global_policy()

    shortcut = np.asarray([[0.2, 0.1, 0.4]])
    residual = np.asarray([[0.2, 0.4, 0.5]])

    output = StochasticDepth()([shortcut, residual])
    assert output.dtype == policy.compute_dtype

    output = StochasticDepth()([shortcut, residual], training=True)
    assert output.dtype == policy.compute_dtype


def test_serialization():
    stoch_depth = StochasticDepth(survival_probability=0.5)
    serialized_stoch_depth = tf.keras.layers.serialize(stoch_depth)
    new_layer = tf.keras.layers.deserialize(serialized_stoch_depth)
    assert stoch_depth.get_config() == new_layer.get_config()
