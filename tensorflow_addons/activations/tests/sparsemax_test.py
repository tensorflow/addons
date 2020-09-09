# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.activations import sparsemax
from tensorflow_addons.utils import test_utils

test_obs = 17


def _np_sparsemax(z):
    z = z - np.mean(z, axis=1)[:, np.newaxis]

    # sort z
    z_sorted = np.sort(z, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, z.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    # use argmax to get the index by row as .nonzero() doesn't
    # take an axis argument. np.argmax return the first index, but the last
    # index is required here, use np.flip to get the last index and
    # `z.shape[axis]` to compensate for np.flip afterwards.
    k_z = z.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(z)
    tau_sum = z_cumsum[np.arange(0, z.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    # calculate p
    return np.maximum(0, z - tau_z)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sparsemax_against_numpy_axis(dtype):
    """check sparsemax kernel against numpy."""
    random = np.random.RandomState(1)

    z = random.uniform(low=-3, high=3, size=(test_obs, 10))

    tf_sparsemax_out = sparsemax(z.astype(dtype), axis=0).numpy()
    np_sparsemax = np.transpose(_np_sparsemax(np.transpose(z))).astype(dtype)

    test_utils.assert_allclose_according_to_type(
        np_sparsemax, tf_sparsemax_out, half_atol=5e-3
    )
    assert np_sparsemax.shape == tf_sparsemax_out.shape


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sparsemax_against_numpy_low_rank(dtype):
    """check sparsemax kernel against numpy."""
    random = np.random.RandomState(1)

    z = random.uniform(low=-3, high=3, size=(10))

    tf_sparsemax_out = sparsemax(z.astype(dtype)).numpy()
    np_sparsemax = np.reshape(_np_sparsemax(np.reshape(z, [1, 10])), [10]).astype(dtype)

    test_utils.assert_allclose_according_to_type(
        np_sparsemax, tf_sparsemax_out, half_atol=5e-3
    )
    assert np_sparsemax.shape == tf_sparsemax_out.shape


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sparsemax_against_numpy(dtype):
    """check sparsemax kernel against numpy."""
    random = np.random.RandomState(1)

    z = random.uniform(low=-3, high=3, size=(test_obs, 10))

    tf_sparsemax_out = sparsemax(z.astype(dtype))
    np_sparsemax = _np_sparsemax(z).astype(dtype)

    test_utils.assert_allclose_according_to_type(np_sparsemax, tf_sparsemax_out)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sparsemax_against_numpy_high_rank(dtype):
    """check sparsemax kernel against numpy."""
    random = np.random.RandomState(1)

    z = random.uniform(low=-3, high=3, size=(test_obs, test_obs, 10))

    tf_sparsemax_out = sparsemax(z.astype(dtype))
    np_sparsemax = np.reshape(
        _np_sparsemax(np.reshape(z, [test_obs * test_obs, 10])),
        [test_obs, test_obs, 10],
    ).astype(dtype)

    test_utils.assert_allclose_according_to_type(np_sparsemax, tf_sparsemax_out)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sparsemax_of_nan(dtype):
    """check sparsemax transfers nan."""
    z_nan = np.asarray(
        [[0, np.nan, 0], [0, np.nan, np.nan], [np.nan, np.nan, np.nan]]
    ).astype(dtype)

    tf_sparsemax_nan = sparsemax(z_nan)
    np.testing.assert_equal(
        np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        ),
        tf_sparsemax_nan,
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sparsemax_of_inf(dtype):
    """check sparsemax is infinity safe."""
    z_neg = np.asarray(
        [[0, -np.inf, 0], [0, -np.inf, -np.inf], [-np.inf, -np.inf, -np.inf]]
    ).astype(dtype)
    z_pos = np.asarray(
        [[0, np.inf, 0], [0, np.inf, np.inf], [np.inf, np.inf, np.inf]]
    ).astype(dtype)
    z_mix = np.asarray(
        [[0, np.inf, 0], [0, np.inf, -np.inf], [-np.inf, np.inf, -np.inf]]
    ).astype(dtype)

    tf_sparsemax_neg = sparsemax(z_neg)
    np.testing.assert_equal(
        np.array([[0.5, 0, 0.5], [1, 0, 0], [np.nan, np.nan, np.nan]]), tf_sparsemax_neg
    )

    tf_sparsemax_pos = sparsemax(z_pos)
    np.testing.assert_equal(
        np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        ),
        tf_sparsemax_pos,
    )

    tf_sparsemax_mix = sparsemax(z_mix)
    np.testing.assert_equal(
        np.array(
            [
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        ),
        tf_sparsemax_mix,
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sparsemax_of_zero(dtype):
    """check sparsemax proposition 1, part 1."""
    z = np.zeros((1, 10))

    tf_sparsemax_out = sparsemax(z.astype(dtype))
    np_sparsemax = np.ones_like(z, dtype=dtype) / z.size

    test_utils.assert_allclose_according_to_type(np_sparsemax, tf_sparsemax_out)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_sparsemax_of_to_inf(dtype):
    """check sparsemax proposition 1, part 2."""
    random = np.random.RandomState(4)

    z = random.uniform(low=-3, high=3, size=(test_obs, 10))

    # assume |A(z)| = 1, as z is continues random
    z_sort_arg = np.argsort(z, axis=1)[:, ::-1]
    z_sort = np.sort(z, axis=-1)[:, ::-1]
    gamma_z = z_sort[:, 0] - z_sort[:, 1]
    epsilon = (0.99 * gamma_z * 1).reshape(-1, 1)

    # construct the expected 1_A(z) array
    p_expected = np.zeros((test_obs, 10), dtype=dtype)
    p_expected[np.arange(0, test_obs), z_sort_arg[:, 0]] = 1

    tf_sparsemax_out = sparsemax(((1 / epsilon) * z).astype(dtype))

    test_utils.assert_allclose_according_to_type(p_expected, tf_sparsemax_out)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_constant_add(dtype):
    """check sparsemax proposition 2."""
    random = np.random.RandomState(5)

    z = random.uniform(low=-3, high=3, size=(test_obs, 10)).astype(dtype)
    c = random.uniform(low=-3, high=3, size=(test_obs, 1)).astype(dtype)

    tf_sparsemax_zpc = sparsemax((z + c))

    tf_sparsemax_z = sparsemax(z)

    test_utils.assert_allclose_according_to_type(
        tf_sparsemax_zpc, tf_sparsemax_z, half_atol=5e-3
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_two_dimentional(dtype):
    """check two dimentation sparsemax case."""
    t = np.linspace(-2, 2, test_obs, dtype=dtype)
    z = np.vstack([t, np.zeros(test_obs, dtype=dtype)]).T

    tf_sparsemax_out = sparsemax(z.astype(dtype)).numpy()

    p0_expected = np.select([t < -1, t <= 1, t > 1], [0, (t + 1) / 2, 1])

    test_utils.assert_allclose_according_to_type(p0_expected, tf_sparsemax_out[:, 0])
    test_utils.assert_allclose_according_to_type(
        1 - p0_expected, tf_sparsemax_out[:, 1]
    )
    assert z.shape == tf_sparsemax_out.shape


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_diffrence(dtype):
    """check sparsemax proposition 4."""
    random = np.random.RandomState(7)

    z = random.uniform(low=-3, high=3, size=(test_obs, 10))
    p = sparsemax(z.astype(dtype)).numpy()

    etol = {np.float32: 1e-6, np.float64: 1e-9}[dtype]

    for val in range(0, test_obs):
        for i in range(0, 10):
            for j in range(0, 10):
                # check condition, the obesite pair will be checked anyway
                if z[val, i] > z[val, j]:
                    continue

                assert 0 <= p[val, j] - p[val, i] <= z[val, j] - z[val, i] + etol


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_permutation(dtype):
    """check sparsemax proposition 3."""
    random = np.random.RandomState(6)

    z = random.uniform(low=-3, high=3, size=(test_obs, 10))
    p = sparsemax(z.astype(dtype)).numpy()

    for i in range(test_obs):
        per = random.permutation(10)

        tf_sparsemax_out = sparsemax(z[i, per].reshape(1, -1).astype(dtype))
        p_expected = p[i, per].reshape(1, -1)

        test_utils.assert_allclose_according_to_type(
            p_expected, tf_sparsemax_out, half_atol=5e-3
        )
        assert p_expected.shape == tf_sparsemax_out.shape


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_gradient_against_estimate(dtype):
    """check sparsemax Rop, against estimated Rop."""
    random = np.random.RandomState(9)

    # sparsemax is not a smooth function so gradient estimation is only
    # possible for float64.
    if dtype != "float64":
        return

    z = random.uniform(low=-1, high=1, size=(test_obs, 10)).astype(dtype)

    (jacob_sym,), (jacob_num,) = tf.test.compute_gradient(
        lambda logits: sparsemax(logits), [z], delta=1e-6
    )
    np.testing.assert_allclose(jacob_sym, jacob_num)
