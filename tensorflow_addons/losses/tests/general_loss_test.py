# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for General Loss."""


import pytest
import numpy as np
import tensorflow as tf
from tensorflow_addons.losses import (
    general_loss,
    GeneralLoss,
)


def test_config():
    bce_obj = GeneralLoss(reduction=tf.keras.losses.Reduction.NONE, name="general_loss")
    assert bce_obj.name == "general_loss"
    assert bce_obj.reduction == tf.keras.losses.Reduction.NONE


def precompute_loss_fun_inputs(float_dtype):
    """Precompute a loss and its derivatives for random inputs and parameters.
    Generates a large number of random inputs to the loss, and random
    shape/scale parameters for the loss function at each sample, and
    computes the loss and its derivative with respect to all inputs and
    parameters, returning everything to be used to assert various properties
    in our unit tests.

    Args:
      float_dtype: The float precision to be used (np.float32 or np.float64).

    Returns:
      A tuple containing:
       (the number (int) of samples, and the length of all following arrays,
        A np.array (float_dtype) of losses for each sample,
        A np.array (float_dtype) of residuals of each sample (the loss inputs),
        A np array (float_dtype) of shape parameters of each loss,
        A np.array (float_dtype) of scale parameters of each loss,
        A np.array (float_dtype) of derivatives of each loss wrt each x,
        A np.array (float_dtype) of derivatives of each loss wrt each alpha,
        A np.array (float_dtype) of derivatives of each loss wrt each scale)

    Typical usage example:
    (num_samples, loss, x, alpha, scale, d_x, d_alpha, d_scale)
        = self._precompute_lossfun_inputs(np.float32)
    """
    num_samples = 100000
    # Normally distributed inputs.
    x = float_dtype(np.random.normal(size=num_samples))

    # Uniformly distributed values in (-16, 3), quantized to the nearest 0.1
    # to ensure that we hit the special cases at 0, 2.
    alpha = float_dtype(np.round(np.random.uniform(-16, 3, num_samples) * 10) / 10.0)
    # Push the sampled alphas at the extents of the range to +/- infinity, so
    # that we probe those cases too.
    alpha[alpha == 3.0] = float_dtype(float("inf"))
    alpha[alpha == -16.0] = -float_dtype(float("inf"))

    # Random log-normally distributed values in approx (1e-5, 100000):
    scale = float_dtype(np.exp(np.random.normal(size=num_samples) * 4.0) + 1e-5)

    x, alpha, scale = [tf.convert_to_tensor(z) for z in (x, alpha, scale)]
    with tf.GradientTape(persistent=True) as tape:
        for z in (x, alpha, scale):
            tape.watch(z)
        loss = general_loss(x, x, alpha, scale)
        d_x, d_alpha, d_scale = [
            tape.gradient(tf.reduce_sum(loss), z) for z in (x, alpha, scale)
        ]
    return (num_samples, loss, x, alpha, scale, d_x, d_alpha, d_scale)


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_loss_fun_preserves_dtype(float_dtype):
    """Check the loss's output has the same precision as its input."""
    n = 16
    x = float_dtype(np.random.normal(size=n))
    alpha = float_dtype(np.random.normal(size=n))
    scale = float_dtype(np.exp(np.random.normal(size=n)))
    y = general_loss(x, x, alpha, scale)
    assert y.dtype == float_dtype


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_derivative_is_monotonic_wrt_x(float_dtype):
    # Check that the loss increases monotonically with |x|.
    _, _, x, alpha, _, d_x, _, _ = precompute_loss_fun_inputs(float_dtype)
    # This is just to suppress a warning below.
    d_x = tf.where(tf.math.is_finite(d_x), d_x, tf.zeros_like(d_x))
    mask = np.isfinite(alpha) & (np.abs(d_x) > (300.0 * np.finfo(float_dtype).eps))
    np.testing.assert_array_equal(np.sign(d_x[mask]), np.sign(x[mask]))


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_loss_is_near_zero_at_origin(float_dtype):
    # Check that the loss is near-zero when x is near-zero.
    _, loss, x, _, _, _, _, _ = precompute_loss_fun_inputs(float_dtype)
    assert np.all(np.abs(loss[np.abs(x) < 1e-5]) < 1e-5)


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_loss_is_quadratic_near_origin(float_dtype):
    # Check that the loss is well-approximated by a quadratic bowl when
    # |x| < scale
    _, loss, x, _, scale, _, _, _ = precompute_loss_fun_inputs(float_dtype)
    mask = np.abs(x) < (0.5 * scale)
    loss_quad = 0.5 * np.square((x - x) / scale)
    # print(max(loss_quad[mask]))
    np.testing.assert_allclose(loss_quad[mask], loss[mask], rtol=1e-5, atol=1e-2)


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_loss_is_bounded_when_alpha_is_negative(float_dtype):
    # Assert that loss < (alpha - 2)/alpha when alpha < 0.
    _, loss, _, alpha, _, _, _, _ = precompute_loss_fun_inputs(float_dtype)
    mask = alpha < 0.0
    min_val = np.finfo(float_dtype).min
    alpha_clipped = np.maximum(min_val, alpha[mask])
    assert np.all(loss[mask] <= ((alpha_clipped - 2.0) / alpha_clipped))


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_derivative_is_bounded_when_alpha_is_below_2(float_dtype):
    # Assert that |d_x| < |x|/scale^2 when alpha <= 2.
    _, _, x, alpha, scale, d_x, _, _ = precompute_loss_fun_inputs(float_dtype)
    mask = np.isfinite(alpha) & (alpha <= 2)
    assert np.all(
        (
            np.abs(d_x[mask])
            <= (
                (np.abs(x[mask]) + (300.0 * np.finfo(float_dtype).eps))
                / scale[mask] ** 2
            )
        )
    )


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_derivative_is_bounded_when_alpha_is_below_1(float_dtype):
    # Assert that |d_x| < 1/scale when alpha <= 1.
    _, _, _, alpha, scale, d_x, _, _ = precompute_loss_fun_inputs(float_dtype)
    mask = np.isfinite(alpha) & (alpha <= 1)
    assert np.all(
        (
            np.abs(d_x[mask])
            <= ((1.0 + (300.0 * np.finfo(float_dtype).eps)) / scale[mask])
        )
    )


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_alpha_derivative_is_positive(float_dtype):
    # Assert that d_loss / d_alpha > 0.
    _, _, _, alpha, _, _, d_alpha, _ = precompute_loss_fun_inputs(float_dtype)
    mask = np.isfinite(alpha)
    assert np.all(d_alpha[mask] > (-300.0 * np.finfo(float_dtype).eps))


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_scale_derivative_is_negative(float_dtype):
    # Assert that d_loss / d_scale < 0.
    _, _, _, alpha, _, _, _, d_scale = precompute_loss_fun_inputs(float_dtype)
    mask = np.isfinite(alpha)
    assert np.all(d_scale[mask] < (300.0 * np.finfo(float_dtype).eps))


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_loss_is_scale_invariant(float_dtype):
    # Check that loss(mult * x, alpha, mult * scale) == loss(x, alpha, scale)
    (num_samples, loss, x, alpha, scale, _, _, _) = precompute_loss_fun_inputs(
        float_dtype
    )
    # Random log-normally distributed scalings in ~(0.2, 20)
    mult = float_dtype(np.maximum(0.2, np.exp(np.random.normal(size=num_samples))))
    # Compute the scaled loss.
    loss_scaled = general_loss(mult * x, mult * x, alpha, mult * scale)
    np.testing.assert_allclose(loss, loss_scaled, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_alpha_equals_negative_infinity(float_dtype):
    # Check that alpha == -Infinity reproduces Welsch aka Leclerc loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(-float("inf"))
    scale = float_dtype(1.7)
    # Our loss.
    loss = general_loss(x, x, alpha, scale)
    # Welsch/Leclerc loss.
    loss_true = 1.0 - tf.math.exp(-0.5 * tf.square((x - x) / scale))
    np.testing.assert_allclose(loss, loss_true)


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_alpha_equals_negative_two(float_dtype):
    # Check that alpha == -2 reproduces Geman-McClure loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(-2.0)
    scale = float_dtype(1.7)
    # Our loss.
    loss = general_loss(x, x, alpha, scale)
    # Geman-McClure loss.
    loss_true = (
        2.0 * tf.square((x - x) / scale) / (tf.square((x - x) / scale) + 4.0)
    )  # noqa
    np.testing.assert_allclose(loss, loss_true)


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_alpha_equals_zero(float_dtype):
    # Check that alpha == 0 reproduces Cauchy aka Lorentzian loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(0.0)
    scale = float_dtype(1.7)
    # Our loss.
    loss = general_loss(x, x, alpha, scale)
    # Cauchy/Lorentzian loss.
    loss_true = tf.math.log(0.5 * tf.square((x - x) / scale) + 1.0)
    np.testing.assert_allclose(loss, loss_true)


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_alpha_equals_one(float_dtype):
    # Check that alpha == 1 reproduces Charbonnier aka pseudo-Huber loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(1.0)
    scale = float_dtype(1.7)
    # Our loss.
    loss = general_loss(x, x, alpha, scale)
    # Charbonnier loss.
    loss_true = tf.sqrt(tf.square((x - x) / scale) + 1.0) - 1.0
    # print(max(loss_true))
    np.testing.assert_allclose(loss, loss_true, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_alpha_equals_two(float_dtype):
    # Check that alpha == 2 reproduces L2 loss.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(2.0)
    scale = float_dtype(1.7)
    # Our loss.
    loss = general_loss(x, x, alpha, scale)
    # L2 Loss.
    loss_true = 0.5 * tf.square((x - x) / scale)
    np.testing.assert_allclose(loss, loss_true, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_alpha_equals_four(float_dtype):
    # Check that alpha == 4 reproduces a quartic.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(4.0)
    scale = float_dtype(1.7)
    # Our loss.
    loss = general_loss(x, x, alpha, scale)
    # The true loss.
    loss_true = (
        tf.square(tf.square((x - x) / scale)) / 8.0 + tf.square((x - x) / scale) / 2.0
    )
    np.testing.assert_allclose(loss, loss_true, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_alpha_equals_infinity(float_dtype):
    # Check that alpha == Infinity takes the correct form.
    x = np.arange(-20, 20, 0.1, float_dtype)
    alpha = float_dtype(float("inf"))
    scale = float_dtype(1.7)
    # Our loss.
    loss = general_loss(x, x, alpha, scale)
    # The true loss.
    loss_true = tf.math.exp(0.5 * tf.square((x - x) / scale)) - 1.0
    np.testing.assert_allclose(loss, loss_true, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_approximate_loss_is_accurate(float_dtype):
    # Check that the approximate loss (lossfun() with epsilon=1e-6) reasonably
    # approximates the true loss (lossfun() with epsilon=0.) for a range of
    # values of alpha (skipping alpha=0, where the approximation is poor).
    x = np.arange(-10, 10, 0.1, float_dtype)
    scale = float_dtype(1.7)
    for alpha in [-4, -2, -0.2, -0.01, 0.01, 0.2, 1, 1.99, 2, 2.01, 4]:
        alpha = float_dtype(alpha)
        loss = general_loss(x, x, alpha, scale)
        loss_approx = general_loss(x, x, alpha, scale, approximate=True)
        np.testing.assert_allclose(loss, loss_approx, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_loss_and_gradients_are_finite(float_dtype):
    # Test that the loss and its approximation both give finite losses and
    # derivatives everywhere that they should for a wide range of values.
    for approximate in [False, True]:
        num_samples = 100000
        # Normally distributed inputs.
        x = float_dtype(np.random.normal(size=num_samples))
        # Uniformly distributed values in (-16, 3), quantized to the nearest
        # 0.1 to ensure that we hit the special cases at 0, 2.
        alpha = float_dtype(
            np.round(np.random.uniform(-16, 3, num_samples) * 10) / 10.0
        )
        # Random log-normally distributed values in approx (1e-5, 100000):
        scale = float_dtype(np.exp(np.random.normal(size=num_samples) * 4.0) + 1e-5)
        # Compute the loss and its derivative with respect to all three inputs.
        x, alpha, scale = [tf.convert_to_tensor(z) for z in (x, alpha, scale)]
        with tf.GradientTape(persistent=True) as tape:
            for z in (x, alpha, scale):
                tape.watch(z)
            loss = general_loss(x, x, alpha, scale, approximate=approximate)
            d_x, d_alpha, d_scale = [
                tape.gradient(tf.reduce_sum(loss), z) for z in (x, alpha, scale)
            ]
        for v in [loss, d_x, d_alpha, d_scale]:
            assert np.all(np.isfinite(v))


@pytest.mark.parametrize("float_dtype", [np.float32, np.float64])
def test_gradient_matches_finite_differences(float_dtype):
    # Test that the loss and its approximation both return gradients that are
    # close to the numerical gradient from finite differences, with forward
    # differencing. Returning correct gradients is TensorFlow's job, so this is
    # just an aggressive sanity check in case some implementation detail causes
    # gradients to incorrectly go to zero due to quantization or stop_gradients
    # in some op that is used by the loss.
    for approximate in [False, True]:
        num_samples = 100000
        # Normally distributed inputs.
        x = float_dtype(np.random.normal(size=num_samples))
        # Uniformly distributed values in (-16, 3), quantized to the nearest
        # 0.1 and then shifted by 0.05 so that we avoid the special cases at
        # 0 and 2 where the analytical gradient wont match finite differences.
        alpha = float_dtype(
            np.round(np.random.uniform(-16, 3, num_samples) * 10) / 10.0
        )
        # Random uniformy distributed values in [0.5, 1.5]
        scale = float_dtype(np.random.uniform(0.5, 1.5, num_samples))
        # Compute the loss and its derivative with respect to all three inputs.
        x, alpha, scale = [tf.convert_to_tensor(z) for z in (x, alpha, scale)]
        with tf.GradientTape(persistent=True) as tape:
            for z in (x, alpha, scale):
                tape.watch(z)
            loss = general_loss(x, x, alpha, scale, approximate=approximate)
            d_x, d_alpha, d_scale = [
                tape.gradient(tf.reduce_sum(loss), z) for z in (x, alpha, scale)
            ]

        # Assert that the 95th percentile of errors is <= 1e-2.
        def assert_percentile_close(v1, v2):
            np.testing.assert_array_less(np.percentile(np.abs(v1 - v2), 95), 1e-2)

        step_size = float_dtype(1e-3)
        n_x = (
            general_loss(x + step_size, x + step_size, alpha, scale) - loss
        ) / step_size
        n_alpha = (general_loss(x, x, alpha + step_size, scale) - loss) / step_size
        n_scale = (general_loss(x, x, alpha, scale + step_size) - loss) / step_size
        assert_percentile_close(n_x, d_x)
        assert_percentile_close(n_alpha, d_alpha)
        assert_percentile_close(n_scale, d_scale)
