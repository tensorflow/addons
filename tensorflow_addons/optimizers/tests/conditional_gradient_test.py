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
"""Tests for Conditional Gradient."""

import numpy as np
import pytest
import platform

import tensorflow as tf
from tensorflow_addons.utils import test_utils
from tensorflow_addons.optimizers import conditional_gradient as cg_lib


def _dtypes_to_test(use_gpu):
    # Based on issue #347 in the following link,
    #        "https://github.com/tensorflow/addons/issues/347"
    # tf.half is not registered for 'ResourceScatterUpdate' OpKernel
    # for 'GPU' devices.
    # So we have to remove tf.half when testing with gpu.
    # The function "_DtypesToTest" is from
    #       "https://github.com/tensorflow/tensorflow/blob/5d4a6cee737a1dc6c20172a1dc1
    #        5df10def2df72/tensorflow/python/kernel_tests/conv_ops_3d_test.py#L53-L62"
    #
    #  Update cpu to use tf.half once issue in TF2.4 is fixed: https://github.com/tensorflow/tensorflow/issues/45136
    if use_gpu:
        return [tf.float32, tf.float64]
    else:
        return [tf.float32, tf.float64]


def _dtypes_with_checking_system(use_gpu, system):
    # Based on issue #36764 in the following link,
    #        "https://github.com/tensorflow/tensorflow/issues/36764"
    # tf.half is not registered for tf.linalg.svd function on Windows
    # CPU version.
    # So we have to remove tf.half when testing with Windows CPU version.
    if system == "Windows":
        return [tf.float32, tf.float64]
    else:
        return _dtypes_to_test(use_gpu)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_like_dist_belief_nuclear_cg01():
    db_grad, db_out = _db_params_nuclear_cg01()
    num_samples = len(db_grad)
    var0 = tf.Variable([0.0] * num_samples)
    grads0 = tf.constant([0.0] * num_samples)
    ord = "nuclear"
    cg_opt = cg_lib.ConditionalGradient(learning_rate=0.1, lambda_=0.1, ord=ord)

    for i in range(num_samples):
        grads0 = tf.constant(db_grad[i])
        cg_opt.apply_gradients(zip([grads0], [var0]))
        np.testing.assert_allclose(
            np.array(db_out[i]), var0.numpy(), rtol=1e-6, atol=1e-6
        )


@pytest.mark.with_device(["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32, tf.float64])
def test_minimize_sparse_resource_variable_frobenius(dtype, device):
    if "gpu" in device and dtype == tf.float16:
        pytest.xfail("See https://github.com/tensorflow/addons/issues/347")
    var0 = tf.Variable([[1.0, 2.0]], dtype=dtype)

    def loss():
        x = tf.constant([[4.0], [5.0]], dtype=dtype)
        pred = tf.matmul(tf.nn.embedding_lookup([var0], [0]), x)
        return pred * pred

    # the gradient based on the current loss function
    grads0_0 = 32 * 1.0 + 40 * 2.0
    grads0_1 = 40 * 1.0 + 50 * 2.0
    grads0 = tf.constant([[grads0_0, grads0_1]], dtype=dtype)
    norm0 = tf.math.reduce_sum(grads0**2) ** 0.5

    learning_rate = 0.1
    lambda_ = 0.1
    ord = "fro"
    opt = cg_lib.ConditionalGradient(
        learning_rate=learning_rate, lambda_=lambda_, ord=ord
    )
    _ = opt.minimize(loss, var_list=[var0])
    test_utils.assert_allclose_according_to_type(
        [
            [
                1.0 * learning_rate - (1 - learning_rate) * lambda_ * grads0_0 / norm0,
                2.0 * learning_rate - (1 - learning_rate) * lambda_ * grads0_1 / norm0,
            ]
        ],
        var0.numpy(),
    )


@pytest.mark.parametrize("dtype", [(tf.half, 0), (tf.float32, 1), (tf.float64, 2)])
@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("use_resource", [True, False])
def test_basic_frobenius(dtype, use_resource):
    if use_resource:
        var0 = tf.Variable([1.0, 2.0], dtype=dtype[0], name="var0_%d" % dtype[1])
        var1 = tf.Variable([3.0, 4.0], dtype=dtype[0], name="var0_%d" % dtype[1])
    else:
        var0 = tf.Variable([1.0, 2.0], dtype=dtype[0])
        var1 = tf.Variable([3.0, 4.0], dtype=dtype[0])
    grads0 = tf.constant([0.1, 0.1], dtype=dtype[0])
    grads1 = tf.constant([0.01, 0.01], dtype=dtype[0])
    norm0 = tf.math.reduce_sum(grads0**2) ** 0.5
    norm1 = tf.math.reduce_sum(grads1**2) ** 0.5

    def learning_rate():
        return 0.5

    def lambda_():
        return 0.01

    ord = "fro"

    cg_opt = cg_lib.ConditionalGradient(
        learning_rate=learning_rate, lambda_=lambda_, ord=ord
    )
    _ = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

    # Check we have slots
    assert ["conditional_gradient"] == cg_opt.get_slot_names()
    slot0 = cg_opt.get_slot(var0, "conditional_gradient")
    assert slot0.get_shape() == var0.get_shape()
    slot1 = cg_opt.get_slot(var1, "conditional_gradient")
    assert slot1.get_shape() == var1.get_shape()

    test_utils.assert_allclose_according_to_type(
        np.array(
            [
                1.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0,
                2.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0,
            ]
        ),
        var0.numpy(),
    )
    test_utils.assert_allclose_according_to_type(
        np.array(
            [
                3.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1,
                4.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1,
            ]
        ),
        var1.numpy(),
    )

    # Step 2: the conditional_gradient contain the previous update.
    cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    test_utils.assert_allclose_according_to_type(
        np.array(
            [
                (1.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0) * 0.5
                - (1 - 0.5) * 0.01 * 0.1 / norm0,
                (2.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0) * 0.5
                - (1 - 0.5) * 0.01 * 0.1 / norm0,
            ]
        ),
        var0.numpy(),
    )
    test_utils.assert_allclose_according_to_type(
        np.array(
            [
                (3.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1) * 0.5
                - (1 - 0.5) * 0.01 * 0.01 / norm1,
                (4.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1) * 0.5
                - (1 - 0.5) * 0.01 * 0.01 / norm1,
            ]
        ),
        var1.numpy(),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("use_resource", [True, False])
def test_basic_nuclear(use_resource):
    # TODO:
    #       to address issue #36764
    for i, dtype in enumerate(
        _dtypes_with_checking_system(
            use_gpu=test_utils.is_gpu_available(), system=platform.system()
        )
    ):

        if use_resource:
            var0 = tf.Variable([1.0, 2.0], dtype=dtype, name="var0_%d" % i)
            var1 = tf.Variable([3.0, 4.0], dtype=dtype, name="var1_%d" % i)
        else:
            var0 = tf.Variable([1.0, 2.0], dtype=dtype)
            var1 = tf.Variable([3.0, 4.0], dtype=dtype)

        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        top_singular_vector0 = cg_lib.ConditionalGradient._top_singular_vector(grads0)
        top_singular_vector1 = cg_lib.ConditionalGradient._top_singular_vector(grads1)

        def learning_rate():
            return 0.5

        def lambda_():
            return 0.01

        ord = "nuclear"

        cg_opt = cg_lib.ConditionalGradient(
            learning_rate=learning_rate, lambda_=lambda_, ord=ord
        )
        _ = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        # Check we have slots
        assert ["conditional_gradient"] == cg_opt.get_slot_names()
        slot0 = cg_opt.get_slot(var0, "conditional_gradient")
        assert slot0.get_shape() == var0.get_shape()
        slot1 = cg_opt.get_slot(var1, "conditional_gradient")
        assert slot1.get_shape() == var1.get_shape()

        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    1.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[0],
                    2.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[1],
                ]
            ),
            var0.numpy(),
        )
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    3.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[0],
                    4.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[1],
                ]
            ),
            var1.numpy(),
        )

        # Step 2: the conditional_gradient contain the previous update.
        cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    (1.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[0]) * 0.5
                    - (1 - 0.5) * 0.01 * top_singular_vector0[0],
                    (2.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[1]) * 0.5
                    - (1 - 0.5) * 0.01 * top_singular_vector0[1],
                ]
            ),
            var0.numpy(),
        )
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    (3.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[0]) * 0.5
                    - (1 - 0.5) * 0.01 * top_singular_vector1[1],
                    (4.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[0]) * 0.5
                    - (1 - 0.5) * 0.01 * top_singular_vector1[1],
                ]
            ),
            var1.numpy(),
        )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_minimize_sparse_resource_variable_nuclear():
    # TODO:
    #       to address issue #347 and #36764.
    for dtype in _dtypes_with_checking_system(
        use_gpu=test_utils.is_gpu_available(), system=platform.system()
    ):
        var0 = tf.Variable([[1.0, 2.0]], dtype=dtype)

        def loss():
            x = tf.constant([[4.0], [5.0]], dtype=dtype)
            pred = tf.matmul(tf.nn.embedding_lookup([var0], [0]), x)
            return pred * pred

        # the gradient based on the current loss function
        grads0_0 = 32 * 1.0 + 40 * 2.0
        grads0_1 = 40 * 1.0 + 50 * 2.0
        grads0 = tf.constant([[grads0_0, grads0_1]], dtype=dtype)
        top_singular_vector0 = cg_lib.ConditionalGradient._top_singular_vector(grads0)

        learning_rate = 0.1
        lambda_ = 0.1
        ord = "nuclear"
        opt = cg_lib.ConditionalGradient(
            learning_rate=learning_rate, lambda_=lambda_, ord=ord
        )
        _ = opt.minimize(loss, var_list=[var0])

        # Validate updated params
        test_utils.assert_allclose_according_to_type(
            [
                [
                    1.0 * learning_rate
                    - (1 - learning_rate) * lambda_ * top_singular_vector0[0][0],
                    2.0 * learning_rate
                    - (1 - learning_rate) * lambda_ * top_singular_vector0[0][1],
                ]
            ],
            var0.numpy(),
        )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_tensor_learning_rate_and_conditional_gradient_nuclear():
    for dtype in _dtypes_with_checking_system(
        use_gpu=test_utils.is_gpu_available(), system=platform.system()
    ):
        # TODO:
        # Based on issue #36764 in the following link,
        #        "https://github.com/tensorflow/tensorflow/issues/36764"
        # tf.half is not registered for tf.linalg.svd function on Windows
        # CPU version.
        # So we have to remove tf.half when testing with Windows CPU version.
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        top_singular_vector0 = cg_lib.ConditionalGradient._top_singular_vector(grads0)
        top_singular_vector1 = cg_lib.ConditionalGradient._top_singular_vector(grads1)
        ord = "nuclear"
        cg_opt = cg_lib.ConditionalGradient(
            learning_rate=tf.constant(0.5), lambda_=tf.constant(0.01), ord=ord
        )
        _ = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        # Check we have slots
        assert ["conditional_gradient"] == cg_opt.get_slot_names()
        slot0 = cg_opt.get_slot(var0, "conditional_gradient")
        assert slot0.get_shape() == var0.get_shape()
        slot1 = cg_opt.get_slot(var1, "conditional_gradient")
        assert slot1.get_shape() == var1.get_shape()

        # Check that the parameters have been updated.
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    1.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[0],
                    2.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[1],
                ]
            ),
            var0.numpy(),
        )
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    3.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[0],
                    4.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[1],
                ]
            ),
            var1.numpy(),
        )
        # Step 2: the conditional_gradient contain the
        # previous update.
        cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        # Check that the parameters have been updated.
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    (1.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[0]) * 0.5
                    - (1 - 0.5) * 0.01 * top_singular_vector0[0],
                    (2.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector0[1]) * 0.5
                    - (1 - 0.5) * 0.01 * top_singular_vector0[1],
                ]
            ),
            var0.numpy(),
        )
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    (3.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[0]) * 0.5
                    - (1 - 0.5) * 0.01 * top_singular_vector1[0],
                    (4.0 * 0.5 - (1 - 0.5) * 0.01 * top_singular_vector1[1]) * 0.5
                    - (1 - 0.5) * 0.01 * top_singular_vector1[1],
                ]
            ),
            var1.numpy(),
        )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_variables_across_graphs_frobenius():
    optimizer = cg_lib.ConditionalGradient(0.01, 0.5, ord="fro")
    var0 = tf.Variable([1.0, 2.0], dtype=tf.float32, name="var0")
    var1 = tf.Variable([3.0, 4.0], dtype=tf.float32, name="var1")

    def loss():
        return tf.math.reduce_sum(var0 + var1)

    optimizer.minimize(loss, var_list=[var0, var1])
    optimizer_variables = optimizer.variables()
    # There should be three items. The first item is iteration,
    # and one item for each variable.
    assert optimizer_variables[1].name.startswith("ConditionalGradient/var0")
    assert optimizer_variables[2].name.startswith("ConditionalGradient/var1")
    assert 3 == len(optimizer_variables)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_variables_across_graphs_nuclear():
    optimizer = cg_lib.ConditionalGradient(0.01, 0.5, ord="nuclear")
    var0 = tf.Variable([1.0, 2.0], dtype=tf.float32, name="var0")
    var1 = tf.Variable([3.0, 4.0], dtype=tf.float32, name="var1")

    def loss():
        return tf.math.reduce_sum(var0 + var1)

    optimizer.minimize(loss, var_list=[var0, var1])
    optimizer_variables = optimizer.variables()
    # There should be three items. The first item is iteration,
    # and one item for each variable.
    assert optimizer_variables[1].name.startswith("ConditionalGradient/var0")
    assert optimizer_variables[2].name.startswith("ConditionalGradient/var1")
    assert 3 == len(optimizer_variables)


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_minimize_with_2D_indicies_for_embedding_lookup_frobenius():
    # This test invokes the ResourceSparseApplyConditionalGradient
    # operation.
    var0 = tf.Variable(tf.ones([2, 2]))

    def loss():
        return tf.math.reduce_sum(tf.nn.embedding_lookup(var0, [[1]]))

    # the gradient for this loss function:
    grads0 = tf.constant([[0, 0], [1, 1]], dtype=tf.float32)
    norm0 = tf.math.reduce_sum(grads0**2) ** 0.5

    learning_rate = 0.1
    lambda_ = 0.1
    ord = "fro"
    opt = cg_lib.ConditionalGradient(
        learning_rate=learning_rate, lambda_=lambda_, ord=ord
    )
    _ = opt.minimize(loss, var_list=[var0])

    # Run 1 step of cg_op
    test_utils.assert_allclose_according_to_type(
        [
            [1, 1],
            [
                learning_rate * 1 - (1 - learning_rate) * lambda_ * 1 / norm0,
                learning_rate * 1 - (1 - learning_rate) * lambda_ * 1 / norm0,
            ],
        ],
        var0.numpy(),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_minimize_with_2D_indicies_for_embedding_lookup_nuclear():
    # This test invokes the ResourceSparseApplyConditionalGradient
    # operation.
    var0 = tf.Variable(tf.ones([2, 2]))

    def loss():
        return tf.math.reduce_sum(tf.nn.embedding_lookup(var0, [[1]]))

    # the gradient for this loss function:
    grads0 = tf.constant([[0, 0], [1, 1]], dtype=tf.float32)
    top_singular_vector0 = cg_lib.ConditionalGradient._top_singular_vector(grads0)

    learning_rate = 0.1
    lambda_ = 0.1
    ord = "nuclear"
    opt = cg_lib.ConditionalGradient(
        learning_rate=learning_rate, lambda_=lambda_, ord=ord
    )
    _ = opt.minimize(loss, var_list=[var0])

    # Run 1 step of cg_op
    test_utils.assert_allclose_according_to_type(
        [
            learning_rate * 1
            - (1 - learning_rate) * lambda_ * top_singular_vector0[1][0],
            learning_rate * 1
            - (1 - learning_rate) * lambda_ * top_singular_vector0[1][1],
        ],
        var0[1],
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
def test_tensor_learning_rate_and_conditional_gradient_frobenius(dtype):
    var0 = tf.Variable([1.0, 2.0], dtype=dtype)
    var1 = tf.Variable([3.0, 4.0], dtype=dtype)
    grads0 = tf.constant([0.1, 0.1], dtype=dtype)
    grads1 = tf.constant([0.01, 0.01], dtype=dtype)
    norm0 = tf.math.reduce_sum(grads0**2) ** 0.5
    norm1 = tf.math.reduce_sum(grads1**2) ** 0.5
    ord = "fro"
    cg_opt = cg_lib.ConditionalGradient(
        learning_rate=tf.constant(0.5), lambda_=tf.constant(0.01), ord=ord
    )
    _ = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

    # Check we have slots
    assert ["conditional_gradient"] == cg_opt.get_slot_names()
    slot0 = cg_opt.get_slot(var0, "conditional_gradient")
    assert slot0.get_shape() == var0.get_shape()
    slot1 = cg_opt.get_slot(var1, "conditional_gradient")
    assert slot1.get_shape() == var1.get_shape()

    # Check that the parameters have been updated.
    test_utils.assert_allclose_according_to_type(
        np.array(
            [
                1.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0,
                2.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0,
            ]
        ),
        var0.numpy(),
    )
    test_utils.assert_allclose_according_to_type(
        np.array(
            [
                3.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1,
                4.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1,
            ]
        ),
        var1.numpy(),
    )
    # Step 2: the conditional_gradient contain the
    # previous update.
    cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    # Check that the parameters have been updated.
    test_utils.assert_allclose_according_to_type(
        np.array(
            [
                (1.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0) * 0.5
                - (1 - 0.5) * 0.01 * 0.1 / norm0,
                (2.0 * 0.5 - (1 - 0.5) * 0.01 * 0.1 / norm0) * 0.5
                - (1 - 0.5) * 0.01 * 0.1 / norm0,
            ]
        ),
        var0.numpy(),
    )
    test_utils.assert_allclose_according_to_type(
        np.array(
            [
                (3.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1) * 0.5
                - (1 - 0.5) * 0.01 * 0.01 / norm1,
                (4.0 * 0.5 - (1 - 0.5) * 0.01 * 0.01 / norm1) * 0.5
                - (1 - 0.5) * 0.01 * 0.01 / norm1,
            ]
        ),
        var1.numpy(),
    )


def _db_params_frobenius_cg01():
    """Return dist-belief conditional_gradient values.

    Return values been generated from the dist-belief
    conditional_gradient unittest, running with a learning rate of 0.1
    and a lambda_ of 0.1.

    These values record how a parameter vector of size 10, initialized
    with 0.0, gets updated with 10 consecutive conditional_gradient
    steps.
    It uses random gradients.

    Returns:
        db_grad: The gradients to apply
        db_out: The parameters after the conditional_gradient update.
    """
    db_grad = [[]] * 10
    db_out = [[]] * 10
    db_grad[0] = [
        0.00096264342,
        0.17914793,
        0.93945462,
        0.41396621,
        0.53037018,
        0.93197989,
        0.78648776,
        0.50036013,
        0.55345792,
        0.96722615,
    ]
    db_out[0] = [
        -4.1555551e-05,
        -7.7334875e-03,
        -4.0554531e-02,
        -1.7870162e-02,
        -2.2895107e-02,
        -4.0231861e-02,
        -3.3951234e-02,
        -2.1599628e-02,
        -2.3891762e-02,
        -4.1753378e-02,
    ]
    db_grad[1] = [
        0.17075552,
        0.88821375,
        0.20873757,
        0.25236958,
        0.57578111,
        0.15312378,
        0.5513742,
        0.94687688,
        0.16012503,
        0.22159521,
    ]
    db_out[1] = [
        -0.00961733,
        -0.0507779,
        -0.01580694,
        -0.01599489,
        -0.03470477,
        -0.01264373,
        -0.03443632,
        -0.05546713,
        -0.01140388,
        -0.01665068,
    ]
    db_grad[2] = [
        0.35077485,
        0.47304362,
        0.44412705,
        0.44368884,
        0.078527533,
        0.81223965,
        0.31168157,
        0.43203235,
        0.16792089,
        0.24644311,
    ]
    db_out[2] = [
        -0.02462724,
        -0.03699233,
        -0.03154434,
        -0.03153357,
        -0.00876844,
        -0.05606323,
        -0.02447166,
        -0.03469437,
        -0.0124694,
        -0.01829169,
    ]
    db_grad[3] = [
        0.9694621,
        0.75035888,
        0.28171822,
        0.83813518,
        0.53807181,
        0.3728098,
        0.81454384,
        0.03848977,
        0.89759839,
        0.93665648,
    ]
    db_out[3] = [
        -0.04124615,
        -0.03371741,
        -0.0144246,
        -0.03668303,
        -0.02240246,
        -0.02052062,
        -0.03503307,
        -0.00500922,
        -0.03715545,
        -0.0393002,
    ]
    db_grad[4] = [
        0.38578293,
        0.8536852,
        0.88722926,
        0.66276771,
        0.13678469,
        0.94036359,
        0.69107032,
        0.81897682,
        0.5433259,
        0.67860287,
    ]
    db_out[4] = [
        -0.01979208,
        -0.0380417,
        -0.03747472,
        -0.0305847,
        -0.00779536,
        -0.04024222,
        -0.03156913,
        -0.0337613,
        -0.02578116,
        -0.03148952,
    ]
    db_grad[5] = [
        0.27885768,
        0.76100707,
        0.24625534,
        0.81354135,
        0.18959245,
        0.48038563,
        0.84163809,
        0.41172323,
        0.83259648,
        0.44941229,
    ]
    db_out[5] = [
        -0.01555188,
        -0.04084422,
        -0.01573331,
        -0.04265549,
        -0.01000746,
        -0.02740575,
        -0.04412147,
        -0.02341569,
        -0.0431026,
        -0.02502293,
    ]
    db_grad[6] = [
        0.27233034,
        0.056316052,
        0.5039115,
        0.24105175,
        0.35697976,
        0.75913221,
        0.73577434,
        0.16014607,
        0.57500273,
        0.071136251,
    ]
    db_out[6] = [
        -0.01890448,
        -0.00767214,
        -0.03367592,
        -0.01962219,
        -0.02374279,
        -0.05110247,
        -0.05128598,
        -0.01254396,
        -0.04094185,
        -0.00703416,
    ]
    db_grad[7] = [
        0.58697265,
        0.2494842,
        0.08106143,
        0.39954534,
        0.15892942,
        0.12683646,
        0.74053431,
        0.16033,
        0.66625422,
        0.73515922,
    ]
    db_out[7] = [
        -0.03772914,
        -0.01599993,
        -0.00831695,
        -0.02635719,
        -0.01207801,
        -0.01285448,
        -0.05034328,
        -0.01104364,
        -0.04477356,
        -0.04558991,
    ]
    db_grad[8] = [
        0.8215279,
        0.41994119,
        0.95172721,
        0.68000203,
        0.79439718,
        0.43384039,
        0.55561525,
        0.22567581,
        0.93331909,
        0.29438227,
    ]
    db_out[8] = [
        -0.03919835,
        -0.01970845,
        -0.04187151,
        -0.03195836,
        -0.03546333,
        -0.01999326,
        -0.02899324,
        -0.01083582,
        -0.04472339,
        -0.01725317,
    ]
    db_grad[9] = [
        0.68297005,
        0.67758518,
        0.1748755,
        0.13266537,
        0.70697063,
        0.055731893,
        0.68593478,
        0.50580865,
        0.12602448,
        0.093537711,
    ]
    db_out[9] = [
        -0.04510314,
        -0.04282944,
        -0.0147322,
        -0.0111956,
        -0.04617687,
        -0.00535998,
        -0.0442614,
        -0.03158399,
        -0.01207165,
        -0.00736567,
    ]
    return db_grad, db_out


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_like_dist_belief_frobenius_cg01():
    db_grad, db_out = _db_params_frobenius_cg01()
    num_samples = len(db_grad)
    var0 = tf.Variable([0.0] * num_samples)
    grads0 = tf.constant([0.0] * num_samples)
    ord = "fro"
    cg_opt = cg_lib.ConditionalGradient(learning_rate=0.1, lambda_=0.1, ord=ord)

    for i in range(num_samples):
        grads0 = tf.constant(db_grad[i])
        cg_opt.apply_gradients(zip([grads0], [var0]))
        np.testing.assert_allclose(
            np.array(db_out[i]), var0.numpy(), rtol=1e-06, atol=1e-06
        )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_frobenius():
    # TODO:
    #       To address the issue #347.
    for dtype in _dtypes_to_test(use_gpu=test_utils.is_gpu_available()):
        var0 = tf.Variable(tf.zeros([4, 2], dtype=dtype))
        var1 = tf.Variable(tf.constant(1.0, dtype, [4, 2]))
        grads0 = tf.IndexedSlices(
            tf.constant([[0.1, 0.1]], dtype=dtype),
            tf.constant([1]),
            tf.constant([4, 2]),
        )
        grads1 = tf.IndexedSlices(
            tf.constant([[0.01, 0.01], [0.01, 0.01]], dtype=dtype),
            tf.constant([2, 3]),
            tf.constant([4, 2]),
        )
        norm0 = tf.math.reduce_sum(tf.math.multiply(grads0, grads0)) ** 0.5
        norm1 = tf.math.reduce_sum(tf.math.multiply(grads1, grads1)) ** 0.5
        learning_rate = 0.1
        lambda_ = 0.1
        ord = "fro"
        cg_opt = cg_lib.ConditionalGradient(
            learning_rate=learning_rate, lambda_=lambda_, ord=ord
        )
        _ = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        # Check we have slots
        assert ["conditional_gradient"] == cg_opt.get_slot_names()
        slot0 = cg_opt.get_slot(var0, "conditional_gradient")
        assert slot0.get_shape() == var0.get_shape()
        slot1 = cg_opt.get_slot(var1, "conditional_gradient")
        assert slot1.get_shape() == var1.get_shape()

        # Check that the parameters have been updated.
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    0 - (1 - learning_rate) * lambda_ * 0 / norm0,
                    0 - (1 - learning_rate) * lambda_ * 0 / norm0,
                ]
            ),
            var0[0].numpy(),
        )
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    0 - (1 - learning_rate) * lambda_ * 0.1 / norm0,
                    0 - (1 - learning_rate) * lambda_ * 0.1 / norm0,
                ]
            ),
            var0[1].numpy(),
        )
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    1.0 * learning_rate - (1 - learning_rate) * lambda_ * 0.01 / norm1,
                    1.0 * learning_rate - (1 - learning_rate) * lambda_ * 0.01 / norm1,
                ]
            ),
            var1[2].numpy(),
        )
        # Step 2: the conditional_gradient contain the
        # previous update.
        cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        # Check that the parameters have been updated.
        np.testing.assert_allclose(np.array([0, 0]), var0[0].numpy())
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    (0 - (1 - learning_rate) * lambda_ * 0.1 / norm0) * learning_rate
                    - (1 - learning_rate) * lambda_ * 0.1 / norm0,
                    (0 - (1 - learning_rate) * lambda_ * 0.1 / norm0) * learning_rate
                    - (1 - learning_rate) * lambda_ * 0.1 / norm0,
                ]
            ),
            var0[1].numpy(),
        )
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    (1.0 * learning_rate - (1 - learning_rate) * lambda_ * 0.01 / norm1)
                    * learning_rate
                    - (1 - learning_rate) * lambda_ * 0.01 / norm1,
                    (1.0 * learning_rate - (1 - learning_rate) * lambda_ * 0.01 / norm1)
                    * learning_rate
                    - (1 - learning_rate) * lambda_ * 0.01 / norm1,
                ]
            ),
            var1[2].numpy(),
        )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
@pytest.mark.parametrize("dtype", [tf.half, tf.float32, tf.float64])
def test_sharing_frobenius(dtype):
    var0 = tf.Variable([1.0, 2.0], dtype=dtype)
    var1 = tf.Variable([3.0, 4.0], dtype=dtype)
    grads0 = tf.constant([0.1, 0.1], dtype=dtype)
    grads1 = tf.constant([0.01, 0.01], dtype=dtype)
    norm0 = tf.math.reduce_sum(grads0**2) ** 0.5
    norm1 = tf.math.reduce_sum(grads1**2) ** 0.5
    learning_rate = 0.1
    lambda_ = 0.1
    ord = "fro"
    cg_opt = cg_lib.ConditionalGradient(
        learning_rate=learning_rate, lambda_=lambda_, ord=ord
    )
    _ = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

    # Check we have slots
    assert ["conditional_gradient"] == cg_opt.get_slot_names()
    slot0 = cg_opt.get_slot(var0, "conditional_gradient")
    assert slot0.get_shape() == var0.get_shape()
    slot1 = cg_opt.get_slot(var1, "conditional_gradient")
    assert slot1.get_shape() == var1.get_shape()

    # Because in the eager mode, as we declare two cg_update
    # variables, it already altomatically finish executing them.
    # Thus, we cannot test the param value at this time for
    # eager mode. We can only test the final value of param
    # after the second execution.

    # Step 2: the second conditional_gradient contain
    # the previous update.
    # Check that the parameters have been updated.
    cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    test_utils.assert_allclose_according_to_type(
        np.array(
            [
                (1.0 * learning_rate - (1 - learning_rate) * lambda_ * 0.1 / norm0)
                * learning_rate
                - (1 - learning_rate) * lambda_ * 0.1 / norm0,
                (2.0 * learning_rate - (1 - learning_rate) * lambda_ * 0.1 / norm0)
                * learning_rate
                - (1 - learning_rate) * lambda_ * 0.1 / norm0,
            ]
        ),
        var0.numpy(),
    )
    test_utils.assert_allclose_according_to_type(
        np.array(
            [
                (3.0 * learning_rate - (1 - learning_rate) * lambda_ * 0.01 / norm1)
                * learning_rate
                - (1 - learning_rate) * lambda_ * 0.01 / norm1,
                (4.0 * learning_rate - (1 - learning_rate) * lambda_ * 0.01 / norm1)
                * learning_rate
                - (1 - learning_rate) * lambda_ * 0.01 / norm1,
            ]
        ),
        var1.numpy(),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sharing_nuclear():
    # TODO:
    #       To address the issue #36764.
    for dtype in _dtypes_with_checking_system(
        use_gpu=test_utils.is_gpu_available(), system=platform.system()
    ):
        var0 = tf.Variable([1.0, 2.0], dtype=dtype)
        var1 = tf.Variable([3.0, 4.0], dtype=dtype)
        grads0 = tf.constant([0.1, 0.1], dtype=dtype)
        grads1 = tf.constant([0.01, 0.01], dtype=dtype)
        top_singular_vector0 = cg_lib.ConditionalGradient._top_singular_vector(grads0)
        top_singular_vector1 = cg_lib.ConditionalGradient._top_singular_vector(grads1)
        learning_rate = 0.1
        lambda_ = 0.1
        ord = "nuclear"
        cg_opt = cg_lib.ConditionalGradient(
            learning_rate=learning_rate, lambda_=lambda_, ord=ord
        )
        _ = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        # Check we have slots
        assert ["conditional_gradient"] == cg_opt.get_slot_names()
        slot0 = cg_opt.get_slot(var0, "conditional_gradient")
        assert slot0.get_shape() == var0.get_shape()
        slot1 = cg_opt.get_slot(var1, "conditional_gradient")
        assert slot1.get_shape() == var1.get_shape()

        # Because in the eager mode, as we declare two cg_update
        # variables, it already altomatically finish executing them.
        # Thus, we cannot test the param value at this time for
        # eager mode. We can only test the final value of param
        # after the second execution.

        # Step 2: the second conditional_gradient contain
        # the previous update.
        # Check that the parameters have been updated.
        cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    (
                        1.0 * learning_rate
                        - (1 - learning_rate) * lambda_ * top_singular_vector0[0]
                    )
                    * learning_rate
                    - (1 - learning_rate) * lambda_ * top_singular_vector0[0],
                    (
                        2.0 * learning_rate
                        - (1 - learning_rate) * lambda_ * top_singular_vector0[1]
                    )
                    * learning_rate
                    - (1 - learning_rate) * lambda_ * top_singular_vector0[1],
                ]
            ),
            var0.numpy(),
        )
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    (
                        3.0 * learning_rate
                        - (1 - learning_rate) * lambda_ * top_singular_vector1[0]
                    )
                    * learning_rate
                    - (1 - learning_rate) * lambda_ * top_singular_vector1[0],
                    (
                        4.0 * learning_rate
                        - (1 - learning_rate) * lambda_ * top_singular_vector1[1]
                    )
                    * learning_rate
                    - (1 - learning_rate) * lambda_ * top_singular_vector1[1],
                ]
            ),
            var1.numpy(),
        )


def _db_params_nuclear_cg01():
    """Return dist-belief conditional_gradient values.

    Return values been generated from the dist-belief
    conditional_gradient unittest, running with a learning rate of 0.1
    and a lambda_ of 0.1.

    These values record how a parameter vector of size 10, initialized
    with 0.0, gets updated with 10 consecutive conditional_gradient
    steps.
    It uses random gradients.

    Returns:
        db_grad: The gradients to apply
        db_out: The parameters after the conditional_gradient update.
    """
    db_grad = [[]] * 10
    db_out = [[]] * 10
    db_grad[0] = [
        0.00096264342,
        0.17914793,
        0.93945462,
        0.41396621,
        0.53037018,
        0.93197989,
        0.78648776,
        0.50036013,
        0.55345792,
        0.96722615,
    ]
    db_out[0] = [
        -4.1552783e-05,
        -7.7334875e-03,
        -4.0554535e-02,
        -1.7870164e-02,
        -2.2895109e-02,
        -4.0231861e-02,
        -3.3951234e-02,
        -2.1599628e-02,
        -2.3891764e-02,
        -4.1753381e-02,
    ]
    db_grad[1] = [
        0.17075552,
        0.88821375,
        0.20873757,
        0.25236958,
        0.57578111,
        0.15312378,
        0.5513742,
        0.94687688,
        0.16012503,
        0.22159521,
    ]
    db_out[1] = [
        -0.00961733,
        -0.0507779,
        -0.01580694,
        -0.01599489,
        -0.03470477,
        -0.01264373,
        -0.03443632,
        -0.05546713,
        -0.01140388,
        -0.01665068,
    ]
    db_grad[2] = [
        0.35077485,
        0.47304362,
        0.44412705,
        0.44368884,
        0.078527533,
        0.81223965,
        0.31168157,
        0.43203235,
        0.16792089,
        0.24644311,
    ]
    db_out[2] = [
        -0.02462724,
        -0.03699233,
        -0.03154433,
        -0.03153357,
        -0.00876844,
        -0.05606324,
        -0.02447166,
        -0.03469437,
        -0.0124694,
        -0.01829169,
    ]
    db_grad[3] = [
        0.9694621,
        0.75035888,
        0.28171822,
        0.83813518,
        0.53807181,
        0.3728098,
        0.81454384,
        0.03848977,
        0.89759839,
        0.93665648,
    ]
    db_out[3] = [
        -0.04124615,
        -0.03371741,
        -0.0144246,
        -0.03668303,
        -0.02240246,
        -0.02052062,
        -0.03503307,
        -0.00500922,
        -0.03715545,
        -0.0393002,
    ]
    db_grad[4] = [
        0.38578293,
        0.8536852,
        0.88722926,
        0.66276771,
        0.13678469,
        0.94036359,
        0.69107032,
        0.81897682,
        0.5433259,
        0.67860287,
    ]
    db_out[4] = [
        -0.01979207,
        -0.0380417,
        -0.03747472,
        -0.0305847,
        -0.00779536,
        -0.04024221,
        -0.03156913,
        -0.0337613,
        -0.02578116,
        -0.03148951,
    ]
    db_grad[5] = [
        0.27885768,
        0.76100707,
        0.24625534,
        0.81354135,
        0.18959245,
        0.48038563,
        0.84163809,
        0.41172323,
        0.83259648,
        0.44941229,
    ]
    db_out[5] = [
        -0.01555188,
        -0.04084422,
        -0.01573331,
        -0.04265549,
        -0.01000746,
        -0.02740575,
        -0.04412147,
        -0.02341569,
        -0.0431026,
        -0.02502293,
    ]
    db_grad[6] = [
        0.27233034,
        0.056316052,
        0.5039115,
        0.24105175,
        0.35697976,
        0.75913221,
        0.73577434,
        0.16014607,
        0.57500273,
        0.071136251,
    ]
    db_out[6] = [
        -0.01890448,
        -0.00767214,
        -0.03367592,
        -0.01962219,
        -0.02374278,
        -0.05110246,
        -0.05128598,
        -0.01254396,
        -0.04094184,
        -0.00703416,
    ]
    db_grad[7] = [
        0.58697265,
        0.2494842,
        0.08106143,
        0.39954534,
        0.15892942,
        0.12683646,
        0.74053431,
        0.16033,
        0.66625422,
        0.73515922,
    ]
    db_out[7] = [
        -0.03772915,
        -0.01599993,
        -0.00831695,
        -0.0263572,
        -0.01207801,
        -0.01285448,
        -0.05034329,
        -0.01104364,
        -0.04477356,
        -0.04558992,
    ]
    db_grad[8] = [
        0.8215279,
        0.41994119,
        0.95172721,
        0.68000203,
        0.79439718,
        0.43384039,
        0.55561525,
        0.22567581,
        0.93331909,
        0.29438227,
    ]
    db_out[8] = [
        -0.03919835,
        -0.01970845,
        -0.04187151,
        -0.03195836,
        -0.03546333,
        -0.01999326,
        -0.02899324,
        -0.01083582,
        -0.04472339,
        -0.01725317,
    ]
    db_grad[9] = [
        0.68297005,
        0.67758518,
        0.1748755,
        0.13266537,
        0.70697063,
        0.055731893,
        0.68593478,
        0.50580865,
        0.12602448,
        0.093537711,
    ]
    db_out[9] = [
        -0.04510314,
        -0.04282944,
        -0.0147322,
        -0.0111956,
        -0.04617687,
        -0.00535998,
        -0.0442614,
        -0.031584,
        -0.01207165,
        -0.00736567,
    ]
    return db_grad, db_out


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_sparse_nuclear():
    # TODO:
    #       To address the issue #347 and issue #36764.
    for dtype in _dtypes_with_checking_system(
        use_gpu=test_utils.is_gpu_available(), system=platform.system()
    ):
        var0 = tf.Variable(tf.zeros([4, 2], dtype=dtype))
        var1 = tf.Variable(tf.constant(1.0, dtype, [4, 2]))
        grads0 = tf.IndexedSlices(
            tf.constant([[0.1, 0.1]], dtype=dtype),
            tf.constant([1]),
            tf.constant([4, 2]),
        )
        grads1 = tf.IndexedSlices(
            tf.constant([[0.01, 0.01], [0.01, 0.01]], dtype=dtype),
            tf.constant([2, 3]),
            tf.constant([4, 2]),
        )
        top_singular_vector0 = tf.constant(
            [[0.0, 0.0], [0.7071067, 0.7071067], [0.0, 0.0], [0.0, 0.0]], dtype=dtype
        )
        top_singular_vector1 = tf.constant(
            [
                [-4.2146844e-08, -4.2146844e-08],
                [0.0000000e00, 0.0000000e00],
                [4.9999994e-01, 4.9999994e-01],
                [4.9999994e-01, 4.9999994e-01],
            ],
            dtype=dtype,
        )
        learning_rate = 0.1
        lambda_ = 0.1
        ord = "nuclear"
        cg_opt = cg_lib.ConditionalGradient(
            learning_rate=learning_rate, lambda_=lambda_, ord=ord
        )
        _ = cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        # Check we have slots
        assert ["conditional_gradient"] == cg_opt.get_slot_names()
        slot0 = cg_opt.get_slot(var0, "conditional_gradient")
        assert slot0.get_shape() == var0.get_shape()
        slot1 = cg_opt.get_slot(var1, "conditional_gradient")
        assert slot1.get_shape() == var1.get_shape()

        # Check that the parameters have been updated.
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    0 - (1 - learning_rate) * lambda_ * top_singular_vector0[0][0],
                    0 - (1 - learning_rate) * lambda_ * top_singular_vector0[0][1],
                ]
            ),
            var0[0].numpy(),
        )
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    0 - (1 - learning_rate) * lambda_ * top_singular_vector0[1][0],
                    0 - (1 - learning_rate) * lambda_ * top_singular_vector0[1][1],
                ]
            ),
            var0[1].numpy(),
        )
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    1.0 * learning_rate
                    - (1 - learning_rate) * lambda_ * top_singular_vector1[2][0],
                    1.0 * learning_rate
                    - (1 - learning_rate) * lambda_ * top_singular_vector1[2][1],
                ]
            ),
            var1[2].numpy(),
        )
        # Step 2: the conditional_gradient contain the
        # previous update.
        cg_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

        # Check that the parameters have been updated.
        np.testing.assert_allclose(np.array([0, 0]), var0[0].numpy())
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    (0 - (1 - learning_rate) * lambda_ * top_singular_vector0[1][0])
                    * learning_rate
                    - (1 - learning_rate) * lambda_ * top_singular_vector0[1][0],
                    (0 - (1 - learning_rate) * lambda_ * top_singular_vector0[1][1])
                    * learning_rate
                    - (1 - learning_rate) * lambda_ * top_singular_vector0[1][1],
                ]
            ),
            var0[1].numpy(),
        )
        test_utils.assert_allclose_according_to_type(
            np.array(
                [
                    (
                        1.0 * learning_rate
                        - (1 - learning_rate) * lambda_ * top_singular_vector1[2][0]
                    )
                    * learning_rate
                    - (1 - learning_rate) * lambda_ * top_singular_vector1[2][0],
                    (
                        1.0 * learning_rate
                        - (1 - learning_rate) * lambda_ * top_singular_vector1[2][1]
                    )
                    * learning_rate
                    - (1 - learning_rate) * lambda_ * top_singular_vector1[2][1],
                ]
            ),
            var1[2].numpy(),
        )


def test_serialization():
    learning_rate = 0.1
    lambda_ = 0.1
    ord = "nuclear"
    optimizer = cg_lib.ConditionalGradient(
        learning_rate=learning_rate, lambda_=lambda_, ord=ord
    )
    config = tf.keras.optimizers.serialize(optimizer)
    new_optimizer = tf.keras.optimizers.deserialize(config)
    assert optimizer.get_config() == new_optimizer.get_config()
