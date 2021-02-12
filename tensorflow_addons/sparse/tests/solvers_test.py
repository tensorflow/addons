# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tfa.sparse.sparse."""
import numpy as np
import pytest
import tensorflow.compat.v2 as tf
from tensorflow_addons.sparse import bicgstab_solver


def _init_complex(random_state, size):
  return tf.Variable(
      random_state.normal(size=size) + 1j * random_state.normal(size=size))


def _setup(batching, adjoint_a):
  mat_length = 4  # The matrix is expected to be square for bicgstab.
  num_cols = 2

  if batching:
    mat_size = (3, mat_length, mat_length)
    vec_size = (3, mat_length, num_cols)
  else:
    mat_size = (mat_length, mat_length)
    vec_size = (mat_length, num_cols)

  random_state = np.random.RandomState(seed=1)
  mat = _init_complex(random_state, mat_size)
  expected_solution = _init_complex(random_state, vec_size)
  rhs = tf.matmul(mat, expected_solution, adjoint_a=adjoint_a)
  return mat, rhs, expected_solution


@pytest.mark.parametrize("batching", [True, False])
@pytest.mark.parametrize("adjoint_a", [True, False])
def test_bicgstab_solver(batching, adjoint_a):
  mat, rhs, expected_solution = _setup(batching, adjoint_a)
  csr_mat = tf.raw_ops.DenseToCSRSparseMatrix(dense_input=mat,
                                              indices=tf.where(mat))
  tf.debugging.assert_near(
      bicgstab_solver(a=csr_mat, b=rhs, adjoint_a=adjoint_a), expected_solution,
        rtol=1e-10)


@pytest.mark.parametrize("batching", [True, False])
@pytest.mark.parametrize("adjoint_a", [True, False])
def test_bicgstab_solver_gradient(batching, adjoint_a):
  mat, rhs, _ = _setup(batching, adjoint_a)

  with tf.GradientTape() as tape:
    csr_mat = tf.raw_ops.DenseToCSRSparseMatrix(dense_input=mat,
                                                indices=tf.where(mat))
    solution_sparse = bicgstab_solver(a=csr_mat, b=rhs, adjoint_a=adjoint_a)
  grad_sparse = tape.gradient(solution_sparse, mat)

  with tf.GradientTape() as tape:
    solution_dense = tf.linalg.solve(matrix=mat, rhs=rhs, adjoint=adjoint_a)
  grad_dense = tape.gradient(solution_dense, mat)
  tf.debugging.assert_near(grad_sparse, grad_dense, rtol=1e-10)
