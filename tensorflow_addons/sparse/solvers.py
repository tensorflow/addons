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
"""Solvers for sparse linear systems."""

import tensorflow.compat.v2 as tf
from tensorflow_addons.utils.resource_loader import LazySO

_sparse_ops_so = LazySO("custom_ops/sparse/_sparse_ops.so")

bicgstab_solver = _sparse_ops_so.ops.addons_sparse_matrix_bi_cgstab_solver


@tf.RegisterGradient("Addons>SparseMatrixBiCGSTABSolver")
def _bicgstab_solver(op, grad):
  """Gradient for bicgstab_solver op."""

  def _pruned_dense_matrix_multiplication(a, b, indices):
    rank = len(a.shape)
    dense_shape = (a.shape[-2], b.shape[-1])  # dense shape not incluing batch.

    if rank == 2:
      rows = indices[:, 0]
      cols = indices[:, 1]
      a_rows = tf.gather(a, indices=rows)
      b_cols = tf.gather(b, indices=cols)
    elif rank == 3:
      dense_shape = (a.shape[0],) + dense_shape
      rows = indices[:, :2]
      cols = tf.stack([indices[:, 0], indices[:, 2]], axis=1)
      a_rows = tf.gather_nd(a, indices=rows)
      b_cols = tf.gather_nd(b, indices=cols)

    return tf.reduce_sum(a_rows * b_cols, axis=-1)

  a = op.inputs[0]
  adjoint_a = op.get_attr("adjoint_a")
  c = op.outputs[0]  # A^-1 b
  a_coo = tf.raw_ops.CSRSparseMatrixToSparseTensor(
      sparse_matrix=a, type=c.dtype)
  indices = a_coo.indices
  grad_b = bicgstab_solver(a=a, b=grad, adjoint_a=not adjoint_a)
  if adjoint_a:
    grad_a_values = -_pruned_dense_matrix_multiplication(
        c, tf.math.conj(grad_b), indices)
  else:
    grad_a_values = -_pruned_dense_matrix_multiplication(
        grad_b, tf.math.conj(c), indices)
  grad_a = tf.raw_ops.SparseTensorToCSRSparseMatrix(
      indices=indices, values=grad_a_values, dense_shape=a_coo.dense_shape)
  return (grad_a, grad_b)
