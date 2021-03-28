// Copyright 2021 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace addons {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeAndType;
using shape_inference::ShapeHandle;

Status GetVariantInput(InferenceContext* c, int index,
                       ShapeAndType* shape_and_type) {
  ShapeHandle variant;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(index), 0, &variant));
  auto* shapes_and_types = c->input_handle_shapes_and_types(index);
  if (shapes_and_types == nullptr || shapes_and_types->size() != 1) {
    return errors::InvalidArgument(
        "Unable to access shape and type info from variant input ", index);
  }
  *shape_and_type = shapes_and_types->at(0);
  return Status::OK();
}

// Validates that a shape represents a (rank-2) square matrix or a (rank-3)
// batch of square matrices.
Status ValidateSquareMatrixShape(InferenceContext* c,
                                 const ShapeHandle& matrix_shape,
                                 DimensionHandle* matrix_dimension) {
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(matrix_shape, 2, &out));
  TF_RETURN_IF_ERROR(c->WithRankAtMost(matrix_shape, 3, &out));
  if (!c->RankKnown(matrix_shape)) {
    return errors::Internal("Sparse matrix has an unknown rank.");
  }

  TF_RETURN_IF_ERROR(c->Merge(c->Dim(matrix_shape, -2),
                              c->Dim(matrix_shape, -1), matrix_dimension));
  return Status::OK();
}

REGISTER_OP("Addons>SparseMatrixBiCGSTABSolver")
    .Input("a: variant")
    .Input("b: Type")
    .Attr("Type: type")
    .Attr("adjoint_a: bool = false")
    .Output("output: Type")
    .SetShapeFn([](InferenceContext* c) {
      ShapeAndType sparse_matrix_shape_and_type;
      TF_RETURN_IF_ERROR(GetVariantInput(c, 0, &sparse_matrix_shape_and_type));
      ShapeHandle a_shape = sparse_matrix_shape_and_type.shape;
      DimensionHandle n;
      TF_RETURN_IF_ERROR(ValidateSquareMatrixShape(c, a_shape, &n));

      ShapeHandle b_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 2, &b_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(b_shape, 3, &b_shape));

      // Batch dims match between inputs.
      ShapeHandle a_batch_dims;
      ShapeHandle b_batch_dims;
      ShapeHandle batch_dims;
      TF_RETURN_IF_ERROR(c->Subshape(a_shape, 0, -2, &a_batch_dims));
      TF_RETURN_IF_ERROR(c->Subshape(b_shape, 0, -2, &b_batch_dims));
      TF_RETURN_IF_ERROR(c->Merge(a_batch_dims, b_batch_dims, &batch_dims));

      // Assert inner dims match.
      shape_inference::DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(a_shape, -1), c->Dim(b_shape, -2), &unused));

      // Notice the 'a' matrix is square, so we don't need to use a different
      // dimension for the output rows when using adjoint_a.
      auto output_rows = c->Dim(a_shape, -2);
      auto output_cols = c->Dim(b_shape, -1);

      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Concatenate(
          batch_dims, c->Matrix(output_rows, output_cols), &out));
      c->set_output_handle_shapes_and_types(
          0, {ShapeAndType{out, sparse_matrix_shape_and_type.dtype}});
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
The input sparse matrix `a` and the dense tensor `b`  must both have rank 2 or
3. The case of rank 3 corresponds to batching along the first dimension, in
which case the first dimension of `a` and `b` must match. There is no support
for broadcasting.

If `a` and `b` have rank 2, the second dimension of `a` must match the first
dimension of `b`. Analogously, if batching (i.e., when `a` and `b` have rank 3),
the third dimension of `a` must match the second dimension of `b`.

If `a` has rank 2, it must be square. If batching, its second and third
dimensions must match.

The returned tensor `output` is dense and has the same rank and dimensions as
`b`. It corresponds to the solution `x` of the (possibly batched) linear system
of equations

```
  a * x = b
```

The `type` parameter denotes the type of the matrix elements. The supported
types are: `float32`, `float64`, `complex64` and `complex128`.

Usage example:

```python
import tensorflow as tf
from tensorflow_addons import sparse

csr_mat = tf.raw_ops.SparseTensorToCSRSparseMatrix(
    indices=tf.constant([[0, 0], [0, 2], [1, 1], [2, 2]], dtype=tf.int64),
    values=tf.constant([1.0, 2.0, 1.0, 3.0], tf.float32),
    dense_shape=tf.constant((3, 3), dtype=tf.int64))

rhs = tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32)

solution = sparse.bicgstab_solver(a=csr_mat, b=rhs)
```

`solution` stores the matrix:

```
   [[-0.99999994],
    [ 2.        ],
    [ 1.        ]]
```
)doc");

}  // end namespace addons
}  // end namespace tensorflow
