/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS

// clang-format off
#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/SparseCore"
#include "Eigen/IterativeLinearSolvers"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/util/work_sharder.h"
// clang-format on

namespace tensorflow {
namespace addons {

// Op to compute the sparse inverse using the BiCGSTAB method.
//
// Implements a CPU kernel to solve a sparse linear system using Eigen
// SparseMatrix and its implementation of BiCGSTAB.
//
// The CSRSparseMatrix `a` represents a sparse matrix (rank 2) or batch of
// sparse matrices (rank 3). The Tensor `b` represents a dense matrix (rank 2)
// or batch of dense matrices (rank 3). Supports taking the adjoint of the
// sparse matrix on the fly without constructing the matrix. The rank of `a`
// and `b` must match (does not support broadcasting).
//
// Explicitly, this solves for x in the (possibly batched) system of equations
// a * x = b. Returns the (possibly batched) solution as a Tensor with the same
// shape as `b`.
//
// Supported types are float, double, complex64, and complex128.
//
// TODO(tabakg): Consider adding optional arguments (max iterations, tolerance)
// and option for initial guess. Also consider a GPU implementation.

template <typename Type>
class BiCGSTABSolverCPUOp : public OpKernel {
  using SparseMatrix = Eigen::SparseMatrix<Type, Eigen::RowMajor>;
  using Matrix =
      Eigen::Matrix<Type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixMap = Eigen::Map<Matrix>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using ConstSparseMatrixMap = Eigen::Map<const SparseMatrix>;

 public:
  explicit BiCGSTABSolverCPUOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("adjoint_a", &adjoint_a_));
  }

  void Compute(OpKernelContext* ctx) final {
    const CSRSparseMatrix* input_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &input_matrix));
    const Tensor& input_right_hand_side = ctx->input(1);
    const TensorShape& rhs_shape = input_right_hand_side.shape();

    int64 num_rows;
    int batch_size;

    ValidateInputs(ctx, *input_matrix, input_right_hand_side, &batch_size,
                   &num_rows);
    const int64 num_rhs_cols =
        input_right_hand_side.dim_size(input_right_hand_side.dims() - 1);

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());

    // The actual cost will depend on the number of iterations required.
    const int64 cost_per_batch = 100 * (input_matrix->total_nnz() / batch_size);

    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, rhs_shape, &output));

    Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
          cost_per_batch, [&](int64 batch_begin, int64 batch_end) {
            for (int64 batch_index = batch_begin; batch_index < batch_end;
                 ++batch_index) {
              ConstSparseMatrixMap sparse_matrix(
                  num_rows, num_rows,
                  input_matrix->nnz(static_cast<int>(batch_index)),
                  input_matrix->row_pointers_vec(static_cast<int>(batch_index))
                      .data(),
                  input_matrix->col_indices_vec(static_cast<int>(batch_index))
                      .data(),
                  input_matrix->values_vec<Type>(static_cast<int>(batch_index))
                      .data());

              // Map the corresponding rows of the rhs.
              ConstMatrixMap rhs_map(input_right_hand_side.flat<Type>().data() +
                                         batch_index * num_rows * num_rhs_cols,
                                     num_rows, num_rhs_cols);

              Eigen::BiCGSTAB<SparseMatrix> solver;
              if (this->adjoint_a_) {
                solver.compute(sparse_matrix.adjoint());
              } else {
                solver.compute(sparse_matrix);
              }

              MatrixMap output_map(output->flat<Type>().data() +
                                       batch_index * num_rows * num_rhs_cols,
                                   num_rows, num_rhs_cols);
              output_map.noalias() = solver.solve(rhs_map);
            }
          });
  }

 private:
  void ValidateInputs(OpKernelContext* ctx,
                      const CSRSparseMatrix& sparse_matrix,
                      const Tensor& right_hand_side, int* batch_size,
                      int64* num_rows) {
    OP_REQUIRES(ctx, (sparse_matrix.dtype() == right_hand_side.dtype()),
                errors::InvalidArgument(
                    "Input types don't match.  a.dtype == ",
                    DataTypeString(sparse_matrix.dtype()), " vs. b.dtype == ",
                    DataTypeString(right_hand_side.dtype())));

    const Tensor& dense_shape = sparse_matrix.dense_shape();
    const int rank = static_cast<int>(dense_shape.dim_size(0));
    OP_REQUIRES(ctx, rank == 2 || rank == 3,
                errors::InvalidArgument("sparse matrix must have rank 2 or 3; ",
                                        "but dense_shape has size ", rank));
    const int row_dim = (rank == 2) ? 0 : 1;
    auto dense_shape_vec = dense_shape.vec<int64>();

    // Notice the 'a' matrix is square, so we don't need to use a different
    // dimension for the output rows when using adjoint_a.
    *num_rows = dense_shape_vec(row_dim);
    const int64 num_cols = dense_shape_vec(row_dim + 1);
    OP_REQUIRES(ctx, *num_rows == num_cols,
                errors::InvalidArgument("sparse matrix must be square; got: ",
                                        *num_rows, " != ", num_cols));

    const TensorShape& rhs_shape = right_hand_side.shape();
    OP_REQUIRES(
        ctx, rhs_shape.dims() == rank,
        errors::InvalidArgument("sparse matrix must have the same rank as the "
                                "right hand side; got: ",
                                rank, " != ", rhs_shape.dims()));
    OP_REQUIRES(
        ctx, rhs_shape.dim_size(rank - 2) == num_cols,
        errors::InvalidArgument(
            "The number of rows of the right hand side must match the number of"
            "columns of the sparse matrix in each batch; got: ",
            rhs_shape.dim_size(rank - 2), " != ", num_cols));

    *batch_size = sparse_matrix.batch_size();
    if (*batch_size > 1) {
      OP_REQUIRES(
          ctx, rhs_shape.dim_size(0) == *batch_size,
          errors::InvalidArgument("The right hand side must have the same "
                                  "batch size as sparse matrix; got: ",
                                  rhs_shape.dim_size(0), " != ", *batch_size));
    }
  }

  bool adjoint_a_;
};

#define REGISTER_CPU(Type)                                          \
  REGISTER_KERNEL_BUILDER(Name("Addons>SparseMatrixBiCGSTABSolver") \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<Type>("Type"),        \
                          BiCGSTABSolverCPUOp<Type>);

REGISTER_CPU(float);
REGISTER_CPU(double);
REGISTER_CPU(complex64);
REGISTER_CPU(complex128);

#undef REGISTER_CPU
}  // namespace addons
}  // namespace tensorflow
