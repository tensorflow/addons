/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/stream_executor_util.h"
#include "tensorflow_addons/custom_ops/image/cc/kernels/adjust_hsv_in_yiq_op.h"

namespace tensorflow {
namespace addons {
namespace internal {

__global__ void compute_transformation_matrix_cuda(const float* const delta_h,
                                                   const float* const scale_s,
                                                   const float* const scale_v,
                                                   float* const matrix,
                                                   const int matrix_size) {
  if (matrix_size == kChannelSize * kChannelSize) {
    compute_transformation_matrix<kChannelSize * kChannelSize>(
        *delta_h, *scale_s, *scale_v, matrix);
  }
}
}  // namespace internal

namespace functor {

void AdjustHsvInYiqGPU::operator()(OpKernelContext* ctx, int channel_count,
                                   const Tensor* const input,
                                   const float* const delta_h,
                                   const float* const scale_s,
                                   const float* const scale_v,
                                   Tensor* const output) {
  const uint64 m = channel_count;
  const uint64 k = kChannelSize;
  const uint64 n = kChannelSize;
  auto* cu_stream = ctx->eigen_device<GPUDevice>().stream();
  OP_REQUIRES(ctx, cu_stream, errors::Internal("No GPU stream available."));
  Tensor transformation_matrix;
  OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                          DT_FLOAT, TensorShape({kChannelSize * kChannelSize}),
                          &transformation_matrix));
  // TODO(huangyp): It takes about 3.5 us to compute transformation_matrix
  // with one thread. Improve its performance if necessary.
  TF_CHECK_OK(CudaLaunchKernel(internal::compute_transformation_matrix_cuda, 1,
                               1, 0, cu_stream, delta_h, scale_s, scale_v,
                               transformation_matrix.flat<float>().data(),
                               transformation_matrix.flat<float>().size()));
  // Call cuBlas C = A * B directly.
  auto no_transpose = se::blas::Transpose::kNoTranspose;
  auto a_ptr = StreamExecutorUtil::AsDeviceMemory<float>(*input);
  auto b_ptr = StreamExecutorUtil::AsDeviceMemory<float>(transformation_matrix);
  auto c_ptr = StreamExecutorUtil::AsDeviceMemory<float>(*output);
  auto* stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));
  // TODO(huangyp): share/use autotune cublas algorithms in Matmul.op.
  bool blas_launch_status =
      stream
          ->ThenBlasGemm(no_transpose, no_transpose, n, m, k, 1.0f, b_ptr, n,
                         a_ptr, k, 0.0f, &c_ptr, n)
          .ok();
  if (!blas_launch_status) {
    ctx->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                    ", n=", n, ", k=", k));
  }
}
}  // namespace functor
}  // end namespace addons
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
