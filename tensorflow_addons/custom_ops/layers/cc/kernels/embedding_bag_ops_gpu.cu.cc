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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow_addons/custom_ops/layers/cc/kernels/embedding_bag_ops.h"

namespace tensorflow {
namespace addons {

typedef Eigen::GpuDevice GPUDevice;

namespace {
// Define the GPU kernel.
template <typename T, typename Tindices, const int kThreadsPerBlock>
__global__ void EmbeddingBagGPUKernel(const Tindices *__restrict__ indices,
                                      const T *__restrict__ params,
                                      const T *__restrict__ weights,
                                      T *__restrict__ output,
                                      const Eigen::Index output_dim,
                                      const Eigen::Index sequence_length,
                                      Combiner combiner) {
  // blockIdx.x indicates which row of the output we are writing to. It also
  // indicates which `bag` we're reading from.
  // blockIdx.y indicates which chunk of that row we are writing to.
  // threadIdx.x indicates which element of that chunk we are writing to.

  // feature_idx is the position in the final dimension of the output that we
  // are writing to.
  const Eigen::Index feature_idx = blockIdx.y * kThreadsPerBlock + threadIdx.x;
  // It's necessary in case output_dim is not evenly divided by blockDim.x.
  if (feature_idx < output_dim) {
    // output_idx is the offset of the output we are writing to.
    const Eigen::Index output_idx = blockIdx.x * output_dim + feature_idx;
    // bag_offset is the offset in indices corresponding to the first
    // index of the `bag` that we will be summing over.
    const Eigen::Index bag_offset = blockIdx.x * sequence_length;
    T accum = static_cast<T>(0);
    for (Eigen::Index idx_offset = bag_offset;
         idx_offset < bag_offset + sequence_length; ++idx_offset) {
      accum += params[indices[idx_offset] * output_dim + feature_idx] *
               weights[idx_offset];
    }
    if (combiner == Combiner::kMean) {
      accum /= static_cast<T>(sequence_length);
    }
    output[output_idx] = accum;
  }
}
}  // namespace

namespace functor {
// Define the GPU implementation that launches the CUDA kernel.
template <typename T, typename Tindices>
struct EmbeddingBagFunctor<GPUDevice, T, Tindices> {
  static constexpr int kThreadsPerBlock = 32;

  void operator()(const GPUDevice &device,
                  typename TTypes<Tindices, 2>::ConstTensor indices,
                  typename TTypes<T, 2>::ConstTensor params,
                  typename TTypes<T, 2>::ConstTensor weights,
                  typename TTypes<T, 2>::Tensor output, Combiner combiner) {
    const Eigen::Index bags = indices.dimension(0);
    const Eigen::Index sequence_length = indices.dimension(1);
    const Eigen::Index output_dim = params.dimension(1);

    const int blocks_per_value_vec =
        Eigen::divup(output_dim, static_cast<Eigen::Index>(kThreadsPerBlock));
    const dim3 grids = dim3(bags, blocks_per_value_vec);

    TF_CHECK_OK(GpuLaunchKernel(
        EmbeddingBagGPUKernel<T, Tindices, kThreadsPerBlock>, grids,
        kThreadsPerBlock, 0, device.stream(), indices.data(), params.data(),
        weights.data(), output.data(), output_dim, sequence_length, combiner));
  }
};

// Explicit instantiation of the GPU functor.
#define DECLARE_GPU_SPECS(T)                                \
  template struct EmbeddingBagFunctor<GPUDevice, T, int32>; \
  template struct EmbeddingBagFunctor<GPUDevice, T, int64>;

DECLARE_GPU_SPECS(Eigen::half);
DECLARE_GPU_SPECS(float);
DECLARE_GPU_SPECS(double);
#undef DECLARE_GPU_SPECS

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
