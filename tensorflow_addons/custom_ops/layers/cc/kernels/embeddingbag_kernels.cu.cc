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

#include "embeddingbag.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

constexpr int MAX_THREADS_PER_BLOCK = 1024;

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.
template <typename T_indices>
__global__ void EmbeddingBagCudaKernel(const int value_dim, const int bag_dim, const int indices_size, const T_indices* __restrict__ indices, const float* __restrict__ values, const float* __restrict__ weights, float* __restrict__ out) {
     // blockIdx.x indicates which row of the output we are writing to
     //            It also indicates which 'bag' we're reading from
     // blockIdx.y indicates which chunk of that row we are writing to
     // threadIdx.x indicates which element of that chunk we are writing to

     // feature_idx is the position in the final dimension of the output that we are writing to
     const int feature_idx = blockIdx.y * blockDim.x + threadIdx.x;
     if (feature_idx < value_dim)  // necessary in case value_dim is not evenly divided by blockDim.x
     {
       // out_idx is the offset of the output we are writing to
       const int out_idx = blockIdx.x * value_dim + feature_idx;
       // bag_start_offset is the offset in indices corresponding to the first index of the 'bag'
       // that we will be summing over
       const int bag_start_offset = blockIdx.x * bag_dim;
       float accum = 0.0f;
       for (int idx_offset = bag_start_offset; idx_offset < bag_start_offset + bag_dim; ++idx_offset)
       {
         accum += values[indices[idx_offset] * value_dim + feature_idx] * weights[idx_offset];
       }
         out[out_idx] = accum;
     }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T_indices>
struct EmbeddingBagFunctor<GPUDevice, T_indices> {
  void operator()(const GPUDevice& d, const int value_dim, const int bag_dim, const int indices_size, const T_indices* indices, const float* values, const float* weights, float* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int threadsPerBlock = 32;
    int blocksPerValueVec = (value_dim + threadsPerBlock - 1) / threadsPerBlock; // ceiling division

    int block_x = indices_size / bag_dim;
    dim3 gridShape = dim3(block_x, blocksPerValueVec, 1);
    // gridDim.X indicates which row of the output we are writing to
    // gridDim.Y indicates which 'chunk' of that row we are writing to
    // blockDim.X indicates where we are within that chunk
    EmbeddingBagCudaKernel<T_indices>
        <<<gridShape, threadsPerBlock, 0, d.stream()>>>(value_dim, bag_dim, indices_size, indices, values, weights, out);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct EmbeddingBagFunctor<GPUDevice, int32>;
template struct EmbeddingBagFunctor<GPUDevice, int64>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
