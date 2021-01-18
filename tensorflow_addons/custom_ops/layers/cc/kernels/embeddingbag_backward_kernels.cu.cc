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

#include "embeddingbag_backward.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/fill.h>

constexpr int MAX_THREADS_PER_BLOCK = 1024;

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;


template <typename T_indices>
__global__ void PrepTempArraysKernel(const T_indices* indices, T_indices* sortedIndices, T_indices* sortedIndicesCounter, const int indices_size) {
  const int arrayIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (arrayIdx < indices_size) {  // Make sure we don't run off the end of the actual array
    sortedIndices[arrayIdx] = indices[arrayIdx];
    sortedIndicesCounter[arrayIdx] = arrayIdx;
  }
}


// Define the CUDA kernel.
template <typename T_indices>
__global__ void EmbeddingBagWeightsGradKernel(const int value_dim, const T_indices* indices, const float* values, const float* dloss, float* weights_grad) {
  const int sample_idx = blockIdx.x;
  const int bag_idx = blockIdx.y;
  const int bag_dim = gridDim.y;
  const int valueBaseIdx = indices[(sample_idx * bag_dim) + bag_idx] * value_dim;
  const int dlossBaseIdx = sample_idx * value_dim;
  // Let's just start with the weights/scores grad
  // It has the same shape as indices, i.e. (batch_dims, bag_dim)
  // To compute one element of this, we need to dot product the corresponding
  // value slice and loss grad slice
  float partialDotProduct = 0.0f;
  for (int i = threadIdx.x; i < value_dim; i += blockDim.x) // Note that some threads may stop one iteration earlier if the block straddles the end of the array
  {
    partialDotProduct += values[valueBaseIdx + i] * dloss[dlossBaseIdx + i];
  }
  unsigned activeMask = 0xffffffff;
  for (int offset = blockDim.x/2; offset > 0; offset /= 2)
  {
    partialDotProduct += __shfl_down_sync(activeMask, partialDotProduct, offset);
  }
  // Thread 0 now has the full dot product
  if (threadIdx.x == 0)
  {
    weights_grad[(sample_idx * bag_dim) + bag_idx] = partialDotProduct;
  }
}

template <typename T_indices>
__global__ void EmbeddingBagValuesGradKernel(const int value_dim, const int bag_dim, const T_indices* __restrict__ sortedIndices, const T_indices* __restrict__ counter, const float* __restrict__ values, const float* __restrict__ weights, const float* __restrict__ dloss, float* __restrict__ values_grad)
{
  const int startIdx = blockIdx.x;
  const int chunk = blockIdx.y;
  const int threadsPerChunk = blockDim.x;
  const int featureIdx = threadIdx.x + (chunk * threadsPerChunk);
  const int valuesIdx = ldg(sortedIndices + startIdx);
  if (startIdx > 0)
  {
    const int prevIdx = ldg(sortedIndices + startIdx - 1);
    if (prevIdx == valuesIdx)
    {
      return;  // Another block is handling this index, exit
    }
  }
  int endIdx = startIdx;
  while(endIdx < gridDim.x - 1)  // Don't run off the end of the array
  {
    int nextIdx = endIdx + 1;
    int nextValuesIdx = ldg(sortedIndices + nextIdx);
    if (nextValuesIdx == valuesIdx)
    {
      endIdx += 1;
    }
    else
    {
      break;
    }
  }
  if (featureIdx < value_dim) // Don't run off the end of the row
  {
    const int outputOffset = (valuesIdx * value_dim) + featureIdx;
    float accum = 0.0f;

    for (int currentIdx = startIdx; currentIdx <= endIdx; ++currentIdx)
    {
      int originalIdxPosition = ldg(counter + currentIdx);
      float weight = weights[originalIdxPosition];
      // The floor division on this line is correct and intentional
      float featureDloss = ldg(dloss + (originalIdxPosition / bag_dim) + featureIdx);
      accum += weight * featureDloss;
    }
    values_grad[outputOffset] = accum;
  }
}


// Define the GPU implementation that launches the CUDA kernel.
template <typename T_indices>
struct EmbeddingBagBackwardFunctor<GPUDevice, T_indices> {
  // indices should remain unchanged, but thrust complains if it's a const pointer
  void operator()(const GPUDevice& d, const int value_dim, const int bag_dim, const int indices_size, const int values_size,
                  const T_indices* indices, const float* values,
                  const float* weights, const float* dloss, float* values_grad, float* weights_grad, T_indices* sortedIndices, T_indices* sortedIndicesCounter) {


    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    // Note: I tried splitting the two kernels into different streams but performance was barely affected.
    int threadsPerBlock = 32;
    dim3 gridShape = dim3(indices_size / bag_dim, bag_dim, 1);
    EmbeddingBagWeightsGradKernel<T_indices>
        <<<gridShape, threadsPerBlock, 0, d.stream()>>>(value_dim, indices, values, dloss, weights_grad);

    gridShape = dim3((indices_size + threadsPerBlock - 1) / threadsPerBlock, 1, 1);  // Ceiling division
    PrepTempArraysKernel<T_indices>
        <<<gridShape, threadsPerBlock, 0, d.stream()>>>(indices, sortedIndices, sortedIndicesCounter, indices_size);

    thrust::device_ptr<T_indices> sortedIndicesCounterDevicePtr(sortedIndicesCounter);
    thrust::device_ptr<T_indices> sortedIndicesDevicePtr(sortedIndices);
    thrust::device_ptr<float> valuesGradDevicePtr(values_grad);
    thrust::fill(valuesGradDevicePtr, valuesGradDevicePtr + values_size, 0.0f);
    thrust::sort_by_key(sortedIndicesDevicePtr, sortedIndicesDevicePtr + indices_size, sortedIndicesCounterDevicePtr);
    threadsPerBlock = value_dim;
    int blocksPerRow = 1;
    while (threadsPerBlock > MAX_THREADS_PER_BLOCK)
    {
      threadsPerBlock = (threadsPerBlock + 1) / 2; // Ceiling division
      blocksPerRow *= 2;
    }
    gridShape = dim3(indices_size, blocksPerRow, 1);
    EmbeddingBagValuesGradKernel<T_indices>
          <<<gridShape, threadsPerBlock, 0, d.stream()>>>(value_dim, bag_dim, sortedIndices, sortedIndicesCounter, values, weights, dloss, values_grad);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct EmbeddingBagBackwardFunctor<GPUDevice, int32>;
template struct EmbeddingBagBackwardFunctor<GPUDevice, int64>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
