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

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/sort.h>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow_addons/custom_ops/layers/cc/kernels/embedding_bag_ops.h"

constexpr int MAX_THREADS_PER_BLOCK = 1024;

namespace tensorflow {
namespace addons {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename Tindices, const int kThreadsPerBlock>
__global__ void PrepTempArraysKernel(
    const Tindices *__restrict__ indices, Tindices *__restrict__ sortedIndices,
    Tindices *__restrict__ sortedIndicesCounter, const int indices_size) {
  const int arrayIdx = (blockIdx.x * kThreadsPerBlock) + threadIdx.x;
  if (arrayIdx <
      indices_size) {  // Make sure we don't run off the end of the actual array
    sortedIndices[arrayIdx] = indices[arrayIdx];
    sortedIndicesCounter[arrayIdx] = arrayIdx;
  }
}

// Define the CUDA kernel.
template <typename T, typename Tindices, const int kThreadsPerBlock>
__global__ void EmbeddingBagWeightsGradKernel(
    const int value_dim, const Tindices *__restrict__ indices,
    const T *__restrict__ values, const T *__restrict__ dloss,
    T *__restrict__ weights_grad, Combiner combiner) {
  const int sample_idx = blockIdx.x;
  const int bag_idx = blockIdx.y;
  const int bag_dim = gridDim.y;
  const int valueBaseIdx =
      indices[(sample_idx * bag_dim) + bag_idx] * value_dim;
  const int dlossBaseIdx = sample_idx * value_dim;
  // Use a full-precision accumulator even for half-precision inputs
  float partialDotProduct = 0.0f;
  for (int i = threadIdx.x; i < value_dim;
       i += blockDim.x)  // Note that some threads may stop one iteration
                         // earlier if the block straddles the end of the array
  {
    partialDotProduct +=
        static_cast<float>(values[valueBaseIdx + i] * dloss[dlossBaseIdx + i]);
  }
  unsigned activeMask = 0xffffffff;
#pragma unroll
  for (int offset = kThreadsPerBlock / 2; offset > 0; offset /= 2) {
    partialDotProduct +=
        __shfl_down_sync(activeMask, partialDotProduct, offset);
  }
  if (combiner == Combiner::kMean) {
    partialDotProduct /= static_cast<float>(bag_dim);
  }
  // Thread 0 now has the full dot product
  if (threadIdx.x == 0) {
    weights_grad[(sample_idx * bag_dim) + bag_idx] =
        static_cast<T>(partialDotProduct);
  }
}

template <typename T, typename Tindices>
__global__ void EmbeddingBagValuesGradKernel(
    const int value_dim, const int bag_dim,
    const Tindices *__restrict__ sortedIndices,
    const Tindices *__restrict__ counter, const T *__restrict__ values,
    const T *__restrict__ weights, const T *__restrict__ dloss,
    T *__restrict__ values_grad, Combiner combiner) {
  const int startIdx = blockIdx.x;
  const int chunk = blockIdx.y;
  const int kThreadsPerBlock = blockDim.x;
  const int featureIdx = threadIdx.x + (chunk * kThreadsPerBlock);
  // The core problem here is that we want to avoid parallel writes to the
  // same element of the grads. We avoid that by pre-sorting a copy of the
  // indices tensor, and also co-sorting a 'counter' array so that we still know
  // which element of the incoming gradient tensor corresponds to each. Then, we
  // take the slightly lazy approach of spinning up a warp for each element of
  // the indices array, but having each warp check the previous element before
  // it starts. If the two elements are the same, then the warp immediately
  // returns without doing anything. If not, then the warp iterates forward and
  // accumulates gradient until it hits a different index element, at which
  // point it writes the accumulated value and returns. This ensures that each
  // row of the values grad tensor is handled by one and exactly one warp.
  const int valuesIdx = ldg(sortedIndices + startIdx);
  if (startIdx > 0) {
    const int prevIdx = ldg(sortedIndices + startIdx - 1);
    if (prevIdx == valuesIdx) {
      return;  // Another block is handling this index, exit
    }
  }
  int endIdx = startIdx;
  while (endIdx < gridDim.x - 1)  // Don't run off the end of the array
  {
    int nextIdx = endIdx + 1;
    int nextValuesIdx = ldg(sortedIndices + nextIdx);
    if (nextValuesIdx == valuesIdx) {
      endIdx += 1;
    } else {
      break;
    }
  }
  if (featureIdx < value_dim)  // Don't run off the end of the row
  {
    const int outputOffset = (valuesIdx * value_dim) + featureIdx;
    float accum = 0.0f;  // Full precision even if the inputs aren't

    for (int currentIdx = startIdx; currentIdx <= endIdx; ++currentIdx) {
      int originalIdxPosition = ldg(counter + currentIdx);
      T weight = weights[originalIdxPosition];
      // The floor division on this line is correct and intentional
      T featureDloss =
          ldg(dloss + (originalIdxPosition / bag_dim) + featureIdx);
      accum += static_cast<float>(weight * featureDloss);
    }
    if (combiner == Combiner::kMean) {
      accum /= static_cast<float>(bag_dim);
    }
    values_grad[outputOffset] = static_cast<T>(accum);
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T, typename Tindices>
struct EmbeddingBagBackwardFunctor<GPUDevice, T, Tindices> {
  // indices should remain unchanged, but thrust complains if it's a const
  // pointer
  void operator()(const GPUDevice &d,
                  typename TTypes<Tindices, 2>::ConstTensor indices,
                  typename TTypes<T, 2>::ConstTensor params,
                  typename TTypes<T, 2>::ConstTensor weights,
                  typename TTypes<T, 2>::ConstTensor grads,
                  typename TTypes<T, 2>::Tensor params_grads,
                  typename TTypes<T, 2>::Tensor weights_grads,
                  Combiner combiner, OpKernelContext *context) {
    // I copy-pasted this bit from histogram_op_gpu.cu.cc and I sure hope it
    // works
    tensorflow::AllocatorAttributes gpu_allocator;
    gpu_allocator.set_on_host(false);
    gpu_allocator.set_gpu_compatible(true);

    Tensor sortedIndicesTensor;
    Tensor sortedIndicesCounterTensor;

    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<Tindices>::value,
                                          TensorShape({indices.size()}),
                                          &sortedIndicesTensor, gpu_allocator));
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<Tindices>::value,
                                TensorShape({indices.size()}),
                                &sortedIndicesCounterTensor, gpu_allocator));
    auto sortedIndices = sortedIndicesTensor.flat<Tindices>();
    auto sortedIndicesCounter = sortedIndicesCounterTensor.flat<Tindices>();
    // Note: I tried splitting the two kernels into different streams but
    // performance was barely affected.
    const Eigen::Index batch_dim = indices.dimension(0);
    const Eigen::Index bag_dim = indices.dimension(1);
    const Eigen::Index output_dim = params.dimension(1);
    const auto params_size = params.size();
    const int kThreadsPerBlock = 32;
    dim3 gridShape = dim3(batch_dim, bag_dim, 1);
    TF_CHECK_OK(GpuLaunchKernel(
        EmbeddingBagWeightsGradKernel<T, Tindices, kThreadsPerBlock>, gridShape,
        kThreadsPerBlock, 0, d.stream(), output_dim, indices.data(),
        params.data(), grads.data(), weights_grads.data(), combiner));

    const int indices_size = indices.size();
    const int values_size = params.size();
    const int total_blocks = Eigen::divup(indices_size, kThreadsPerBlock);
    gridShape = dim3(total_blocks, 1, 1);

    TF_CHECK_OK(GpuLaunchKernel(
        PrepTempArraysKernel<Tindices, kThreadsPerBlock>, gridShape,
        kThreadsPerBlock, 0, d.stream(), indices.data(), sortedIndices.data(),
        sortedIndicesCounter.data(), indices_size));

    thrust::device_ptr<Tindices> sortedIndicesCounterDevicePtr(
        sortedIndicesCounter.data());
    thrust::device_ptr<Tindices> sortedIndicesDevicePtr(sortedIndices.data());
    thrust::device_ptr<T> paramsGradDevicePtr(params_grads.data());
    thrust::fill(paramsGradDevicePtr,
                 paramsGradDevicePtr + static_cast<int>(params_size),
                 static_cast<T>(0.0f));
    thrust::sort_by_key(sortedIndicesDevicePtr,
                        sortedIndicesDevicePtr + indices_size,
                        sortedIndicesCounterDevicePtr);
    // Handle each row with as few thread blocks as possible
    int threadsPerBlock;
    int blocksPerRow;
    if (output_dim <= MAX_THREADS_PER_BLOCK) {
      blocksPerRow = 1;
      threadsPerBlock = output_dim;
    } else {
      blocksPerRow =
          Eigen::divup(static_cast<int>(output_dim), MAX_THREADS_PER_BLOCK);
      threadsPerBlock =
          Eigen::divup(static_cast<int>(output_dim), blocksPerRow);
    }
    // int blocksPerRow = 1;
    // while (threadsPerBlock > MAX_THREADS_PER_BLOCK) {
    //   threadsPerBlock = (threadsPerBlock + 1) / 2;  // Ceiling division
    //   blocksPerRow *= 2;
    // }
    gridShape = dim3(indices_size, blocksPerRow, 1);
    TF_CHECK_OK(GpuLaunchKernel(
        EmbeddingBagValuesGradKernel<T, Tindices>, gridShape, threadsPerBlock,
        0, d.stream(), output_dim, bag_dim, sortedIndices.data(),
        sortedIndicesCounter.data(), params.data(), weights.data(),
        grads.data(), params_grads.data(), combiner));
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct EmbeddingBagBackwardFunctor<GPUDevice, double, int32>;
template struct EmbeddingBagBackwardFunctor<GPUDevice, float, int32>;
template struct EmbeddingBagBackwardFunctor<GPUDevice, Eigen::half, int32>;
template struct EmbeddingBagBackwardFunctor<GPUDevice, double, int64>;
template struct EmbeddingBagBackwardFunctor<GPUDevice, float, int64>;
template struct EmbeddingBagBackwardFunctor<GPUDevice, Eigen::half, int64>;
}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
