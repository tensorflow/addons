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

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow_addons/custom_ops/image/cc/kernels/euclidean_distance_transform_op.h"

namespace tensorflow {
namespace addons {

namespace functor {

template <typename T>
__global__ void EuclideanDistanceTransformGPUKernel(
    const uint8 *__restrict__ input_ptr, T *__restrict__ output_ptr,
    const int batch_size, const int height, const int width,
    const int channels) {
  for (int index : GpuGridRangeX<int>(batch_size * channels)) {
    int batch_id = index / channels;
    int channel = index % channels;
    EuclideanDistanceTransformSample<T>(input_ptr, output_ptr, batch_id,
                                        channel, height, width, channels);
  }
}

template <typename T>
struct EuclideanDistanceTransformFunctor<GPUDevice, T> {
  typedef typename TTypes<uint8, 4>::ConstTensor InputType;
  typedef typename TTypes<T, 4>::Tensor OutputType;

  void operator()(OpKernelContext *ctx, OutputType *output,
                  const InputType &images) const {
    auto d = ctx->eigen_device<GPUDevice>();
    GpuLaunchConfig config =
        GetGpuLaunchConfig(images.dimension(0) * images.dimension(3), d,
                           EuclideanDistanceTransformGPUKernel<T>, 0, 256);
    TF_CHECK_OK(GpuLaunchKernel(EuclideanDistanceTransformGPUKernel<T>,
                                config.block_count, config.thread_per_block, 0,
                                d.stream(), images.data(), output->data(),
                                images.dimension(0), images.dimension(1),
                                images.dimension(2), images.dimension(3)));
  }
};

template struct EuclideanDistanceTransformFunctor<GPUDevice, Eigen::half>;
template struct EuclideanDistanceTransformFunctor<GPUDevice, float>;
template struct EuclideanDistanceTransformFunctor<GPUDevice, double>;

}  // end namespace functor

}  // end namespace addons
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
