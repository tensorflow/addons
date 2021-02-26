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
    const batch_size, T *input_ptr, const int input_height,
    const int input_width, const int input_channel, T *output_ptr,
    const int output_height, const int output_width, const int output_channel) {
  const typename TTypes<T, 4>::ConstTensor images(
      input_ptr, batch_size, input_height, input_width, input_channel);
  typename TTypes<T, 4>::Tensor output(output_ptr, batch_size, output_height,
                                       output_width, output_channel);
  auto edt_generator =
      EuclideanDistanceTransformGenerator<GPUDevice, T>(images);
  for (int k : GpuGridRangeX<int>(images.dimension(0))) {
    edt_generator(output, k);
  }
}

template <typename T>
struct EuclideanDistanceTransformFunctor<GPUDevice, T> {
  typedef typename TTypes<T, 4>::ConstTensor InputType;
  typedef typename TTypes<T, 4>::Tensor OutputType;

  void operator()(OpKernelContext *ctx, OutputType *output,
                  const InputType &images) const {
    auto d = ctx->eigen_device<GPUDevice>();
    GpuLaunchConfig config = GetGpuLaunchConfig(images.size(), d);
    TF_CHECK_OK(GpuLaunchKernel(
        EuclideanDistanceTransformGPUKernel<T>, config.block_count,
        config.thread_per_block, 0, d.stream(), images.dimension(0),
        images.data(), images.dimension(1), images.dimension(2),
        images.dimension(3), output->data(), output->dimension(1),
        output->dimension(2), output->dimension(3)));
  }
};

template struct EuclideanDistanceTransformFunctor<GPUDevice, Eigen::half>;
template struct EuclideanDistanceTransformFunctor<GPUDevice, float>;
template struct EuclideanDistanceTransformFunctor<GPUDevice, double>;

}  // end namespace functor

}  // end namespace addons
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
