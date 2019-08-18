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

#include <cmath>
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "tensorflow_addons/custom_ops/activations/cc/kernels/gelu_op_functor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
__global__ void ApproximateGeluKernel(const int32 count, const T* input,
                                      T* output) {
  // output[i] = 0.5x * (1 + tanh(sqrt(2 / PI) * (x + 0.044715 * x^3)))
  const T kAlpha = static_cast<T>(M_2_SQRTPI * M_SQRT1_2);
  GPU_1D_KERNEL_LOOP(i, count) {
    T x = input[i];
    output[i] = static_cast<T>(0.5) * x *
                (static_cast<T>(1) +
                 tanh(kAlpha * (x + static_cast<T>(0.044715) * (x * x * x))));
  }
}

template <typename T>
__global__ void GeluKernel(const int32 count, const T* input, T* output) {
  // output[i] = x * P(X <= x) = x * normcdf(x) = 0.5x * (1 + erf(x / sqrt(2))
  GPU_1D_KERNEL_LOOP(i, count) {
    T x = input[i];
    output[i] =
        static_cast<T>(0.5) * x *
        (static_cast<T>(1) + Eigen::numext::erf(x * static_cast<T>(M_SQRT1_2)));
  }
}

template <typename T>
__global__ void ApproximateGeluGradKernel(const int32 count, const T* gradients,
                                          const T* features, T* backprops) {
  const T kAlpha = static_cast<T>(M_2_SQRTPI * M_SQRT1_2);
  const T kBeta = kAlpha * static_cast<T>(0.044715) * static_cast<T>(3);
  GPU_1D_KERNEL_LOOP(i, count) {
    T x = features[i];
    const T y = tanh(kAlpha * ((static_cast<T>(0.044715) * x * x * x) + x));
    backprops[i] = ((-x * (y * y) + x) * (kBeta * x * x + kAlpha) +
                    static_cast<T>(1) + y) *
                   gradients[i] * static_cast<T>(0.5);
  }
}

template <typename T>
__global__ void GeluGradKernel(const int32 count, const T* gradients,
                               const T* features, T* backprops) {
  const T kAlpha = static_cast<T>(M_2_SQRTPI * M_SQRT1_2 * 0.5);
  GPU_1D_KERNEL_LOOP(i, count) {
    T x = features[i];
    backprops[i] =
        gradients[i] * (kAlpha * x * exp(-x * x * static_cast<T>(0.5)) +
                        (static_cast<T>(0.5) *
                         (static_cast<T>(1) +
                          Eigen::numext::erf(x * static_cast<T>(M_SQRT1_2)))));
  }
}

template <typename T>
struct Gelu<GPUDevice, T> {
  // Computes Gelu activation.
  //
  // features: any shape.
  // approximate: whether to enable approximation.
  // activations: same shape as "features".
  void operator()(const GPUDevice& d, typename TTypes<T>::ConstTensor features,
                  bool approximate, typename TTypes<T>::Tensor activations) {
    const int32 count = features.size();
    if (count == 0) return;

    auto kernel = approximate ? ApproximateGeluKernel<T> : GeluKernel<T>;

    GpuLaunchConfig config = GetGpuLaunchConfig(count, d, kernel, 0, 0);

    TF_CHECK_OK(GpuLaunchKernel(kernel, config.block_count,
                                config.thread_per_block, 0, d.stream(), count,
                                features.data(), activations.data()));
  }
};

template <typename T>
struct GeluGrad<GPUDevice, T> {
  // Computes GeluGrad backprop.
  //
  // gradients: gradient backpropagated to the Gelu op.
  // features: either the inputs that were passed to the Gelu, or its outputs
  //           (using either one yields the same result here).
  // approximate: whether to enable approximation.
  // backprops: gradient to backpropagate to the Gelu inputs.
  void operator()(const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor features, bool approximate,
                  typename TTypes<T>::Tensor backprops) {
    const int32 count = gradients.size();
    if (count == 0) return;

    auto kernel =
        approximate ? ApproximateGeluGradKernel<T> : GeluGradKernel<T>;

    GpuLaunchConfig config = GetGpuLaunchConfig(count, d, kernel, 0, 0);

    TF_CHECK_OK(GpuLaunchKernel(
        kernel, config.block_count, config.thread_per_block, 0, d.stream(),
        count, gradients.data(), features.data(), backprops.data()));
  }
};

}  // namespace functor

#define DEFINE_GPU_KERNELS(T)                  \
  template struct functor::Gelu<GPUDevice, T>; \
  template struct functor::GeluGrad<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
