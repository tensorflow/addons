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

#define EIGEN_USE_THREADS

#include "tensorflow_addons/custom_ops/activations/cc/kernels/softshrink_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace addons {

using CPUDevice = Eigen::ThreadPoolDevice;

#define REGISTER_SOFTSHRINK_KERNELS(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Addons>Softshrink").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SoftshrinkOp<CPUDevice, type>);                                         \
  REGISTER_KERNEL_BUILDER(Name("Addons>SoftshrinkGrad")                       \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<type>("T"),                     \
                          SoftshrinkGradOp<CPUDevice, type>);

// Softshrink only makes sense with floating points.
TF_CALL_GPU_NUMBER_TYPES(REGISTER_SOFTSHRINK_KERNELS);
#undef REGISTER_SOFTSHRINK_KERNELS

#if GOOGLE_CUDA

using GPUDevice = Eigen::GpuDevice;

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                  \
  template <>                                                                \
  void Softshrink<GPUDevice, T>::operator()(                                 \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features, T lower, \
      T upper, typename TTypes<T>::Tensor activations);                      \
  extern template struct Softshrink<GPUDevice, T>;                           \
                                                                             \
  template <>                                                                \
  void SoftshrinkGrad<GPUDevice, T>::operator()(                             \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,         \
      typename TTypes<T>::ConstTensor features, T lower, T upper,            \
      typename TTypes<T>::Tensor backprops);                                 \
  extern template struct SoftshrinkGrad<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_SOFTSHRINK_GPU_KERNELS(type)                                 \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("Addons>Softshrink").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SoftshrinkOp<GPUDevice, type>);                                         \
  REGISTER_KERNEL_BUILDER(Name("Addons>SoftshrinkGrad")                       \
                              .Device(DEVICE_GPU)                             \
                              .TypeConstraint<type>("T"),                     \
                          SoftshrinkGradOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_SOFTSHRINK_GPU_KERNELS);
#undef REGISTER_SOFTSHRINK_GPU_KERNELS

#endif  // GOOGLE_CUDA

}  // namespace addons
}  // namespace tensorflow
