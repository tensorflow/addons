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

#include "tensorflow_addons/custom_ops/activations/cc/kernels/rrelu_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace addons {

using CPUDevice = Eigen::ThreadPoolDevice;

#define REGISTER_RRELU_KERNELS(T)                                         \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Addons>Rrelu").Device(DEVICE_CPU).TypeConstraint<T>("T"),     \
      RreluOp<CPUDevice, T>);                                             \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Addons>RreluGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RreluGradOp<CPUDevice, T>);

// Rrelu only makes sense with floating points.
TF_CALL_GPU_NUMBER_TYPES(REGISTER_RRELU_KERNELS);
#undef REGISTER_RRELU_KERNELS

#if GOOGLE_CUDA

using GPUDevice = Eigen::GpuDevice;

namespace functor {
#define DECLARE_GPU_SPEC(T)                                                   \
  template <>                                                                 \
  void Rrelu<GPUDevice, T>::operator()(                                       \
      const GPUDevice& d, typename TTypes<T>::ConstTensor features, T lower,  \
      T upper, bool training, typename TTypes<T>::Tensor activations,         \
      typename TTypes<T>::Tensor alpha);                                      \
  extern template struct Rrelu<GPUDevice, T>;                                 \
                                                                              \
  template <>                                                                 \
  void RreluGrad<GPUDevice, T>::operator()(                                   \
      const GPUDevice& d, typename TTypes<T>::ConstTensor gradients,          \
      typename TTypes<T>::ConstTensor features,                               \
      typename TTypes<T>::ConstTensor alpha, T lower, T upper, bool training, \
      typename TTypes<T>::Tensor backprops);                                  \
  extern template struct RreluGrad<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_RRELU_GPU_KERNELS(T)                                     \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Addons>Rrelu").Device(DEVICE_GPU).TypeConstraint<T>("T"),     \
      RreluOp<GPUDevice, T>);                                             \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("Addons>RreluGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      RreluGradOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_RRELU_GPU_KERNELS);
#undef REGISTER_RRELU_GPU_KERNELS

#endif  // GOOGLE_CUDA
}  // namespace addons
}  // namespace tensorflow