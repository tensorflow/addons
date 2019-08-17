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

#include "tensorflow_addons/custom_ops/activations/cc/kernels/gelu_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

#ifdef GOOGLE_CUDA
using GPUDevice = Eigen::GpuDevice;
#endif

#define REGISTER_GELU_KERNELS(type)                                 \
    REGISTER_KERNEL_BUILDER(                                        \
        Name("Gelu").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
        GeluOp<CPUDevice, type>);                                   \
    REGISTER_KERNEL_BUILDER(                                        \
        Name("GeluGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),  \
        GeluGradOp<CPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GELU_KERNELS);
#undef REGISTER_GELU_KERNELS

#ifdef GOOGLE_CUDA

namespace functor {
#define DECLARE_GPU_SPEC(T)                                                             \
    template <>                                                                         \
    void Gelu<GPUDevice, T>::operator()(const GPUDevice& d,                             \
                                        typename TTypes<T>::ConstTensor features,       \
                                        typename TTypes<T>::Tensor activations);        \
    extern template struct Gelu<GPUDevice, T>;                                          \
                                                                                        \
    template <>                                                                         \
    void GeluGrad<GPUDevice, T>::operator()(const GPUDevice& d,                         \
                                            typename TTypes<T>::ConstTensor gradients,   \
                                            typename TTypes<T>::ConstTensor features,      \
                                            typename TTypes<T>::Tensor backprops);       \
    extern template struct GeluGrad<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
} // namespace functor

#define REGISTER_GPU_KERNELS(type)                                  \
    REGISTER_KERNEL_BUILDER(                                        \
        Name("Gelu").Device(DEVICE_GPU).TypeConstraint<type>("T"),  \
        GeluOp<GPUDevice, type>);                                   \
    REGISTER_KERNEL_BUILDER(                                        \
        Name("GeluGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),  \
        GeluGradOp<GPUDevice, type>);                                   

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

#endif // GOOGLE_CUDA

} // namespace tensorflow
