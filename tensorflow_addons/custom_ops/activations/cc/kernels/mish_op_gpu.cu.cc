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
#include "tensorflow_addons/custom_ops/activations/cc/kernels/mish_op.h"
#include "third_party/eigen3/Eigen/Core"

namespace tensorflow {
namespace addons {

using GPUDevice = Eigen::GpuDevice;

#define DEFINE_GPU_KERNELS(T)                  \
  template struct functor::Mish<GPUDevice, T>; \
  template struct functor::MishGrad<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);

}  // namespace addons
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
