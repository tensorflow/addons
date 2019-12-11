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

namespace tensorflow {
namespace addons {

using GPUDevice = Eigen::GpuDevice;

#define REGISTER_FAST_NON_MAX_SUPPRESSION_GPU_KERNELS(T)                       \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("FastNonMaxSuppression").TypeConstraint<T>("T").Device(DEVICE_GPU), \
      FastNonMaxSuppressionOp<GPUDevice, T>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_FAST_NON_MAX_SUPPRESSION_GPU_KERNELS);
#undef REGISTER_FAST_NON_MAX_SUPPRESSION_GPU_KERNELS
}  // namespace addons
}  // namespace tensorflow