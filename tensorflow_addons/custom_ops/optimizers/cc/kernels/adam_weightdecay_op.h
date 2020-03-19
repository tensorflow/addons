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

#ifndef TENSORFLOW_ADDONS_OPTIMIZER_KERNEL_LAMB_OP_H_
#define TENSORFLOW_ADDONS_OPTIMIZER_KERNEL_LAMB_OP_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace addons {
namespace functor {

template <typename Device, typename T>
struct AdamWithWeightdecay {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat var,
                  typename TTypes<T>::Flat var_update,
                  typename TTypes<T>::ConstFlat m,
                  typename TTypes<T>::Flat out_m,
                  typename TTypes<T>::ConstFlat v,
                  typename TTypes<T>::Flat out_v,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar wd,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad, bool use_nesterov);
};

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif
