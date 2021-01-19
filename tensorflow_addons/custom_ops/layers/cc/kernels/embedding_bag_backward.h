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

#ifndef TENSORFLOW_ADDONS_LAYERS_KERNELS_EMBEDDING_BACKWARD_H_
#define TENSORFLOW_ADDONS_LAYERS_KERNELS_EMBEDDING_BACKWARD_H_

namespace tensorflow {
namespace addons {
namespace functor {

// template <typename Device, typename T_indices>
template <typename Device, typename T_indices>
struct EmbeddingBagBackwardFunctor {
  void operator()(const Device& d, const int value_dim, const int bag_dim,
                  const int indices_size, const int values_size,
                  const T_indices* indices, const float* values,
                  const float* weights, const float* dloss, float* values_grad,
                  float* weights_grad, T_indices* dummy1, T_indices* dummy2);
};

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif  // TENSORFLOW_ADDONS_LAYERS_KERNELS_EMBEDDING_BACKWARD_H_
