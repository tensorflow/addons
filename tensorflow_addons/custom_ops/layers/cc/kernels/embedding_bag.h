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

#ifndef TENSORFLOW_ADDONS_LAYERS_KERNELS_EMBEDDING_BAG_H_
#define TENSORFLOW_ADDONS_LAYERS_KERNELS_EMBEDDING_BAG_H_

#if PLATFORM_WINDOWS
#define __restrict__ __restrict
#endif

namespace tensorflow {
namespace addons {
namespace functor {

template <typename Device, typename T, typename Tindices>
struct EmbeddingBagFunctor {
  void operator()(const Device& d, const Eigen::Index value_dim,
                  const Eigen::Index sequence_length,
                  const Eigen::Index num_bags,
                  const Tindices* __restrict__ indices,
                  const T* __restrict__ values, const T* __restrict__ weights,
                  T* __restrict__ output);
};

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif  // TENSORFLOW_ADDONS_LAYERS_KERNELS_EMBEDDING_BAG_H_
