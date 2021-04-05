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

#ifndef TENSORFLOW_ADDONS_LAYERS_KERNELS_EMBEDDING_BAG_OPS_H_
#define TENSORFLOW_ADDONS_LAYERS_KERNELS_EMBEDDING_BAG_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace addons {

enum class Combiner {
  kSum,
  kMean,
};

namespace functor {

template <typename Device, typename T, typename Tindices>
struct EmbeddingBagFunctor {
  void operator()(const Device &device,
                  typename TTypes<Tindices, 2>::ConstTensor indices,
                  typename TTypes<T, 2>::ConstTensor params,
                  typename TTypes<T, 2>::ConstTensor weights,
                  typename TTypes<T, 2>::Tensor output, Combiner combiner);
};

template <typename Device, typename T, typename Tindices>
struct EmbeddingBagBackwardFunctor {
  void operator()(const Device &device,
                  typename TTypes<Tindices, 2>::ConstTensor indices,
                  typename TTypes<T, 2>::ConstTensor params,
                  typename TTypes<T, 2>::ConstTensor weights,
                  typename TTypes<T, 2>::ConstTensor grads,
                  typename TTypes<T, 2>::Tensor params_grads,
                  typename TTypes<T, 2>::Tensor weights_grads,
                  Combiner combiner, OpKernelContext *context);
};

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif  // TENSORFLOW_ADDONS_LAYERS_KERNELS_EMBEDDING_BAG_OPS_H_
