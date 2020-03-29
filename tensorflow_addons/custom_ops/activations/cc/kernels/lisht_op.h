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

#ifndef TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_LISHT_OP_H_
#define TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_LISHT_OP_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace addons {
namespace functor {

// Functor used by LishtOp to do the computations.
template <typename Device, typename T>
struct Lisht {
  // Computes Lisht activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    activations.device(d) = features * features.tanh();
  }
};

// Functor used by LishtGradOp to do the computations.
template <typename Device, typename T>
struct LishtGrad {
  // Computes LishtGrad backprops.
  //
  // gradients: gradients backpropagated to the Lisht op.
  // features: inputs that were passed to the Lisht op.
  // backprops: gradients to backpropagate to the Lisht inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor backprops) {
    const auto g = features.tanh();
    backprops.device(d) =
        gradients * (g + features * (static_cast<T>(1.0) - g.square()));
  }
};

}  // namespace functor

template <typename Device, typename T>
class LishtOp : public UnaryElementWiseOp<T, LishtOp<Device, T>> {
 public:
  explicit LishtOp(OpKernelConstruction* context)
      : UnaryElementWiseOp<T, LishtOp<Device, T>>::UnaryElementWiseOp(context) {
  }

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::Lisht<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(),
            output->flat<T>());
  }
};

template <typename Device, typename T>
class LishtGradOp : public BinaryElementWiseOp<T, LishtGradOp<Device, T>> {
 public:
  explicit LishtGradOp(OpKernelConstruction* context)
      : BinaryElementWiseOp<T, LishtGradOp<Device, T>>::BinaryElementWiseOp(
            context) {}

  void OperateNoTemplate(OpKernelContext* context, const Tensor& g,
                         const Tensor& a, Tensor* output);

  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OperateNoTemplate(context, g, a, output);
  }
};

template <typename Device, typename T>
void LishtGradOp<Device, T>::OperateNoTemplate(OpKernelContext* context,
                                               const Tensor& g, const Tensor& a,
                                               Tensor* output) {
  functor::LishtGrad<Device, T> functor;
  functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
          output->flat<T>());
}

}  // namespace addons
}  // namespace tensorflow

#undef EIGEN_USE_THREADS

#endif  // TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_LISHT_OP_H_
