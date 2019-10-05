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

#ifndef TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_MISH_OP_H_
#define TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_MISH_OP_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace addons {
namespace functor {

// Functor used by MishOp to do the computations.
template <typename Device, typename T>
struct Mish {
  // Computes Mish activation.
  //
  // features: any shape.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor activations) {
    activations.device(d) = features * features.exp().log1p().tanh();
  }
};

// Functor used by MishGradOp to do the computations.
template <typename Device, typename T>
struct MishGrad {
  // Computes MishGrad backprops.
  //
  // gradients: gradients backpropagated to the Mish op.
  // features: inputs that were passed to the Mish op.
  // backprops: gradients to backpropagate to the Mish inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::Tensor backprops) {
    typename TTypes<T>::Tensor& e = features.exp();
    typename TTypes<T>::Tensor& es = e.square();
    typename TTypes<T>::Tensor& omega = static_cast<T>(4) * (features + static_cast<T>(1)) +
                  static_cast<T>(4) * es + e.cube() +
                  e * (static_cast<T>(4) * features + static_cast<T>(6));
    typename TTypes<T>::Tensor& delta = static_cast<T>(2) * e + es + static_cast<T>(2);
    backprops.device(d) = gradients * e * omega / delta.square();
  }
};

}  // namespace functor

template <typename Device, typename T>
class MishOp : public UnaryElementWiseOp<T, MishOp<Device, T>> {
 public:
  explicit MishOp(OpKernelConstruction* context)
      : UnaryElementWiseOp<T, MishOp<Device, T>>::UnaryElementWiseOp(context) {}

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::Mish<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(),
            output->flat<T>());
  }
};

template <typename Device, typename T>
class MishGradOp : public BinaryElementWiseOp<T, MishGradOp<Device, T>> {
 public:
  explicit MishGradOp(OpKernelConstruction* context)
      : BinaryElementWiseOp<T, MishGradOp<Device, T>>::BinaryElementWiseOp(
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
void MishGradOp<Device, T>::OperateNoTemplate(OpKernelContext* context,
                                              const Tensor& g, const Tensor& a,
                                              Tensor* output) {
  functor::MishGrad<Device, T> functor;
  functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
          output->flat<T>());
}

}  // namespace addons
}  // namespace tensorflow

#undef EIGEN_USE_THREADS

#endif  // TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_MISH_OP_H_
