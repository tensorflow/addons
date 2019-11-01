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
    // softplus implementation
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/softplus_op.h
    static const T threshold =
        Eigen::numext::log(Eigen::NumTraits<T>::epsilon()) + T(2);
    const auto& too_large = features > features.constant(-threshold);
    const auto& too_small = features < features.constant(threshold);
    const auto& features_exp = features.exp();
    const auto& sp = too_large.select(
        features,
        too_small.select(features_exp,
                         (features_exp + features.constant(T(1))).log()));
    activations.device(d) = features * sp.tanh();
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
    // softplus implementation
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/softplus_op.h
    static const T threshold =
        Eigen::numext::log(Eigen::NumTraits<T>::epsilon()) + T(2);
    const auto& too_large = features > features.constant(-threshold);
    const auto& too_small = features < features.constant(threshold);
    const auto& features_exp = features.exp();
    const auto& sp = too_large.select(
        features,
        too_small.select(features_exp,
                         (features_exp + features.constant(T(1))).log()));

    const auto& grad_sp = static_cast<T>(1) - (-sp).exp();
    const auto& tsp = sp.tanh();
    const auto& grad_tsp = ((static_cast<T>(1) - tsp * tsp) * grad_sp);
    const auto& grad = features * grad_tsp + tsp;
    backprops.device(d) = gradients * grad;
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
