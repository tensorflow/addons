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

#ifndef TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_SOFTSHRINK_OP_H_
#define TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_SOFTSHRINK_OP_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace addons {

namespace functor {

// Functor used by SoftshrinkOp to do the computations.
template <typename Device, typename T>
struct Softshrink {
  // Computes Softshrink activation.
  //
  // features: any shape.
  // lower: the lower bound for setting values to zeros.
  // upper: the upper bound for setting values to zeros.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  T lower, T upper, typename TTypes<T>::Tensor activations) {
    activations.device(d) =
        (features < lower)
            .select(features - features.constant(lower),
                    (features > upper)
                        .select(features - features.constant(upper),
                                features.constant(static_cast<T>(0))));
  }
};

// Functor used by SoftshrinkGradOp to do the computations.
template <typename Device, typename T>
struct SoftshrinkGrad {
  // Computes SoftshrinkGrad backprops.
  //
  // gradients: gradients backpropagated to the Softshink op.
  // features: inputs that were passed to the Softshrink op.
  // lower: the lower bound for setting values to zeros.
  // upper: the upper bound for setting values to zeros.
  // backprops: gradients to backpropagate to the Softshrink inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor features, T lower, T upper,
                  typename TTypes<T>::Tensor backprops) {
    backprops.device(d) =
        (features < lower || features > upper)
            .select(gradients, features.constant(static_cast<T>(0)));
  }
};

}  // namespace functor

template <typename Device, typename T>
class SoftshrinkOp : public UnaryElementWiseOp<T, SoftshrinkOp<Device, T>> {
 public:
  explicit SoftshrinkOp(OpKernelConstruction* context)
      : UnaryElementWiseOp<T, SoftshrinkOp<Device, T>>::UnaryElementWiseOp(
            context) {
    float lower, upper;
    OP_REQUIRES_OK(context, context->GetAttr("lower", &lower));
    OP_REQUIRES_OK(context, context->GetAttr("upper", &upper));
    lower_ = static_cast<T>(lower);
    upper_ = static_cast<T>(upper);

    OP_REQUIRES(
        context, lower_ <= upper_,
        errors::InvalidArgument("lower must be less than or equal to upper."));
  }

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::Softshrink<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(), lower_, upper_,
            output->flat<T>());
  }

 private:
  T lower_;
  T upper_;
};

template <typename Device, typename T>
class SoftshrinkGradOp
    : public BinaryElementWiseOp<T, SoftshrinkGradOp<Device, T>> {
 public:
  explicit SoftshrinkGradOp(OpKernelConstruction* context)
      : BinaryElementWiseOp<
            T, SoftshrinkGradOp<Device, T>>::BinaryElementWiseOp(context) {
    float lower, upper;
    OP_REQUIRES_OK(context, context->GetAttr("lower", &lower));
    OP_REQUIRES_OK(context, context->GetAttr("upper", &upper));
    lower_ = static_cast<T>(lower);
    upper_ = static_cast<T>(upper);

    OP_REQUIRES(
        context, lower_ <= upper_,
        errors::InvalidArgument("lower must be less than or equal to upper."));
  }

  void OperateNoTemplate(OpKernelContext* context, const Tensor& g,
                         const Tensor& a, T lower, T upper, Tensor* output);

  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OperateNoTemplate(context, g, a, lower_, upper_, output);
  }

 private:
  T lower_;
  T upper_;
};

template <typename Device, typename T>
void SoftshrinkGradOp<Device, T>::OperateNoTemplate(OpKernelContext* context,
                                                    const Tensor& g,
                                                    const Tensor& a, T lower,
                                                    T upper, Tensor* output) {
  functor::SoftshrinkGrad<Device, T> functor;
  functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(), lower,
          upper, output->flat<T>());
}

}  // namespace addons
}  // namespace tensorflow

#undef EIGEN_USE_THREADS

#endif  // TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_SOFTSHRINK_OP_H_
