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

#ifndef TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_RRELU_OP_H_
#define TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_RRELU_OP_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace addons {

namespace functor {

template <typename Device, typename T>
struct Rrelu {
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  T lower, T upper, bool training,
                  typename TTypes<T>::Tensor activations,
                  typename TTypes<T>::Tensor alpha) {
    if (training) {
      alpha.device(d) = alpha.constant(lower) +
                        alpha.random() * alpha.constant(upper - lower);
    } else {
      alpha.device(d) = features.constant((lower + upper) / static_cast<T>(2));
    }
    activations.device(d) =
        (features >= static_cast<T>(0)).select(features, alpha * features);
  }
};

template <typename Device, typename T>
struct RreluGrad {
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor features,
                  typename TTypes<T>::ConstTensor alpha,
                  typename TTypes<T>::Tensor backprops) {
    backprops.device(d) =
        gradients *
        (features >= static_cast<T>(0))
            .select(features.constant(static_cast<T>(1)), alpha);
  }
};

}  // namespace functor

template <typename Device, typename T>
class RreluOp : public OpKernel {
 public:
  explicit RreluOp(OpKernelConstruction* context) : OpKernel(context) {
    float lower, upper;
    OP_REQUIRES_OK(context, context->GetAttr("lower", &lower));
    OP_REQUIRES_OK(context, context->GetAttr("upper", &upper));
    OP_REQUIRES_OK(context, context->GetAttr("training", &training_));
    lower_ = static_cast<T>(lower);
    OP_REQUIRES(context, lower_ >= static_cast<T>(0),
                errors::InvalidArgument("Need lower >= 0, got ", lower_));
    upper_ = static_cast<T>(upper);
    OP_REQUIRES(context, upper_ < static_cast<T>(1),
                errors::InvalidArgument("Need upper < 1, got ", upper_));
    OP_REQUIRES(
        context, lower_ <= upper_,
        errors::InvalidArgument("lower must be less than or equal to upper."));
  }
  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    Tensor* output_tensor = nullptr;
    Tensor* alpha_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, input_tensor.shape(),
                                                     &alpha_tensor));
    functor::Rrelu<Device, T>()(
        context->eigen_device<Device>(), input_tensor.flat<T>(), lower_, upper_,
        training_, output_tensor->flat<T>(), alpha_tensor->flat<T>());
  }

 private:
  T lower_;
  T upper_;
  bool training_;
};

template <typename Device, typename T>
class RreluGradOp : public OpKernel {
 public:
  explicit RreluGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& gradients = context->input(0);
    const Tensor& input_tensor = context->input(1);
    const Tensor& alpha_tensor = context->input(2);
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    functor::RreluGrad<Device, T>()(context->eigen_device<Device>(),
                                    gradients.flat<T>(), input_tensor.flat<T>(),
                                    alpha_tensor.flat<T>(),
                                    output_tensor->flat<T>());
  }
};

}  // namespace addons
}  // namespace tensorflow

#undef EIGEN_USE_THREADS

#endif  // TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_RRELU_OP_H_
