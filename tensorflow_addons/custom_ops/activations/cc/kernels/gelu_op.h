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

#ifndef TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_GELU_OP_H_
#define TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_GELU_OP_H_

#define EIGEN_USE_THREADS
#include <cmath>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace addons {
namespace functor {

// Functor used by GeluOp to do the computations.
template <typename Device, typename T>
struct Gelu {
  // Computes Gelu activation.
  //
  // features: any shape.
  // approximate: whether to enable approximation.
  // activations: same shape as "features".
  void operator()(const Device& d, typename TTypes<T>::ConstTensor features,
                  bool approximate, typename TTypes<T>::Tensor activations) {
    if (approximate) {
      // y = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
      activations.device(d) =
          static_cast<T>(0.5) * features *
          (static_cast<T>(1) +
           (static_cast<T>(M_2_SQRTPI * M_SQRT1_2) *
            (features + static_cast<T>(0.044715) * features.cube()))
               .tanh());
    } else {
      // y = x * normcdf(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
      activations.device(d) =
          static_cast<T>(0.5) * features *
          (static_cast<T>(1) + (features * static_cast<T>(M_SQRT1_2)).erf());
    }
  }
};

// Functor used by GeluGradOp to do the computations.
template <typename Device, typename T>
struct GeluGrad {
  // Computes GeluGrad backprops.
  //
  // gradients: gradients backpropagated to the Gelu op.
  // features: inputs that were passed to the Gelu op.
  // approximate: whether to enable approximation.
  // backprops: gradients to backpropagate to the Gelu inputs.
  void operator()(const Device& d, typename TTypes<T>::ConstTensor gradients,
                  typename TTypes<T>::ConstTensor features, bool approximate,
                  typename TTypes<T>::Tensor backprops) {
    if (approximate) {
      const T kAlpha = static_cast<T>(M_2_SQRTPI * M_SQRT1_2);
      const T kBeta = kAlpha * static_cast<T>(0.044715) * static_cast<T>(3);
      const auto y =
          (kAlpha * ((static_cast<T>(0.044715) * features.cube()) + features))
              .tanh();
      backprops.device(d) = ((-features * y.square() + features) *
                                 (kBeta * features.square() + kAlpha) +
                             static_cast<T>(1) + y) *
                            gradients * static_cast<T>(0.5);
    } else {
      backprops.device(d) =
          gradients * (static_cast<T>(M_2_SQRTPI * M_SQRT1_2 * 0.5) * features *
                           (-features.square() * static_cast<T>(0.5)).exp() +
                       (static_cast<T>(0.5) *
                        (static_cast<T>(1) +
                         (features * static_cast<T>(M_SQRT1_2)).erf())));
    }
  }
};

}  // namespace functor

template <typename Device, typename T>
class GeluOp : public UnaryElementWiseOp<T, GeluOp<Device, T>> {
 public:
  explicit GeluOp(OpKernelConstruction* context)
      : UnaryElementWiseOp<T, GeluOp<Device, T>>::UnaryElementWiseOp(context) {
    OP_REQUIRES_OK(context, context->GetAttr("approximate", &approximate_));
  }

  void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
    functor::Gelu<Device, T> functor;
    functor(context->eigen_device<Device>(), input.flat<T>(), approximate_,
            output->flat<T>());
  }

 private:
  bool approximate_;
};

template <typename Device, typename T>
class GeluGradOp : public BinaryElementWiseOp<T, GeluGradOp<Device, T>> {
 public:
  explicit GeluGradOp(OpKernelConstruction* context)
      : BinaryElementWiseOp<T, GeluGradOp<Device, T>>::BinaryElementWiseOp(
            context) {
    OP_REQUIRES_OK(context, context->GetAttr("approximate", &approximate_));
  }

  void OperateNoTemplate(OpKernelContext* context, const Tensor& g,
                         const Tensor& a, bool approximate, Tensor* output);

  template <int NDIMS>
  void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
               Tensor* output) {
    OperateNoTemplate(context, g, a, approximate_, output);
  }

 private:
  bool approximate_;
};

template <typename Device, typename T>
void GeluGradOp<Device, T>::OperateNoTemplate(OpKernelContext* context,
                                              const Tensor& g, const Tensor& a,
                                              bool approximate,
                                              Tensor* output) {
  functor::GeluGrad<Device, T> functor;
  functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
          approximate, output->flat<T>());
}

}  // namespace addons
}  // namespace tensorflow

#undef EIGEN_USE_THREADS

#endif  // TENSORFLOW_ADDONS_ACTIVATIONS_KERNELS_GELU_OP_H_
