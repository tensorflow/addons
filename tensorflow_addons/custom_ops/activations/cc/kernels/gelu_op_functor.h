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

#ifndef TENSORFLOW_ADDONS_GELU_OP_FUNCTOR_H_
#define TENSORFLOW_ADDONS_GELU_OP_FUNCTOR_H_

#include <cmath>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

// Functor used by GeluOp to do the computations.
template <typename Device, typename T>
struct Gelu {
    // Computes Gelu activation.
    //
    // features: any shape.
    // activations: same shape as "features".
    void operator()(const Device& d,
                    typename TTypes<T>::ConstTensor features,
                    typename TTypes<T>::Tensor activations) {
        const T kAlpha = static_cast<T>(M_2_SQRTPI * M_SQRT1_2);
        activations.device(d) = static_cast<T>(0.5) * features * (static_cast<T>(1) + (kAlpha * (features + static_cast<T>(0.044715) * features.cube())).tanh());
    }
};

// Functor used by GeluGradOp to do the computations.
template <typename Device, typename T>
struct GeluGrad {
    // Computes GeluGrad backprops.
    //
    // gradients: gradients backpropagated to the Gelu op.
    // features: either the inputs that were passed to the Gelu or, or its
    //           outputs (using either one yields the same result here).
    // backprops: gradients to backpropagate to the Gelu inputs.
    void operator()(const Device& d,
                    typename TTypes<T>::ConstTensor gradients,
                    typename TTypes<T>::ConstTensor features,
                    typename TTypes<T>::Tensor backprops) {
        const T kAlpha = static_cast<T>(M_2_SQRTPI * M_SQRT1_2);
        const T kBeta = kAlpha * static_cast<T>(0.044715) * static_cast<T>(3);
        const auto y = (kAlpha * ((static_cast<T>(0.044715) * features.cube()) + features)).tanh();
        backprops.device(d) = ((-features * y.square() + features) * (kBeta * features.square() + kAlpha) + static_cast<T>(1) + y) * gradients * static_cast<T>(0.5);
    }
};

} // namespace functor
} // namespace tensorflow

#endif // TENSORFLOW_ADDONS_GELU_OP_FUNCTOR_H_
