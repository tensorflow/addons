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

#ifndef TENSORFLOW_ADDONS_GELU_OP_H_
#define TENSORFLOW_ADDONS_GELU_OP_H_

#define EIGEN_USE_THREADS

#include "tensorflow_addons/custom_ops/activations/cc/kernels/gelu_op_functor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

template <typename Device, typename T>
class GeluOp : public UnaryElementWiseOp<T, GeluOp<Device, T>> {
    public:
        using UnaryElementWiseOp<T, GeluOp<Device, T>>::UnaryElementWiseOp;

        void Operate(OpKernelContext* context, const Tensor& input, Tensor* output) {
            functor::Gelu<Device, T> functor;
            functor(context->eigen_device<Device>(), input.flat<T>(), output->flat<T>());
        }
};

template <typename Device, typename T>
class GeluGradOp : public BinaryElementWiseOp<T, GeluGradOp<Device, T>> {
    public:
        using BinaryElementWiseOp<T, GeluGradOp<Device, T>>::BinaryElementWiseOp;

        void OperateNoTemplate(OpKernelContext* context, const Tensor& g,
                               const Tensor& a, Tensor* output);

        template <int NDIMS>
        void Operate(OpKernelContext* context, const Tensor& g, const Tensor& a,
                     Tensor* output) {
            OperateNoTemplate(context, g, a, output);
        }
};

template <typename Device, typename T>
void GeluGradOp<Device, T>::OperateNoTemplate(OpKernelContext* context,
                                              const Tensor& g, const Tensor& a,
                                              Tensor* output) {
    functor::GeluGrad<Device, T> functor;
    functor(context->eigen_device<Device>(), g.flat<T>(), a.flat<T>(),
            output->flat<T>());
}

} // namespace tensorflow

#undef EIGEN_USE_THREADS

#endif // TENSORFLOW_ADDONS_GELU_OP_H_
