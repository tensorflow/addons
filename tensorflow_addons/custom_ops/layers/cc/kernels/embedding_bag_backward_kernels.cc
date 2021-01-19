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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow_addons/custom_ops/layers/cc/kernels/embedding_bag_backward.h"

namespace tensorflow {
namespace addons {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// CPU specialization of actual computation.
// template <typename T_indices>
// struct EmbeddingBagBackwardFunctor<CPUDevice, T_indices> {
template <typename T_indices>
struct EmbeddingBagBackwardFunctor<CPUDevice, T_indices> {
  void operator()(const CPUDevice& d, const int value_dim, const int bag_dim,
                  const int indices_size, const int values_size,
                  const T_indices* indices, const float* values,
                  const float* weights, const float* dloss, float* values_grad,
                  float* weights_grad, T_indices* sortedIndices,
                  T_indices* sortedIndicesCounter) {
    for (int i = 0; i < values_size; ++i) {
      values_grad[i] = 0;  // Zero out the values array before we begin - not
                           // every value is written to and I don't know if it's
                           // guaranteed to be zero-initialized
    }
    for (int bag = 0; bag < indices_size / bag_dim; ++bag) {
      int dloss_base = dloss[value_dim * bag];
      for (int i = 0; i < bag_dim; ++i) {
        float accum = 0.0f;
        int current_idx = bag * bag_dim + i;
        float weight = weights[current_idx];
        int value_base = value_dim * indices[current_idx];
        for (int feature = 0; feature < value_dim; ++feature) {
          accum += values[value_base + feature] * dloss[dloss_base + feature];
          values_grad[value_base + feature] +=
              dloss[dloss_base + feature] * weight;
        }
        weights_grad[current_idx] = accum;
      }
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
// template <typename Device, typename T_indices>
template <typename Device, typename T_indices>
class EmbeddingBagBackwardOp : public OpKernel {
 public:
  explicit EmbeddingBagBackwardOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& indices = context->input(0);
    const Tensor& values = context->input(1);
    const Tensor& weights = context->input(2);
    const Tensor& grad = context->input(3);
    TensorShape indicesShape = indices.shape();
    int valuesDim = values.shape().dim_size(1);
    int bagDim = indicesShape.dim_size(indicesShape.dims() - 1);

    // Create an output tensor
    Tensor* values_grad = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, values.shape(), &values_grad));

    Tensor* weights_grad = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, weights.shape(), &weights_grad));

    Tensor* sortedIndicesTemp = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(2, indices.shape(),
                                                     &sortedIndicesTemp));

    Tensor* sortedIndicesCounterTemp = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                                3, indices.shape(), &sortedIndicesCounterTemp));

    OP_REQUIRES(context, indices.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    OP_REQUIRES(context, values.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    EmbeddingBagBackwardFunctor<Device, T_indices>()(
        context->eigen_device<Device>(), valuesDim, bagDim,
        static_cast<int>(indices.NumElements()),
        static_cast<int>(values.NumElements()),
        indices.flat<T_indices>().data(), values.flat<float>().data(),
        weights.flat<float>().data(), grad.flat<float>().data(),
        values_grad->flat<float>().data(), weights_grad->flat<float>().data(),
        sortedIndicesTemp->flat<T_indices>().data(),
        sortedIndicesCounterTemp->flat<T_indices>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T_indices)                                        \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBagGrad")              \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<T_indices>("T_indices"), \
                          EmbeddingBagBackwardOp<CPUDevice, T_indices>);
REGISTER_CPU(int32);
REGISTER_CPU(int64);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T_indices)                                             \
  extern template struct EmbeddingBagBackwardFunctor<GPUDevice, T_indices>; \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBagGrad")                   \
                              .Device(DEVICE_GPU)                           \
                              .TypeConstraint<T_indices>("T_indices"),      \
                          EmbeddingBagBackwardOp<GPUDevice, T_indices>);

REGISTER_GPU(int32);
REGISTER_GPU(int64);
#endif  // GOOGLE_CUDA
}  // namespace functor
}  // namespace addons
}  // namespace tensorflow
