/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "embeddingbag.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// CPU specialization of actual computation.
template <typename T_indices>
struct EmbeddingBagFunctor<CPUDevice, T_indices> {
  void operator()(const CPUDevice& d, const int value_dim, const int bag_dim, const int indices_size, const T_indices* __restrict__ indices, const float* __restrict__ values, const float* __restrict__ weights, float* __restrict__ out) {
    const int cost = bag_dim * value_dim;
    // auto thread_pool = d->tensorflow_cpu_worker_threads()->workers;

    // for (int i = 0; i < (indices_size / bag_dim) * value_dim; ++i)
    // {
    // out[i] = 0.0f;  // Make sure array is initialized to zero before beginning
    // }

    for (int bag = 0; bag < (indices_size / bag_dim); ++bag) {
      int out_base_idx = bag * value_dim;
      for (int indices_ptr = bag * bag_dim; indices_ptr < (bag + 1) * bag_dim; ++indices_ptr) {
        int values_base_idx = value_dim * indices[indices_ptr];
        float weight = weights[indices_ptr];
        if (indices_ptr == bag * bag_dim) {  // We have to initialize the values, so don't accumulate
          for (int feature_idx = 0; feature_idx < value_dim; ++feature_idx) {
            out[out_base_idx + feature_idx] = values[values_base_idx + feature_idx] * weight;
            }
        }
        else {  // Just accumulate after the first loop iteration
          for (int feature_idx = 0; feature_idx < value_dim; ++feature_idx) {
            out[out_base_idx + feature_idx] += values[values_base_idx + feature_idx] * weight;
            }
        }
      }
    }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T_indices>
class EmbeddingBagOp : public OpKernel {
 public:
  explicit EmbeddingBagOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& indices = context->input(0);
    const Tensor& values = context->input(1);
    const Tensor& weights = context->input(2);

    // I am NOT AN EXPERT at C++ Tensorflow so sorry for ugly code
    TensorShape outShape = indices.shape();
    int valuesDim = values.shape().dim_size(1);
    int bagDim = outShape.dim_size(outShape.dims() - 1);
    outShape.set_dim(outShape.dims() - 1, valuesDim);
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, outShape,
                                                     &output_tensor));

    OP_REQUIRES(context, indices.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));

    OP_REQUIRES(context, values.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("Too many elements in tensor"));
    EmbeddingBagFunctor<Device, T_indices>()(
        context->eigen_device<Device>(),
        valuesDim,
        bagDim,
        static_cast<int>(indices.NumElements()),
        indices.flat<T_indices>().data(),
        values.flat<float>().data(),
        weights.flat<float>().data(),
        output_tensor->flat<float>().data());
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T_indices)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Addons>EmbeddingBag").Device(DEVICE_CPU).TypeConstraint<T_indices>("T_indices"), \
      EmbeddingBagOp<CPUDevice, T_indices>);
REGISTER_CPU(int32);
REGISTER_CPU(int64);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T_indices)                                          \
  extern template struct EmbeddingBagFunctor<GPUDevice, T_indices>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Addons>EmbeddingBag").Device(DEVICE_GPU).TypeConstraint<T_indices>("T_indices"), \
      EmbeddingBagOp<GPUDevice, T_indices>);
REGISTER_GPU(int32);
REGISTER_GPU(int64);
#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow
