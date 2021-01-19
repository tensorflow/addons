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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow_addons/custom_ops/layers/cc/kernels/embedding_bag.h"

namespace tensorflow {
namespace addons {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// CPU specialization of actual computation.
template <typename T, typename Tindices>
struct EmbeddingBagFunctor<CPUDevice, T, Tindices> {
  static constexpr int64 kPacketSize = Eigen::internal::packet_traits<T>::size;
  using VectorMap = Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>;
  using ConstVectorMap = Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>;

  void operator()(const CPUDevice& d, const Eigen::Index value_dim,
                  const Eigen::Index sequence_length,
                  const Eigen::Index num_bags,
                  const Tindices* __restrict__ indices,
                  const T* __restrict__ values, const T* __restrict__ weights,
                  T* __restrict__ output) {
    const auto work = [&](Eigen::Index start, Eigen::Index end) {
      for (Eigen::Index bag = start; bag < end; ++bag) {
        VectorMap output_slice(output + bag * value_dim, value_dim);
        output_slice.setZero();
        for (Eigen::Index indices_offset = bag * sequence_length;
             indices_offset < (bag + 1) * sequence_length; ++indices_offset) {
          const T* values_ptr = values + value_dim * indices[indices_offset];
          T weight = weights[indices_offset];
          const ConstVectorMap values_slice(values_ptr, value_dim);
          output_slice += values_slice * weight;
        }
      }
    };

    const double bytes_loaded =
        sequence_length * (sizeof(Tindices) + 2 * sizeof(T));
    const double bytes_stored = sequence_length * sizeof(T);
    const double compute_cycles =
        sequence_length *
        (Eigen::TensorOpCost::AddCost<T>() + Eigen::TensorOpCost::MulCost<T>());
    const Eigen::TensorOpCost cost(bytes_loaded, bytes_stored, compute_cycles,
                                   /*vectorized=*/true,
                                   /*packet_size=*/kPacketSize);
    d.parallelFor(num_bags, cost, std::move(work));
  }
};

template <typename Device, typename T, typename Tindices>
class EmbeddingBagOp : public OpKernel {
 public:
  explicit EmbeddingBagOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("combiner", &combiner_));
    OP_REQUIRES(context, combiner_ == "SUM",
                errors::InvalidArgument("Only support 'SUM' combiner."));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& indices = context->input(0);
    const Tensor& values = context->input(1);
    const Tensor& weights = context->input(2);

    const TensorShape& indices_shape = indices.shape();
    const TensorShape& values_shape = weights.shape();
    const TensorShape& weights_shape = weights.shape();

    OP_REQUIRES(
        context, TensorShapeUtils::IsMatrixOrHigher(indices_shape),
        errors::InvalidArgument("indices shape should be at least 2-D."));
    OP_REQUIRES(context, indices_shape == weights_shape,
                errors::InvalidArgument(
                    "Shape of indices and weights should be equal."));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(values_shape),
                errors::InvalidArgument("values shape should be 2-D."));

    TensorShape output_shape = indices_shape;
    Eigen::Index value_dim = values.shape().dim_size(1);
    Eigen::Index sequence_length =
        output_shape.dim_size(output_shape.dims() - 1);
    Eigen::Index num_bags = indices.NumElements() / sequence_length;
    output_shape.set_dim(output_shape.dims() - 1, value_dim);

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));

    EmbeddingBagFunctor<Device, T, Tindices>()(
        context->eigen_device<Device>(), value_dim, sequence_length, num_bags,
        indices.flat<Tindices>().data(), values.flat<T>().data(),
        weights.flat<T>().data(), output_tensor->flat<T>().data());
  }

 private:
  std::string combiner_;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBag")             \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<int32>("Tindices"), \
                          EmbeddingBagOp<CPUDevice, T, int32>);   \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBag")             \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<int64>("Tindices"), \
                          EmbeddingBagOp<CPUDevice, T, int64>);
REGISTER_CPU(Eigen::half);
REGISTER_CPU(float);
REGISTER_CPU(double);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(Tindices)                                       \
  extern template struct EmbeddingBagFunctor<GPUDevice, Tindices>;   \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBag")                \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<Tindices>("Tindices"), \
                          EmbeddingBagOp<GPUDevice, Tindices>);
REGISTER_GPU(int32);
REGISTER_GPU(int64);
#endif  // GOOGLE_CUDA
}  // namespace functor
}  // namespace addons
}  // namespace tensorflow
