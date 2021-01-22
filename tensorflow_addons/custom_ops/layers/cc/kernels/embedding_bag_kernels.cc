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

  void operator()(const CPUDevice& device,
                  typename TTypes<Tindices, 2>::ConstTensor indices,
                  typename TTypes<T, 2>::ConstTensor values,
                  typename TTypes<T, 2>::ConstTensor weights,
                  typename TTypes<T, 2>::Tensor output, Combiner combiner) {
    const Eigen::Index bags = indices.dimension(0);
    const Eigen::Index sequence_length = indices.dimension(1);
    const Eigen::Index output_dim = values.dimension(1);

    const auto work = [&](Eigen::Index start, Eigen::Index end) {
      for (Eigen::Index bag = start; bag < end; ++bag) {
        VectorMap output_slice(&output(bag, 0), output_dim);
        output_slice.setZero();
        for (Eigen::Index seq = 0; seq < sequence_length; ++seq) {
          const ConstVectorMap values_slice(&values(indices(bag, seq), 0),
                                            output_dim);
          output_slice += values_slice * weights(bag, seq);
        }
        if (combiner == Combiner::kMean) {
          output_slice /= static_cast<T>(sequence_length);
        }
      }
    };

    const double bytes_loaded =
        (sequence_length * output_dim) * (sizeof(Tindices) + 2 * sizeof(T));
    const double bytes_stored = (sequence_length * output_dim) * sizeof(T);
    const double compute_cycles =
        (sequence_length * output_dim) *
        (Eigen::TensorOpCost::AddCost<T>() + Eigen::TensorOpCost::MulCost<T>());
    const Eigen::TensorOpCost cost(bytes_loaded, bytes_stored, compute_cycles,
                                   /*vectorized=*/true,
                                   /*packet_size=*/kPacketSize);
    device.parallelFor(bags, cost, std::move(work));
  }
};

template <typename Device, typename T, typename Tindices>
class EmbeddingBagOp : public OpKernel {
 public:
  explicit EmbeddingBagOp(OpKernelConstruction* context) : OpKernel(context) {
    std::string combiner_string;
    OP_REQUIRES_OK(context, context->GetAttr("combiner", &combiner_string));
    OP_REQUIRES_OK(context, ValidateCombiner(combiner_string, &combiner_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& indices = context->input(0);
    const Tensor& values = context->input(1);
    const Tensor& weights = context->input(2);

    const TensorShape& indices_shape = indices.shape();
    const TensorShape& values_shape = values.shape();
    const TensorShape& weights_shape = weights.shape();

    OP_REQUIRES(
        context, TensorShapeUtils::IsMatrix(indices_shape),
        errors::InvalidArgument("indices shape should be at least 2-D."));
    OP_REQUIRES(context, indices_shape == weights_shape,
                errors::InvalidArgument(
                    "Shape of indices and weights should be equal."));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(values_shape),
                errors::InvalidArgument("values shape should be 2-D."));

    TensorShape output_shape = indices_shape;
    Eigen::Index output_dim = values.shape().dim_size(1);
    output_shape.set_dim(output_shape.dims() - 1, output_dim);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    EmbeddingBagFunctor<Device, T, Tindices>()(
        context->eigen_device<Device>(), indices.tensor<Tindices, 2>(),
        values.tensor<T, 2>(), weights.tensor<T, 2>(), output->tensor<T, 2>(),
        combiner_);
  }

 private:
  Combiner combiner_;
};

// Register the CPU kernels.
#define REGISTER_CPU_KERNEL(T)                                    \
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
REGISTER_CPU_KERNEL(Eigen::half);
REGISTER_CPU_KERNEL(float);
REGISTER_CPU_KERNEL(double);
#undef REGISTER_CPU_KERNEL

// Register the GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_GPU_KERNEL(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBag")             \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<int32>("Tindices"), \
                          EmbeddingBagOp<GPUDevice, T, int32>);   \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBag")             \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T")             \
                              .TypeConstraint<int64>("Tindices"), \
                          EmbeddingBagOp<GPUDevice, T, int64>);
REGISTER_GPU_KERNEL(Eigen::half);
REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA
}  // namespace functor
}  // namespace addons
}  // namespace tensorflow
