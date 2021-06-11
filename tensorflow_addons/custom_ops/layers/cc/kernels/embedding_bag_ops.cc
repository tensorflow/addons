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

#include "tensorflow_addons/custom_ops/layers/cc/kernels/embedding_bag_ops.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

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

  void operator()(const CPUDevice &device,
                  typename TTypes<Tindices, 2>::ConstTensor indices,
                  typename TTypes<T, 2>::ConstTensor params,
                  typename TTypes<T, 2>::ConstTensor weights,
                  typename TTypes<T, 2>::Tensor output, Combiner combiner) {
    const Eigen::Index bags = indices.dimension(0);
    const Eigen::Index sequence_length = indices.dimension(1);
    const Eigen::Index output_dim = params.dimension(1);

    const auto work = [&](Eigen::Index start, Eigen::Index end) {
      for (Eigen::Index bag = start; bag < end; ++bag) {
        VectorMap output_slice(&output(bag, 0), output_dim);
        output_slice.setZero();
        for (Eigen::Index seq = 0; seq < sequence_length; ++seq) {
          const ConstVectorMap params_slice(&params(indices(bag, seq), 0),
                                            output_dim);
          output_slice += params_slice * weights(bag, seq);
        }
        if (combiner == Combiner::kMean) {
          output_slice /= static_cast<T>(sequence_length);
        }
      }
    };

    const double bytes_loaded =
        sequence_length * (sizeof(Tindices) + sizeof(T)) +
        (sequence_length * output_dim) * sizeof(T);
    const double bytes_stored = output_dim * sizeof(T);
    const double compute_cycles =
        (sequence_length * output_dim) *
        (Eigen::TensorOpCost::AddCost<T>() + Eigen::TensorOpCost::MulCost<T>());
    const Eigen::TensorOpCost cost(bytes_loaded, bytes_stored, compute_cycles,
                                   /*vectorized=*/true,
                                   /*packet_size=*/kPacketSize);
    device.parallelFor(bags, cost, std::move(work));
  }
};

// CPU specialization of actual computation.
template <typename T, typename Tindices>
struct EmbeddingBagBackwardFunctor<CPUDevice, T, Tindices> {
  static constexpr int64 kPacketSize = Eigen::internal::packet_traits<T>::size;
  using VectorMap = Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>;
  using ConstVectorMap = Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>;

  void operator()(const CPUDevice &device,
                  typename TTypes<Tindices, 2>::ConstTensor indices,
                  typename TTypes<T, 2>::ConstTensor params,
                  typename TTypes<T, 2>::ConstTensor weights,
                  typename TTypes<T, 2>::ConstTensor grads,
                  typename TTypes<T, 2>::Tensor params_grads,
                  typename TTypes<T, 2>::Tensor weights_grads,
                  Combiner combiner, OpKernelContext *context) {
    const Eigen::Index sequence_length = indices.dimension(1);
    const Eigen::Index output_dim = params.dimension(1);

    std::unordered_map<Tindices, Eigen::Index> index_map;
    // The pair (x, {y_i}) in index_vec means
    // index y_i in `indices` contributes to bag `x`.
    std::vector<std::pair<Tindices, std::vector<Eigen::Index>>> index_vec;
    for (Eigen::Index i = 0; i < indices.size(); ++i) {
      Tindices index = indices.data()[i];
      if (index_map.find(index) == index_map.end()) {
        index_map[index] = index_vec.size();
        index_vec.push_back({index, {}});
      }
      index_vec[index_map[index]].second.push_back(i);
    }

    const auto compute_params_grads = [&](Eigen::Index start,
                                          Eigen::Index end) {
      for (Eigen::Index i = start; i < end; ++i) {
        VectorMap params_grads_slice(&params_grads(index_vec[i].first, 0),
                                     output_dim);
        for (Eigen::Index index : index_vec[i].second) {
          const Eigen::Index bag = index / sequence_length;
          const Eigen::Index seq = index % sequence_length;
          const ConstVectorMap grads_slice(&grads(bag, 0), output_dim);
          params_grads_slice += grads_slice * weights(bag, seq);
        }
        if (combiner == Combiner::kMean) {
          params_grads_slice /= static_cast<T>(sequence_length);
        }
      }
    };

    const Eigen::Index num_unique_params = index_vec.size();
    const double bytes_loaded = 100 * output_dim * sizeof(T);
    const double bytes_stored = output_dim * sizeof(T);
    const double compute_cycles =
        100 * output_dim *
        (Eigen::TensorOpCost::AddCost<T>() + Eigen::TensorOpCost::MulCost<T>());
    const Eigen::TensorOpCost cost(bytes_loaded, bytes_stored, compute_cycles,
                                   /*vectorized=*/true,
                                   /*packet_size=*/kPacketSize);
    params_grads.setZero();
    device.parallelFor(num_unique_params, cost,
                       std::move(compute_params_grads));

    const auto compute_weights_grads =
        [&](const Eigen::array<Eigen::Index, 2> &coords) -> T {
      const Eigen::Index bag = coords[0];
      const Eigen::Index seq = coords[1];
      const ConstVectorMap grads_slice(&grads(bag, 0), output_dim);
      const ConstVectorMap params_slice(&params(indices(bag, seq), 0),
                                        output_dim);
      T output = params_slice.dot(grads_slice);
      if (combiner == Combiner::kMean) {
        output /= static_cast<T>(sequence_length);
      }
      return output;
    };

    weights_grads.device(device) =
        weights_grads.generate(std::move(compute_weights_grads));
  }
};
}  // namespace functor

namespace {
bool ValidateCombiner(const std::string &combiner_string, Combiner *combiner) {
  if (combiner_string == "SUM") {
    *combiner = Combiner::kSum;
  } else if (combiner_string == "MEAN") {
    *combiner = Combiner::kMean;
  } else {
    return false;
  }
  return true;
}
}  // namespace

template <typename Device, typename T, typename Tindices>
class EmbeddingBagOp : public OpKernel {
 public:
  explicit EmbeddingBagOp(OpKernelConstruction *context) : OpKernel(context) {
    std::string combiner_string;
    OP_REQUIRES_OK(context, context->GetAttr("combiner", &combiner_string));
    OP_REQUIRES(
        context, ValidateCombiner(combiner_string, &combiner_),
        errors::InvalidArgument("Only support 'SUM' and 'MEAN' combiner."));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &indices = context->input(0);
    const Tensor &params = context->input(1);
    const Tensor &weights = context->input(2);

    const TensorShape &indices_shape = indices.shape();
    const TensorShape &params_shape = params.shape();
    const TensorShape &weights_shape = weights.shape();

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(indices_shape),
                errors::InvalidArgument("indices shape should be 2-D."));
    OP_REQUIRES(context, indices_shape == weights_shape,
                errors::InvalidArgument(
                    "Shape of indices and weights should be equal."));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(params_shape),
                errors::InvalidArgument("params shape should be 2-D."));

    TensorShape output_shape = {indices_shape.dim_size(0),
                                params_shape.dim_size(1)};

    Tensor *output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    functor::EmbeddingBagFunctor<Device, T, Tindices>()(
        context->eigen_device<Device>(), indices.tensor<Tindices, 2>(),
        params.tensor<T, 2>(), weights.tensor<T, 2>(), output->tensor<T, 2>(),
        combiner_);
  }

 private:
  Combiner combiner_;
};

template <typename Device, typename T, typename Tindices>
class EmbeddingBagBackwardOp : public OpKernel {
 public:
  explicit EmbeddingBagBackwardOp(OpKernelConstruction *context)
      : OpKernel(context) {
    std::string combiner_string;
    OP_REQUIRES_OK(context, context->GetAttr("combiner", &combiner_string));
    OP_REQUIRES(
        context, ValidateCombiner(combiner_string, &combiner_),
        errors::InvalidArgument("Only support 'SUM' and 'MEAN' combiner."));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &indices = context->input(0);
    const Tensor &params = context->input(1);
    const Tensor &weights = context->input(2);
    const Tensor &grads = context->input(3);

    Tensor *params_grads = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, params.shape(), &params_grads));
    Tensor *weights_grads = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(1, weights.shape(), &weights_grads));
    functor::EmbeddingBagBackwardFunctor<Device, T, Tindices>()(
        context->eigen_device<Device>(), indices.tensor<Tindices, 2>(),
        params.tensor<T, 2>(), weights.tensor<T, 2>(), grads.tensor<T, 2>(),
        params_grads->tensor<T, 2>(), weights_grads->tensor<T, 2>(), combiner_,
        context);  // Pass the context so the GPU op can allocate the temporary
                   // arrays it needs
  }

 private:
  Combiner combiner_;
};

// Register the CPU kernels.
#define REGISTER_CPU_KERNEL(T)                                          \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBag")                   \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<T>("T")                   \
                              .TypeConstraint<int32>("Tindices"),       \
                          EmbeddingBagOp<CPUDevice, T, int32>);         \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBag")                   \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<T>("T")                   \
                              .TypeConstraint<int64>("Tindices"),       \
                          EmbeddingBagOp<CPUDevice, T, int64>);         \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBagGrad")               \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<T>("T")                   \
                              .TypeConstraint<int32>("Tindices"),       \
                          EmbeddingBagBackwardOp<CPUDevice, T, int32>); \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBagGrad")               \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<T>("T")                   \
                              .TypeConstraint<int64>("Tindices"),       \
                          EmbeddingBagBackwardOp<CPUDevice, T, int64>);
REGISTER_CPU_KERNEL(Eigen::half);
REGISTER_CPU_KERNEL(float);
REGISTER_CPU_KERNEL(double);
#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA
namespace functor {
// Forward declarations of the functor specializations for GPU.
#define DECLARE_GPU_SPEC(T, Tindices)                                         \
  template <>                                                                 \
  void EmbeddingBagFunctor<GPUDevice, T, Tindices>::operator()(               \
      const GPUDevice &, typename TTypes<Tindices, 2>::ConstTensor,           \
      typename TTypes<T, 2>::ConstTensor, typename TTypes<T, 2>::ConstTensor, \
      typename TTypes<T, 2>::Tensor, Combiner);                               \
  extern template struct EmbeddingBagFunctor<GPUDevice, T, Tindices>;

#define DECLARE_GPU_SPECS(T)  \
  DECLARE_GPU_SPEC(T, int32); \
  DECLARE_GPU_SPEC(T, int64);

DECLARE_GPU_SPECS(Eigen::half);
DECLARE_GPU_SPECS(float);
DECLARE_GPU_SPECS(double);
#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPECS
}  // namespace functor

// Register the GPU kernels.
#define REGISTER_GPU_KERNEL(T)                                          \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBag")                   \
                              .Device(DEVICE_GPU)                       \
                              .TypeConstraint<T>("T")                   \
                              .TypeConstraint<int32>("Tindices"),       \
                          EmbeddingBagOp<GPUDevice, T, int32>);         \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBag")                   \
                              .Device(DEVICE_GPU)                       \
                              .TypeConstraint<T>("T")                   \
                              .TypeConstraint<int64>("Tindices"),       \
                          EmbeddingBagOp<GPUDevice, T, int64>);         \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBagGrad")               \
                              .Device(DEVICE_GPU)                       \
                              .TypeConstraint<T>("T")                   \
                              .TypeConstraint<int32>("Tindices"),       \
                          EmbeddingBagBackwardOp<GPUDevice, T, int32>); \
  REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBagGrad")               \
                              .Device(DEVICE_GPU)                       \
                              .TypeConstraint<T>("T")                   \
                              .TypeConstraint<int64>("Tindices"),       \
                          EmbeddingBagBackwardOp<GPUDevice, T, int64>);
REGISTER_GPU_KERNEL(Eigen::half);
REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA
}  // namespace addons
}  // namespace tensorflow
