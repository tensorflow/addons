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
#include "tensorflow_addons/custom_ops/layers/cc/kernels/embedding_bag_backward.h"

namespace tensorflow {
namespace addons {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// CPU specialization of actual computation.
template <typename T, typename Tindices>
struct EmbeddingBagBackwardFunctor<CPUDevice, T, Tindices> {
  static constexpr int64 kPacketSize = Eigen::internal::packet_traits<T>::size;
  using VectorMap = Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>>;
  using ConstVectorMap = Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>>;

  void operator()(const CPUDevice& device,
                  typename TTypes<Tindices, 2>::ConstTensor indices,
                  typename TTypes<T, 2>::ConstTensor values,
                  typename TTypes<T, 2>::ConstTensor weights,
                  typename TTypes<T, 2>::ConstTensor grads,
                  typename TTypes<T, 2>::Tensor value_grads,
                  typename TTypes<T, 2>::Tensor weight_grads,
                  Combiner combiner) {
    const Eigen::Index sequence_length = indices.dimension(1);
    const Eigen::Index output_dim = values.dimension(1);

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

    const auto compute_value_grads = [&](Eigen::Index start, Eigen::Index end) {
      for (Eigen::Index i = start; i < end; ++i) {
        VectorMap value_grads_slice(&value_grads(index_vec[i].first, 0),
                                    output_dim);
        for (Eigen::Index index : index_vec[i].second) {
          const Eigen::Index bag = index / sequence_length;
          const Eigen::Index seq = index % sequence_length;
          const ConstVectorMap grads_slice(&grads(bag, 0), output_dim);
          value_grads_slice += grads_slice * weights(bag, seq);
        }
        if (combiner == Combiner::kMean) {
          value_grads_slice /= static_cast<T>(sequence_length);
        }
      }
    };

    const Eigen::Index num_unique_values = index_vec.size();
    const double bytes_loaded =
        100 * (output_dim) * (sizeof(Tindices) + sizeof(T));
    const double bytes_stored = output_dim * sizeof(T);
    const double compute_cycles = 100 * (Eigen::TensorOpCost::AddCost<T>() +
                                         Eigen::TensorOpCost::MulCost<T>());
    const Eigen::TensorOpCost cost(bytes_loaded, bytes_stored, compute_cycles,
                                   /*vectorized=*/true,
                                   /*packet_size=*/kPacketSize);
    value_grads.setZero();
    device.parallelFor(num_unique_values, cost, std::move(compute_value_grads));

    const auto compute_weight_grads =
        [&](const Eigen::array<Eigen::Index, 2>& coords) -> T {
      const Eigen::Index bag = coords[0];
      const Eigen::Index seq = coords[1];
      const ConstVectorMap grads_slice(&grads(bag, 0), output_dim);
      const ConstVectorMap values_slice(&values(indices(bag, seq), 0),
                                        output_dim);
      T output = values_slice.dot(grads_slice);
      if (combiner == Combiner::kMean) {
        output /= static_cast<T>(sequence_length);
      }
      return output;
    };

    weight_grads.device(device) =
        weight_grads.generate(std::move(compute_weight_grads));
  }
};
}  // namespace functor

template <typename Device, typename T, typename Tindices>
class EmbeddingBagBackwardOp : public OpKernel {
 public:
  explicit EmbeddingBagBackwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    std::string combiner_string;
    OP_REQUIRES_OK(context, context->GetAttr("combiner", &combiner_string));
    OP_REQUIRES_OK(context, ValidateCombiner(combiner_string, &combiner_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& indices = context->input(0);
    const Tensor& values = context->input(1);
    const Tensor& weights = context->input(2);
    const Tensor& grads = context->input(3);

    Tensor* value_grads = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, values.shape(), &value_grads));
    Tensor* weight_grads = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, weights.shape(), &weight_grads));

    functor::EmbeddingBagBackwardFunctor<Device, T, Tindices>()(
        context->eigen_device<Device>(), indices.tensor<Tindices, 2>(),
        values.tensor<T, 2>(), weights.tensor<T, 2>(), grads.tensor<T, 2>(),
        value_grads->tensor<T, 2>(), weight_grads->tensor<T, 2>(), combiner_);
  }

 private:
  Combiner combiner_;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                 \
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
REGISTER_CPU(Eigen::half);
REGISTER_CPU(float);
REGISTER_CPU(double);

// Register the GPU kernels.
#if GOOGLE_CUDA
// #define REGISTER_GPU(Tindices)                                             \
//   extern template struct EmbeddingBagBackwardFunctor<GPUDevice, Tindices>; \
//   REGISTER_KERNEL_BUILDER(Name("Addons>EmbeddingBagGrad")                  \
//                               .Device(DEVICE_GPU)                          \
//                               .TypeConstraint<Tindices>("Tindices"),       \
//                           EmbeddingBagBackwardOp<GPUDevice, Tindices>);

// REGISTER_GPU(int32);
// REGISTER_GPU(int64);
#endif  // GOOGLE_CUDA
}  // namespace addons
}  // namespace tensorflow
