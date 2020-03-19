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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow_addons/custom_ops/optimizers/cc/kernels/adam_weightdecay_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/refcount.h"

using namespace tensorflow;

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;
using Index = Eigen::Index;

namespace tensorflow {
namespace addons {

typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class AdamWithWeightdecayOp : public OpKernel {
 public:
  explicit AdamWithWeightdecayOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_locking", &use_exclusive_lock_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_nesterov", &use_nesterov_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor var = ctx->input(0);
    const Tensor m = ctx->input(1);
    const Tensor v = ctx->input(2);

    const Tensor& beta1_power = ctx->input(3);
    const Tensor& beta2_power = ctx->input(4);
    const Tensor& wd = ctx->input(5);
    const Tensor& beta1 = ctx->input(6);
    const Tensor& beta2 = ctx->input(7);
    const Tensor& epsilon = ctx->input(8);

    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1_power.shape()),
                errors::InvalidArgument("beta1_power is not a scalar: ",
                                        beta1_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2_power.shape()),
                errors::InvalidArgument("beta2_power is not a scalar: ",
                                        beta2_power.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(wd.shape()),
                errors::InvalidArgument("wd is not a scalar : ",
                                        wd.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta1.shape()),
                errors::InvalidArgument("beta1 is not a scalar: ",
                                        beta1.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(beta2.shape()),
                errors::InvalidArgument("beta2 is not a scalar: ",
                                        beta2.shape().DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(epsilon.shape()),
                errors::InvalidArgument("epsilon is not a scalar: ",
                                        epsilon.shape().DebugString()));

    const Tensor& grad = ctx->input(9);
    OP_REQUIRES(ctx, var.shape().IsSameSize(m.shape()),
                errors::InvalidArgument("var and m do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        m.shape().DebugString()));
    OP_REQUIRES(ctx, var.shape().IsSameSize(v.shape()),
                errors::InvalidArgument("var and v do not have the same shape",
                                        var.shape().DebugString(), " ",
                                        v.shape().DebugString()));
    OP_REQUIRES(
        ctx, var.shape().IsSameSize(grad.shape()),
        errors::InvalidArgument("var and grad do not have the same shape",
                                var.shape().DebugString(), " ",
                                grad.shape().DebugString()));

    const Device& device = ctx->template eigen_device<Device>();

    Tensor* var_update = NULL;
    Tensor* m_update = NULL;
    Tensor* v_update = NULL;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, var.shape(), &var_update));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, m.shape(), &m_update));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, v.shape(), &v_update));

    functor::AdamWithWeightdecay<Device, T>()(
        device, var.flat<T>(), var_update->flat<T>(), m.flat<T>(),
        m_update->flat<T>(), v.flat<T>(), v_update->flat<T>(),
        beta1_power.scalar<T>(), beta2_power.scalar<T>(), wd.scalar<T>(),
        beta1.scalar<T>(), beta2.scalar<T>(), epsilon.scalar<T>(),
        grad.flat<T>(), use_nesterov_);
  }

 private:
  bool use_exclusive_lock_;
  bool use_nesterov_;
};

namespace functor {
template <typename Device, typename T>
struct AdamWithWeightdecayNoCuda {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat var,
                  typename TTypes<T>::Flat var_update,
                  typename TTypes<T>::ConstFlat m,
                  typename TTypes<T>::Flat out_m,
                  typename TTypes<T>::ConstFlat v,
                  typename TTypes<T>::Flat out_v,
                  typename TTypes<T>::ConstScalar beta1_power,
                  typename TTypes<T>::ConstScalar beta2_power,
                  typename TTypes<T>::ConstScalar wd,
                  typename TTypes<T>::ConstScalar beta1,
                  typename TTypes<T>::ConstScalar beta2,
                  typename TTypes<T>::ConstScalar epsilon,
                  typename TTypes<T>::ConstFlat grad, bool use_nesterov) {
    // Get params length and check if they can be vectorized by packet size.
    Index length = var.size();
    Index packet_size = Eigen::internal::packet_traits<T>::size;
    if (length % packet_size == 0) {
      length = length / packet_size;
    } else {
      packet_size = 1;
    }

    const T* var_ptr = var.data();
    const T* m_ptr = m.data();
    const T* v_ptr = v.data();

    T* var_update_ptr = var_update.data();
    T* out_m_ptr = out_m.data();
    T* out_v_ptr = out_v.data();

    const T* g_ptr = grad.data();
    const T mul_factor =
        Eigen::numext::sqrt(T(1) - beta2_power()) / (T(1) - beta1_power());
    // beta1 == μ
    // beta2 == ν
    // v     == n
    // var   == θ

    auto shard = [this, var_ptr, var_update_ptr, m_ptr, out_m_ptr, v_ptr,
                  out_v_ptr, g_ptr, mul_factor, wd, beta1, beta2, epsilon,
                  use_nesterov, packet_size](int begin, int end) {
      int t_size = (end - begin) * packet_size;
      begin = begin * packet_size;

      auto var =
          typename TTypes<T>::UnalignedConstTensor(var_ptr + begin, t_size);
      auto m = typename TTypes<T>::UnalignedConstTensor(m_ptr + begin, t_size);
      auto v = typename TTypes<T>::UnalignedConstTensor(v_ptr + begin, t_size);
      auto g = typename TTypes<T>::UnalignedConstTensor(g_ptr + begin, t_size);

      auto var_update =
          typename TTypes<T>::UnalignedTensor(var_update_ptr + begin, t_size);
      auto out_m =
          typename TTypes<T>::UnalignedTensor(out_m_ptr + begin, t_size);
      auto out_v =
          typename TTypes<T>::UnalignedTensor(out_v_ptr + begin, t_size);

      out_m = m + (g - m) * (T(1) - beta1());
      out_v = v + (g.square() - v) * (T(1) - beta2());
      if (use_nesterov) {
        var_update = ((g * (T(1) - beta1()) + beta1() * out_m) * mul_factor) /
                         (out_v.sqrt() + epsilon()) +
                     wd() * var;
      } else {
        var_update =
            (out_m * mul_factor) / (out_v.sqrt() + epsilon()) + wd() * var;
      }
    };

    // Input data: var, v, m, grad.
    // Output data: var_update, out_v, out_m.
    const int input_bytes = length * packet_size * sizeof(T) * 4;
    const int output_bytes = length * packet_size * sizeof(T) * 3;
    const int compute_cycles =
        // Consider Sub as Add
        (Eigen::TensorOpCost::AddCost<int>() * 7 +
         Eigen::TensorOpCost::MulCost<int>() * 2 +
         Eigen::TensorOpCost::AddCost<T>() * 12 +
         Eigen::TensorOpCost::MulCost<T>() * 10 +
         Eigen::TensorOpCost::DivCost<T>()) *
        length;

    const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);

    // Eigen device must update 3 variables with 3 different expressions,
    // which is bad for cache locality on CPU. Here use ParallelFor instead of
    // "regular" tensor expressions to get better performance.
    d.parallelFor(length, cost, shard);
  }
};

template <typename T>
struct AdamWithWeightdecay<CPUDevice, T>
    : AdamWithWeightdecayNoCuda<CPUDevice, T> {};

#define REGISTER_KERNELS(D, T)                                    \
  REGISTER_KERNEL_BUILDER(Name("Addons>ApplyAdamWithWeightdecay") \
                              .Device(DEVICE_##D)                 \
                              .TypeConstraint<T>("T"),            \
                          AdamWithWeightdecayOp<D##Device, T>);

REGISTER_KERNELS(CPU, Eigen::half);
REGISTER_KERNELS(CPU, float);
REGISTER_KERNELS(CPU, double);

#if GOOGLE_CUDA

#define DECLARE_GPU_SPEC(T)                                                 \
  template <>                                                               \
  void AdamWithWeightdecay<GPUDevice, T>::operator()(                       \
      const GPUDevice& d, typename TTypes<T>::ConstFlat var,                \
      typename TTypes<T>::Flat var_update, typename TTypes<T>::ConstFlat m, \
      typename TTypes<T>::Flat out_m, typename TTypes<T>::ConstFlat v,      \
      typename TTypes<T>::Flat out_v,                                       \
      typename TTypes<T>::ConstScalar beta1_power,                          \
      typename TTypes<T>::ConstScalar beta2_power,                          \
      typename TTypes<T>::ConstScalar wd,                                   \
      typename TTypes<T>::ConstScalar beta1,                                \
      typename TTypes<T>::ConstScalar beta2,                                \
      typename TTypes<T>::ConstScalar epsilon,                              \
      typename TTypes<T>::ConstFlat grad, bool use_nesterov);               \
  extern template struct AdamWithWeightdecay<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
DECLARE_GPU_SPEC(Eigen::half)
#undef DECLARE_GPU_SPEC

REGISTER_KERNELS(GPU, Eigen::half);
REGISTER_KERNELS(GPU, float);
REGISTER_KERNELS(GPU, double);
#undef REGISTER_GPU_KERNEL
#endif  // GOOGLE_CUDA
}  // namespace functor
}  // namespace addons
}  // namespace tensorflow
