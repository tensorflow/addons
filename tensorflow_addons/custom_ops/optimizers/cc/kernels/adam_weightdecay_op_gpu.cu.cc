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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <algorithm>

#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"
#include "tensorflow_addons/custom_ops/optimizers/cc/kernels/adam_weightdecay_op.h"

using namespace tensorflow;

namespace tensorflow {
namespace addons {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
__global__ void ApplyAdamWithWeightdecayKernel(
    int32 data_dim, const T* var, T* var_update, const T* m, T* out_m,
    const T* v, T* out_v, const T* beta1_power_, const T* const beta2_power_,
    const T* const wd_, const T* const beta1_, const T* const beta2_,
    const T* const epsilon_, const T* grad, bool use_nesterov) {
  eigen_assert(blockDim.y == 1);
  eigen_assert(blockDim.z == 1);
  eigen_assert(gridDim.y == 1);
  eigen_assert(gridDim.z == 1);

  const T mul_factor = sqrt(static_cast<T>(1.0) - (*beta2_power_)) /
                       (static_cast<T>(1.0) - (*beta1_power_));
  const T epsilon = (*epsilon_);
  const T beta1 = (*beta1_);
  const T one_minus_beta1 = static_cast<T>(1.0) - (beta1);
  const T one_minus_beta2 = static_cast<T>(1.0) - (*beta2_);
  const int32 stripe = gridDim.x * blockDim.x;

  for (int32 i = blockIdx.x * blockDim.x + threadIdx.x; i < data_dim;
       i += stripe) {
    auto m_i = m[i];
    auto g_i = grad[i];
    auto v_i = v[i];
    auto var_i = var[i];

    m_i += one_minus_beta1 * (g_i - m_i);
    v_i += one_minus_beta2 * (g_i * g_i - v_i);

    if (use_nesterov) {
      var_update[i] = mul_factor * (m_i * beta1 + one_minus_beta1 * g_i) /
                          (epsilon + sqrt(v_i)) +
                      (*wd_) * var_i;
    } else {
      var_update[i] = mul_factor * m_i / (epsilon + sqrt(v_i)) + (*wd_) * var_i;
    }

    out_m[i] = m_i;
    out_v[i] = v_i;
  }
}

template <typename T>
struct AdamWithWeightdecay<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::ConstFlat var,
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
    int32 data_dim = grad.dimension(0);
    GpuLaunchConfig config = GetGpuLaunchConfig(data_dim, d);
    eigen_assert(static_cast<int64>(grad.dimension(0)) +
                     static_cast<int64>(config.block_count) *
                         static_cast<int64>(config.thread_per_block) <
                 std::numeric_limits<int32>::max());

    TF_CHECK_OK(GpuLaunchKernel(
        ApplyAdamWithWeightdecayKernel<T>, config.block_count,
        config.thread_per_block, 0, d.stream(), data_dim, var.data(),
        var_update.data(), m.data(), out_m.data(), v.data(), out_v.data(),
        beta1_power.data(), beta2_power.data(), wd.data(), beta1.data(),
        beta2.data(), epsilon.data(), grad.data(), use_nesterov));
  }
};

#define DEFINE_GPU_SPECS(T) \
  template struct functor::AdamWithWeightdecay<GPUDevice, T>;
DEFINE_GPU_SPECS(Eigen::half);
DEFINE_GPU_SPECS(float);
DEFINE_GPU_SPECS(double);
#undef DEFINE_GPU_SPECS

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif
