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

#include "tensorflow_addons/custom_ops/image/cc/kernels/euclidean_distance_transform_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace functor {

typedef Eigen::ThreadPoolDevice CPUDevice;

template struct EuclideanDistanceTransformFunctor<CPUDevice, Eigen::half>;
template struct EuclideanDistanceTransformFunctor<CPUDevice, float>;
template struct EuclideanDistanceTransformFunctor<CPUDevice, double>;

}  // end namespace functor

typedef Eigen::ThreadPoolDevice CPUDevice;

using functor::EuclideanDistanceTransformFunctor;
using generator::EuclideanDistanceTransformGenerator;

template <typename Device, typename T>
class EuclideanDistanceTransform : public OpKernel {
 public:
  explicit EuclideanDistanceTransform(OpKernelConstruction *ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    const Tensor &images_t = ctx->input(0);

    OP_REQUIRES(ctx, images_t.shape().dims() == 4,
                errors::InvalidArgument("Input images must have rank 4"));

    Tensor *output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, images_t.shape(), &output_t));

    auto output = output_t->tensor<T, 4>();
    auto images = images_t.tensor<T, 4>();

    EuclideanDistanceTransformFunctor<Device, T> functor;
    functor(ctx->eigen_device<Device>(), &output, images);
  }
};

#define REGISTER(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(Name("EuclideanDistanceTransform")  \
                              .Device(DEVICE_CPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          EuclideanDistanceTransform<CPUDevice, TYPE>)

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

#define DECLARE_FUNCTOR(TYPE)                                              \
  template <>                                                              \
  void EuclideanDistanceTransformFunctor<GPUDevice, TYPE>::operator()(     \
      const GPUDevice &device, OutpuType *output, const InputType *images) \
      const;                                                               \
  extern template struct EuclideanDistanceTransformFunctor<GPUDevice, TYPE>

TF_CALL_half(DECLARE_FUNCTOR);
TF_CALL_float(DECLARE_FUNCTOR);
TF_CALL_double(DECLARE_FUNCTOR);

}  // end namespace functor

#define REGISTER(TYPE)                                        \
  REGISTER_KERNEL_BUILDER(Name("EuclideanDistanceTransform")  \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<TYPE>("dtype"), \
                          .HostMemory("output_shape"),        \
                          EuclideanDistanceTransform<GPUDevice, TYPE>)

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
