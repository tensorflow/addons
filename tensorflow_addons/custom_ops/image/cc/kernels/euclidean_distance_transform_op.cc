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

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace addons {

namespace functor {

template <typename T>
struct EuclideanDistanceTransformFunctor<CPUDevice, T> {
  typedef typename TTypes<uint8, 4>::ConstTensor InputType;
  typedef typename TTypes<T, 4>::Tensor OutputType;
  void operator()(OpKernelContext *ctx, OutputType *output,
                  const InputType &images) const {
    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(
        images.dimension(0) * images.dimension(3),
        images.dimension(1) * images.dimension(2) * 1000,
        [&images, &output](int64 start_index, int64 end_index) {
          for (int index = start_index; index < end_index; index++) {
            int batch_id = index / images.dimension(3);
            int channel = index % images.dimension(3);
            EuclideanDistanceTransformSample<T>(
                images.data(), output->data(), batch_id, channel,
                images.dimension(1), images.dimension(2), images.dimension(3));
          }
        });
  }
};

template struct EuclideanDistanceTransformFunctor<CPUDevice, Eigen::half>;
template struct EuclideanDistanceTransformFunctor<CPUDevice, float>;
template struct EuclideanDistanceTransformFunctor<CPUDevice, double>;

}  // end namespace functor

using functor::EuclideanDistanceTransformFunctor;

template <typename Device, typename T>
class EuclideanDistanceTransform : public OpKernel {
 private:
  EuclideanDistanceTransformFunctor<Device, T> functor_;

 public:
  explicit EuclideanDistanceTransform(OpKernelConstruction *ctx)
      : OpKernel(ctx) {}

  void Compute(OpKernelContext *ctx) override {
    const Tensor &images_t = ctx->input(0);

    OP_REQUIRES(ctx, images_t.shape().dims() == 4,
                errors::InvalidArgument("Input images must have rank 4"));

    OP_REQUIRES(ctx, images_t.NumElements() <= Eigen::NumTraits<int>::highest(),
                errors::InvalidArgument("Input images' size exceeds 2^31-1"));

    Tensor *output_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, images_t.shape(), &output_t));

    auto output = output_t->tensor<T, 4>();
    auto images = images_t.tensor<uint8, 4>();
    functor_(ctx, &output, images);
  }
};

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("Addons>EuclideanDistanceTransform") \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<TYPE>("dtype"),       \
                          EuclideanDistanceTransform<CPUDevice, TYPE>)

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

#define DECLARE_FUNCTOR(TYPE)                                            \
  template <>                                                            \
  void EuclideanDistanceTransformFunctor<GPUDevice, TYPE>::operator()(   \
      OpKernelContext *ctx, OutputType *output, const InputType &images) \
      const;                                                             \
  extern template struct EuclideanDistanceTransformFunctor<GPUDevice, TYPE>

TF_CALL_half(DECLARE_FUNCTOR);
TF_CALL_float(DECLARE_FUNCTOR);
TF_CALL_double(DECLARE_FUNCTOR);

}  // end namespace functor

#define REGISTER(TYPE)                                              \
  REGISTER_KERNEL_BUILDER(Name("Addons>EuclideanDistanceTransform") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<TYPE>("dtype"),       \
                          EuclideanDistanceTransform<GPUDevice, TYPE>)

TF_CALL_half(REGISTER);
TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);

#undef REGISTER

#endif  // GOOGLE_CUDA

}  // end namespace addons
}  // end namespace tensorflow
