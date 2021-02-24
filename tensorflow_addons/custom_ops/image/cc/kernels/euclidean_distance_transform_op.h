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

#ifndef TENSORFLOW_ADDONS_IMAGE_KERNELS_EUCLIDEAN_DISTANCE_TRANSFORM_OP_H_
#define TENSORFLOW_ADDONS_IMAGE_KERNELS_EUCLIDEAN_DISTANCE_TRANSFORM_OP_H_

#define EIGEN_USE_THREADS

#include <limits>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace addons {

namespace generator {

template <typename Device, typename T>
class EuclideanDistanceTransformGenerator {
 private:
  typename TTypes<T, 4>::ConstTensor input_;
  int64 height_, width_;

  void distance(std::vector<T>& f, std::vector<T>& d) {
    int n = d.size();
    std::vector<int> v(n);    // locations of parabolas in lower envelope
    std::vector<T> z(n + 1);  // locations of boundaries between parabolas
    int k = 0;                // index of rightmost parabola in lower envelope
    v[0] = 0;
    z[0] = -std::numeric_limits<T>::max();
    z[1] = std::numeric_limits<T>::max();
    // compute lowest envelope:
    for (int q = 1; q <= n - 1; q++) {
      T s = T(0);
      k++;  // this compensates for first line of next do-while block
      do {
        k--;
        // compute horizontal position of intersection between the parabola from
        // q and the current lowest parabola
        s = ((f[q] + T(q * q)) - (f[v[k]] + T(v[k] * v[k]))) /
            T(2 * (q - v[k]));
      } while (s <= z[k]);
      k++;
      v[k] = q;
      z[k] = s;
      z[k + 1] = std::numeric_limits<T>::max();
    }
    // fill in values of distance transform
    k = 0;
    for (int q = 0; q <= n - 1; q++) {
      while (z[k + 1] < T(q)) {
        k++;
      }
      d[q] = T(std::pow(q - v[k], 2)) + f[v[k]];
    }
  }

 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE EuclideanDistanceTransformGenerator(
      const typename TTypes<T, 4>::ConstTensor& input)
      : input_(input) {
    height_ = input.dimension(1);
    width_ = input.dimension(2);
  }

  void operator()(typename TTypes<T, 4>::Tensor& dp, int64 start_batch,
                  int64 end_batch) {
    for (int k = start_batch; k < end_batch; k++) {
      // init edt matrix
      for (int i = 0; i < height_; i++) {
        for (int j = 0; j < width_; j++) {
          if (input_({k, i, j, 0}) == T(0)) {
            dp({k, i, j, 0}) = T(0);
          } else {
            dp({k, i, j, 0}) = std::numeric_limits<T>::max();
          }
        }
      }
      std::vector<T> f(std::max(height_, width_));
      std::vector<T> d(width_);
      for (int i = 0; i < height_; i++) {
        for (int j = 0; j < width_; j++) {
          f[j] = dp({k, i, j, 0});
        }
        distance(f, d);
        for (int j = 0; j < width_; j++) {
          dp({k, i, j, 0}) = d[j];
        }
      }
      d.resize(height_);
      for (int j = 0; j < width_; j++) {
        for (int i = 0; i < height_; i++) {
          f[i] = dp({k, i, j, 0});
        }
        distance(f, d);
        for (int i = 0; i < height_; i++) {
          dp({k, i, j, 0}) = d[i];
        }
      }
    }
  }
};

}  // end namespace generator

namespace functor {

using generator::EuclideanDistanceTransformGenerator;

template <typename Device, typename T>
struct EuclideanDistanceTransformFunctor {
  typedef typename TTypes<T, 4>::ConstTensor InputType;
  typedef typename TTypes<T, 4>::Tensor OutputType;

  EuclideanDistanceTransformFunctor() {}

  EIGEN_ALWAYS_INLINE
  void operator()(OpKernelContext* ctx, OutputType* output,
                  const InputType& images) const {
    auto edt_generator = EuclideanDistanceTransformGenerator<Device, T>(images);

    auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
    thread_pool->ParallelFor(
        images.dimension(0), images.dimension(1) * images.dimension(2),
        [&edt_generator, &output](int64 start_batch, int64 end_batch) {
          edt_generator(*output, start_batch, end_batch);
        });
    output->device(ctx->eigen_device<Device>()) = output->sqrt();
  }
};

}  // end namespace functor

}  // end namespace addons
}  // end namespace tensorflow

#endif  // TENSORFLOW_ADDONS_IMAGE_KERNELS_EUCLIDEAN_DISTANCE_TRANSFORM_OP_H_
