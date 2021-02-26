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

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace addons {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace generator {

template <typename Device, typename T>
class EuclideanDistanceTransformGenerator {
 private:
  typename TTypes<T, 4>::ConstTensor input_;
  int64 height_, width_, channel_, max_;

  EIGEN_DEVICE_FUNC void distance(const Eigen::Array<T, Eigen::Dynamic, 1>& f,
                                  Eigen::Array<T, Eigen::Dynamic, 1>& d,
                                  int n) {
    Eigen::Array<int, Eigen::Dynamic, 1> v(
        n);  // locations of parabolas in lower envelope
    Eigen::Array<T, Eigen::Dynamic, 1> z(
        n + 1);  // locations of boundaries between parabolas
    int k = 0;   // index of rightmost parabola in lower envelope
    v(0) = 0;
    z(0) = -Eigen::NumTraits<T>::highest();
    z(1) = Eigen::NumTraits<T>::highest();
    // compute lowest envelope:
    for (int q = 1; q <= n - 1; q++) {
      T s = static_cast<T>(0);
      k++;  // this compensates for first line of next do-while block
      do {
        k--;
        // compute horizontal position of intersection between the parabola from
        // q and the current lowest parabola
        s = ((f(q) + static_cast<T>(q * q)) -
             (f(v(k)) + static_cast<T>(v(k) * v(k)))) /
            static_cast<T>(2 * (q - v(k)));
      } while (s <= z(k));
      k++;
      v(k) = q;
      z(k) = s;
      z(k + 1) = Eigen::NumTraits<T>::highest();
    }
    // fill in values of distance transform
    k = 0;
    for (int q = 0; q <= n - 1; q++) {
      while (z(k + 1) < static_cast<T>(q)) {
        k++;
      }
      d(q) = static_cast<T>(Eigen::numext::pow(q - v(k), 2)) + f(v(k));
    }
  }

 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE EuclideanDistanceTransformGenerator(
      const typename TTypes<T, 4>::ConstTensor& input)
      : input_(input) {
    height_ = input.dimension(1);
    width_ = input.dimension(2);
    channel_ = input.dimension(3);
    max_ = Eigen::numext::maxi(height_, width_);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void operator()(
      typename TTypes<T, 4>::Tensor& dp, int k) {
    // init edt matrix
    for (int i = 0; i < height_; i++) {
      for (int j = 0; j < width_; j++) {
        for (int c = 0; c < channel_; c++) {
          if (input_({k, i, j, c}) == static_cast<T>(0)) {
            dp({k, i, j, c}) = static_cast<T>(0);
          } else {
            dp({k, i, j, c}) = Eigen::NumTraits<T>::highest();
          }
        }
      }
    }
    Eigen::Array<T, Eigen::Dynamic, 1> f(max_);
    Eigen::Array<T, Eigen::Dynamic, 1> d(max_);
    for (int c = 0; c < channel_; c++) {
      for (int i = 0; i < height_; i++) {
        for (int j = 0; j < width_; j++) {
          f(j) = dp({k, i, j, c});
        }
        distance(f, d, width_);
        for (int j = 0; j < width_; j++) {
          dp({k, i, j, c}) = d(j);
        }
      }
      for (int j = 0; j < width_; j++) {
        for (int i = 0; i < height_; i++) {
          f(i) = dp({k, i, j, c});
        }
        distance(f, d, height_);
        for (int i = 0; i < height_; i++) {
          dp({k, i, j, c}) = Eigen::numext::sqrt(d(i));
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
  void operator()(OpKernelContext* ctx, OutputType* output,
                  const InputType& images) const;
};
}  // end namespace functor

}  // end namespace addons
}  // end namespace tensorflow

#endif  // TENSORFLOW_ADDONS_IMAGE_KERNELS_EUCLIDEAN_DISTANCE_TRANSFORM_OP_H_
