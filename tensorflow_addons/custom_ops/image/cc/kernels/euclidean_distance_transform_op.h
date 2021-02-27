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

#define GETINDEX(i, j, k, c, height, width, channels) \
  (k * height * width * channels + i * width * channels + j * channels + c)

template <typename T>
EIGEN_DEVICE_FUNC void distance(const T* f, T* d, int n) {
  // locations of parabolas in lower envelope
  int* v = new int[n];
  // locations of boundaries between parabolas
  T* z = new T[n + 1];
  // index of rightmost parabola in lower envelope
  int k = 0;
  v[0] = 0;
  z[0] = -Eigen::NumTraits<T>::highest();
  z[1] = Eigen::NumTraits<T>::highest();
  // compute lowest envelope:
  for (int q = 1; q <= n - 1; q++) {
    T s = static_cast<T>(0);
    k++;  // this compensates for first line of next do-while block
    do {
      k--;
      // compute horizontal position of intersection between the parabola from
      // q and the current lowest parabola
      s = ((f[q] + static_cast<T>(q * q)) -
           (f[v[k]] + static_cast<T>(v[k] * v[k]))) /
          static_cast<T>(2 * (q - v[k]));
    } while (s <= z[k]);
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Eigen::NumTraits<T>::highest();
  }
  // fill in values of distance transform
  k = 0;
  for (int q = 0; q <= n - 1; q++) {
    while (z[k + 1] < static_cast<T>(q)) {
      k++;
    }
    d[q] = static_cast<T>(Eigen::numext::pow(q - v[k], 2)) + f[v[k]];
  }
  delete v;
  delete z;
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void edt_sample(const T* input, T* output,
                                                      int k, int c, int height,
                                                      int width, int channels) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      auto index = GETINDEX(i, j, k, c, height, width, channels);
      if (input[index] == static_cast<T>(0)) {
        output[index] = static_cast<T>(0);
      } else {
        output[index] = Eigen::NumTraits<T>::highest();
      }
    }
  }
  int max = Eigen::numext::maxi(height, width);
  T* f = new T[max];
  T* d = new T[max];
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      auto index = GETINDEX(i, j, k, c, height, width, channels);
      f[j] = output[index];
    }
    distance<T>(f, d, width);
    for (int j = 0; j < width; j++) {
      auto index = GETINDEX(i, j, k, c, height, width, channels);
      output[index] = d[j];
    }
  }
  for (int j = 0; j < width; j++) {
    for (int i = 0; i < height; i++) {
      auto index = GETINDEX(i, j, k, c, height, width, channels);
      f[i] = output[index];
    }
    distance<T>(f, d, height);
    for (int i = 0; i < height; i++) {
      auto index = GETINDEX(i, j, k, c, height, width, channels);
      output[index] = Eigen::numext::sqrt(d[i]);
    }
  }
  delete f;
  delete d;
}

namespace functor {
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
