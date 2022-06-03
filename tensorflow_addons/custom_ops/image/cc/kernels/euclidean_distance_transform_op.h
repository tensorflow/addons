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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace addons {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define GET_INDEX(i, j, k, c) \
  (k * height * width * channels + i * width * channels + j * channels + c)

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void Distance(const T* f, T* d, int* v,
                                                    T* z, int n) {
  // index of rightmost parabola in lower envelope
  int k = 0;
  v[0] = 0;
  z[0] = Eigen::NumTraits<T>::lowest();
  z[1] = Eigen::NumTraits<T>::highest();
  // compute lowest envelope:
  for (int q = 1; q <= n - 1; q++) {
    T s(0);
    k++;  // this compensates for first line of next do-while block
    do {
      k--;
      // compute horizontal position of intersection between the parabola from
      // q and the current lowest parabola
      s = (f[q] - f[v[k]]) / static_cast<T>(2 * (q - v[k])) +
          static_cast<T>((q + v[k]) / 2.0);
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
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void EuclideanDistanceTransformSample(
    const uint8* input, T* output, int k, int c, int height, int width,
    int channels) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = GET_INDEX(i, j, k, c);
      if (input[index] == 0) {
        output[index] = static_cast<T>(0);
      } else {
        output[index] = Eigen::NumTraits<T>::highest();
      }
    }
  }
  int max = Eigen::numext::maxi(height, width);
  T* f = new T[max];
  T* d = new T[max];
  // locations of parabolas in lower envelope
  int* vw = new int[width];
  int* vh = new int[height];
  // locations of boundaries between parabolas
  T* zw = new T[width + 1];
  T* zh = new T[height + 1];
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int index = GET_INDEX(i, j, k, c);
      f[j] = output[index];
    }
    Distance<T>(f, d, vw, zw, width);
    for (int j = 0; j < width; j++) {
      int index = GET_INDEX(i, j, k, c);
      if (Eigen::numext::isinf(d[j])) {
        d[j] = Eigen::NumTraits<T>::highest();
      }
      output[index] = d[j];
    }
  }
  for (int j = 0; j < width; j++) {
    for (int i = 0; i < height; i++) {
      int index = GET_INDEX(i, j, k, c);
      f[i] = output[index];
    }
    Distance<T>(f, d, vh, zh, height);
    for (int i = 0; i < height; i++) {
      int index = GET_INDEX(i, j, k, c);
      if (Eigen::numext::isinf(d[i])) {
        d[i] = Eigen::NumTraits<T>::highest();
      }
      output[index] = Eigen::numext::sqrt(d[i]);
    }
  }
  delete[] f;
  delete[] d;
  delete[] vh;
  delete[] vw;
  delete[] zh;
  delete[] zw;
}

namespace functor {
template <typename Device, typename T>
struct EuclideanDistanceTransformFunctor {
  typedef typename TTypes<uint8, 4>::ConstTensor InputType;
  typedef typename TTypes<T, 4>::Tensor OutputType;
  void operator()(OpKernelContext* ctx, OutputType* output,
                  const InputType& images) const;
};
}  // end namespace functor

}  // end namespace addons
}  // end namespace tensorflow

#endif  // TENSORFLOW_ADDONS_IMAGE_KERNELS_EUCLIDEAN_DISTANCE_TRANSFORM_OP_H_
