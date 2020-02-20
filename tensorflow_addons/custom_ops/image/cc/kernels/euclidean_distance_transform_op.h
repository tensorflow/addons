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

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace addons {

namespace generator {

using Eigen::array;
using Eigen::DenseIndex;
using Eigen::numext::mini;
using Eigen::numext::sqrt;

template <typename Device, typename T>
class EuclideanDistanceTransformGenerator {
 private:
  typename TTypes<T, 4>::ConstTensor input_;
  int64 height_, width_;

 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  EuclideanDistanceTransformGenerator(typename TTypes<T, 4>::ConstTensor input)
      : input_(input) {
    height_ = input_.dimension(1);
    width_ = input_.dimension(2);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const array<DenseIndex, 4> &coords) const {
    const int64 x = coords[1];
    const int64 y = coords[2];

    if (input_(coords) == T(0)) return T(0);

    T minDistance = Eigen::NumTraits<T>::highest();

    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
        if (input_({coords[0], h, w, coords[3]}) == T(0)) {
          T dist = sqrt(T((x - h) * (x - h) + (y - w) * (y - w)));
          minDistance = mini(minDistance, dist);
        }
      }
    }
    return minDistance;
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
  void operator()(const Device &device, OutputType *output,
                  const InputType &images) const {
    output->device(device) = output->generate(
        EuclideanDistanceTransformGenerator<Device, T>(images));
  }
};

}  // end namespace functor

}  // end namespace addons
}  // end namespace tensorflow

#endif  // TENSORFLOW_ADDONS_IMAGE_KERNELS_EUCLIDEAN_DISTANCE_TRANSFORM_OP_H_
