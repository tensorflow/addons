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

#ifndef TENSORFLOW_ADDONS_IMAGE_KERNELS_IMAGE_PROJECTIVE_TRANSFORM_OP_H__
#define TENSORFLOW_ADDONS_IMAGE_KERNELS_IMAGE_PROJECTIVE_TRANSFORM_OP_H__

// See docs in ../ops/image_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace addons {

namespace generator {

enum Interpolation { INTERPOLATION_NEAREST, INTERPOLATION_BILINEAR };
enum Extend {
  EXTEND_REFLECT,
  EXTEND_CONSTANT,
  EXTEND_NEAREST,
  EXTEND_MIRROR,
  EXTEND_WRAP
};

using Eigen::array;
using Eigen::DenseIndex;

template <typename Device, typename T>
class ProjectiveGenerator {
 private:
  typename TTypes<T, 4>::ConstTensor input_;
  typename TTypes<float>::ConstMatrix transforms_;
  const Interpolation interpolation_;
  const Extend extend_;
  const T constant_values_;

 public:
  static const int kNumParameters = 8;

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  ProjectiveGenerator(typename TTypes<T, 4>::ConstTensor input,
                      typename TTypes<float>::ConstMatrix transforms,
                      const Interpolation interpolation, const Extend extend,
                      const T constant_values)
      : input_(input),
        transforms_(transforms),
        interpolation_(interpolation),
        extend_(extend),
        constant_values_(constant_values) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const array<DenseIndex, 4>& coords) const {
    const int64 output_y = coords[1];
    const int64 output_x = coords[2];
    const float* transform =
        transforms_.dimension(0) == 1
            ? transforms_.data()
            : &transforms_.data()[transforms_.dimension(1) * coords[0]];
    float projection = transform[6] * output_x + transform[7] * output_y + 1.f;
    if (projection == 0) {
      // Return the constant_values_ for infinite coordinates,
      // which are outside the input image
      return constant_values_;
    }
    const float input_x =
        (transform[0] * output_x + transform[1] * output_y + transform[2]) /
        projection;
    const float input_y =
        (transform[3] * output_x + transform[4] * output_y + transform[5]) /
        projection;

    switch (interpolation_) {
      case INTERPOLATION_NEAREST:
        // Switch the order of x and y again for indexing into the image.
        return nearest_interpolation(coords[0], input_y, input_x, coords[3]);
      case INTERPOLATION_BILINEAR:
        return bilinear_interpolation(coords[0], input_y, input_x, coords[3]);
    }
    // Unreachable; ImageProjectiveTransform only uses INTERPOLATION_NEAREST
    // or INTERPOLATION_BILINEAR.
    return constant_values_;
  }

 private:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  nearest_interpolation(const DenseIndex batch, const float y, const float x,
                        const DenseIndex channel) const {
    return read_with_fill_value(batch, DenseIndex(std::round(y)),
                                DenseIndex(std::round(x)), channel);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  bilinear_interpolation(const DenseIndex batch, const float y, const float x,
                         const DenseIndex channel) const {
    const float y_floor = std::floor(y);
    const float x_floor = std::floor(x);
    const float y_ceil = y_floor + 1;
    const float x_ceil = x_floor + 1;
    // f(x, y_floor) = (x_ceil - x) / (x_ceil - x_floor) * f(x_floor, y_floor)
    //               + (x - x_floor) / (x_ceil - x_floor) * f(x_ceil, y_floor)
    const float value_yfloor =
        (x_ceil - x) *
            static_cast<float>(read_with_fill_value(
                batch, DenseIndex(y_floor), DenseIndex(x_floor), channel)) +
        (x - x_floor) *
            static_cast<float>(read_with_fill_value(
                batch, DenseIndex(y_floor), DenseIndex(x_ceil), channel));
    // f(x, y_ceil) = (x_ceil - x) / (x_ceil - x_floor) * f(x_floor, y_ceil)
    //              + (x - x_floor) / (x_ceil - x_floor) * f(x_ceil, y_ceil)
    const float value_yceil =
        (x_ceil - x) *
            static_cast<float>(read_with_fill_value(
                batch, DenseIndex(y_ceil), DenseIndex(x_floor), channel)) +
        (x - x_floor) *
            static_cast<float>(read_with_fill_value(
                batch, DenseIndex(y_ceil), DenseIndex(x_ceil), channel));
    // f(x, y) = (y_ceil - y) / (y_ceil - y_floor) * f(x, y_floor)
    //         + (y - y_floor) / (y_ceil - y_floor) * f(x, y_ceil)
    return T((y_ceil - y) * value_yfloor + (y - y_floor) * value_yceil);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE DenseIndex
  map_coordinate(const DenseIndex in, const DenseIndex dim) const {
    DenseIndex out(in);

    if (in < 0) {
      switch (extend_) {
        case EXTEND_MIRROR:
          if (dim <= 1) {
            out = 0;
          } else {
            const DenseIndex d(2 * dim - 2);
            out = d * (-out / d) + out;
            out = (out <= 1 - dim) ? out + d : -out;
          }
          break;
        case EXTEND_REFLECT:
          if (dim <= 1) {
            out = 0;
          } else {
            const DenseIndex d(2 * dim);
            if (out < -d) {
              out = d * (-out / d) + out;
            }
            out = (out < -dim) ? out + d : -out - 1;
          }
          break;
        case EXTEND_WRAP:
          if (dim <= 1) {
            out = 0;
          } else {
            const DenseIndex d(dim - 1);
            out += d * ((-out / d) + 1);
          }
          break;
        case EXTEND_NEAREST:
          out = 0;
          break;
        case EXTEND_CONSTANT:
          out = -1;
          break;
      }
    } else if (in >= dim) {
      switch (extend_) {
        case EXTEND_MIRROR:
          if (dim <= 1) {
            out = 0;
          } else {
            const DenseIndex d(2 * dim - 2);
            out -= d * (out / d);
            out = (out >= dim) ? d - out : out;
          }
          break;
        case EXTEND_REFLECT:
          if (dim <= 1) {
            out = 0;
          } else {
            const DenseIndex d(2 * dim);
            out -= d * (out / d);
            out = (out >= dim) ? d - out - 1 : out;
          }
          break;
        case EXTEND_WRAP:
          if (dim <= 1) {
            out = 0;
          } else {
            const DenseIndex d(dim - 1);
            out -= d * (out / d);
          }
          break;
        case EXTEND_NEAREST:
          out = dim - 1;
          break;
        case EXTEND_CONSTANT:
          out = -1;
          break;
      }
    }

    return out;
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  read_with_fill_value(const DenseIndex batch, const DenseIndex y,
                       const DenseIndex x, const DenseIndex channel) const {
    // batch and channel must be correct, because they are passed unchanged from
    // the input.
    const DenseIndex my(map_coordinate(y, input_.dimension(1))),
        mx(map_coordinate(x, input_.dimension(2)));

    return (my >= 0 && mx >= 0)
               ? input_(array<DenseIndex, 4>{batch, my, mx, channel})
               : constant_values_;
  }
};

}  // end namespace generator

// NOTE(ringwalt): We MUST wrap the generate() call in a functor and explicitly
// instantiate the functor in image_ops_gpu.cu.cc. Otherwise, we will be missing
// some Eigen device code.
namespace functor {

using generator::Extend;
using generator::Interpolation;
using generator::ProjectiveGenerator;

template <typename Device, typename T>
struct FillProjectiveTransform {
  typedef typename TTypes<T, 4>::Tensor OutputType;
  typedef typename TTypes<T, 4>::ConstTensor InputType;
  typedef typename TTypes<float, 2>::ConstTensor TransformsType;
  const Interpolation interpolation_;
  const Extend extend_;
  const T constant_values_;

  FillProjectiveTransform(const Interpolation interpolation,
                          const Extend extend, const T constant_values)
      : interpolation_(interpolation),
        extend_(extend),
        constant_values_(constant_values) {}

  EIGEN_ALWAYS_INLINE
  void operator()(const Device& device, OutputType* output,
                  const InputType& images,
                  const TransformsType& transform) const {
    output->device(device) = output->generate(ProjectiveGenerator<Device, T>(
        images, transform, interpolation_, extend_, constant_values_));
  }
};

}  // end namespace functor

}  // end namespace addons
}  // end namespace tensorflow

#endif  // TENSORFLOW_ADDONS_IMAGE_KERNELS_IMAGE_PROJECTIVE_TRANSFORM_OP_H__
