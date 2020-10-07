// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLECONV2D_OP_H_
#define TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLECONV2D_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace addons {
using Shape8D = Eigen::array<Eigen::DenseIndex, 8>;
using Shape7D = Eigen::array<Eigen::DenseIndex, 7>;
using Shape6D = Eigen::array<Eigen::DenseIndex, 6>;
using Shape5D = Eigen::array<Eigen::DenseIndex, 5>;
using Shape4D = Eigen::array<Eigen::DenseIndex, 4>;
using Shape3D = Eigen::array<Eigen::DenseIndex, 3>;
using Shape2D = Eigen::array<Eigen::DenseIndex, 2>;
using Shape1D = Eigen::array<Eigen::DenseIndex, 1>;

template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
using EigenTensor = Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>;
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
using EigenTensorRef =
    Eigen::TensorRef<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>>;

static const int kMaxParallelImgs = 32;

struct DeformableConv2DParams {
  int32 input_batches;
  int32 input_channels;
  int32 input_rows;
  int32 input_cols;
  int32 filter_channels;
  int32 filter_rows;
  int32 filter_cols;
  int32 padding_rows;
  int32 padding_cols;
  int32 stride_rows;
  int32 stride_cols;
  int32 dilation_rows;
  int32 dilation_cols;
  int32 output_channels;
  int32 output_rows;
  int32 output_cols;
  int32 parallel_imgs;
  int32 weight_groups;
  int32 offset_groups;
};

namespace functor {

template <typename Device, typename Dtype>
struct DeformableConv2DFunctorBase {
  DeformableConv2DFunctorBase(
      typename TTypes<Dtype, 4>::ConstTensor input_tensor,
      typename TTypes<Dtype, 4>::ConstTensor filter_tensor,
      typename TTypes<Dtype, 1>::ConstTensor bias_tensor,
      typename TTypes<Dtype, 4>::ConstTensor offset_tensor,
      typename TTypes<Dtype, 4>::ConstTensor mask_tensor,
      typename TTypes<Dtype, 4>::Tensor column_buffer_tensor,
      DeformableConv2DParams _p)
      : _input_tensor(input_tensor),
        _filter_tensor(filter_tensor),
        _bias_tensor(bias_tensor),
        _offset_tensor(offset_tensor),
        _mask_tensor(mask_tensor),
        _column_buffer_tensor(column_buffer_tensor),
        p(_p) {}

  virtual Status operator()(OpKernelContext* context) = 0;

  Dtype BilinearInterpolate(int32 batch, int32 channel, Dtype y, Dtype x) {
    EigenTensor<Dtype, 2> img = _input_tensor.chip(batch, 0).chip(channel, 0);

    auto max_height = img.dimension(0);
    auto max_width = img.dimension(1);

    if (y <= -1 || max_height <= y || x <= -1 || max_width <= x) {
      return Dtype(0);
    }

    int y_low = floor(y);
    int x_low = floor(x);
    int y_high = y_low + 1;
    int w_high = x_low + 1;

    auto v1 = Dtype(0);
    if (y_low >= 0 && x_low >= 0) {
      v1 = img(y_low, x_low);
    }

    auto v2 = Dtype(0);
    if (y_low >= 0 && w_high <= max_width - 1) {
      v2 = img(y_low, w_high);
    }

    auto v3 = Dtype(0);
    if (y_high <= max_height - 1 && x_low >= 0) {
      v3 = img(y_high, x_low);
    }

    auto v4 = Dtype(0);
    if (y_high <= max_height - 1 && w_high <= max_width - 1) {
      v4 = img(y_high, w_high);
    }

    auto lh = y - y_low;
    auto lw = x - x_low;
    auto hh = 1 - lh;
    auto hw = 1 - lw;

    auto w1 = hh * hw;
    auto w2 = hh * lw;
    auto w3 = lh * hw;
    auto w4 = lh * lw;

    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
  }

  void DeformableIm2Col(int32 b) {
    auto use_mask = _mask_tensor.dimension(0) > 0;
    auto num_kernels =
        p.input_channels * p.output_rows * p.output_cols * p.parallel_imgs;
    auto batches = p.input_batches / p.parallel_imgs;

    EigenTensor<Dtype, 7> offset_tensor =
        _offset_tensor
            .reshape(Shape8D({batches, p.parallel_imgs, p.offset_groups,
                              p.filter_rows, p.filter_cols, 2, p.output_rows,
                              p.output_cols}))
            .chip(b, 0);

    EigenTensor<Dtype, 6> mask_tensor =
        use_mask ? static_cast<EigenTensor<Dtype, 6>>(
                       _mask_tensor
                           .reshape(Shape7D({batches, p.parallel_imgs,
                                             p.offset_groups, p.filter_rows,
                                             p.filter_cols, p.output_rows,
                                             p.output_cols}))
                           .chip(b, 0))
                 : _mask_tensor.reshape(Shape6D({0, 0, 0, 0, 0, 0}));

    for (auto k = 0; k < num_kernels; k++) {
      const auto current_output_col = k % p.output_cols;
      const auto current_output_row = (k / p.output_cols) % p.output_rows;
      const auto current_batch =
          (k / (p.output_rows * p.output_cols)) % p.parallel_imgs;
      const auto current_input_channel =
          k / (p.output_rows * p.output_cols * p.parallel_imgs);
      const auto current_output_channel =
          current_input_channel * p.filter_rows * p.filter_cols;

      auto current_actual_batch = b * p.parallel_imgs + current_batch;

      const auto group_index =
          current_input_channel / (p.input_channels / p.offset_groups);

      EigenTensor<Dtype, 5> offset_tensor_chipped =
          offset_tensor.chip(current_batch, 0).chip(group_index, 0);

      EigenTensor<Dtype, 4> mask_tensor_chipped =
          use_mask
              ? static_cast<EigenTensor<Dtype, 4>>(
                    mask_tensor.chip(current_batch, 0).chip(group_index, 0))
              : mask_tensor.reshape(Shape4D({0, 0, 0, 0}));

      auto column_buffer_tensor_channel = current_output_channel;
      for (auto current_filter_row = 0; current_filter_row < p.filter_rows;
           current_filter_row++) {
        for (auto current_filter_col = 0; current_filter_col < p.filter_cols;
             current_filter_col++) {
          auto offset_h =
              offset_tensor_chipped(current_filter_row, current_filter_col, 0,
                                    current_output_row, current_output_col);
          auto offset_w =
              offset_tensor_chipped(current_filter_row, current_filter_col, 1,
                                    current_output_row, current_output_col);

          auto mask = use_mask ? mask_tensor_chipped(
                                     current_filter_row, current_filter_col,
                                     current_output_row, current_output_col)
                               : Dtype(1);

          auto y = (current_output_row * p.stride_rows - p.padding_rows) +
                   current_filter_row * p.dilation_rows + offset_h;
          auto x = (current_output_col * p.stride_cols - p.padding_cols) +
                   current_filter_col * p.dilation_cols + offset_w;

          _column_buffer_tensor(column_buffer_tensor_channel, current_batch,
                                current_output_row, current_output_col) =
              mask * BilinearInterpolate(current_actual_batch,
                                         current_input_channel, y, x);
          column_buffer_tensor_channel++;
        }
      }
    }
  }

  typename TTypes<Dtype, 4>::ConstTensor _input_tensor;
  typename TTypes<Dtype, 4>::ConstTensor _filter_tensor;
  typename TTypes<Dtype, 1>::ConstTensor _bias_tensor;
  typename TTypes<Dtype, 4>::ConstTensor _offset_tensor;
  typename TTypes<Dtype, 4>::ConstTensor _mask_tensor;
  typename TTypes<Dtype, 4>::Tensor _column_buffer_tensor;
  DeformableConv2DParams p;
};

template <typename Device, typename Dtype>
struct DeformableConv2DFunctor
    : public DeformableConv2DFunctorBase<Device, Dtype> {
  DeformableConv2DFunctor(
      typename TTypes<Dtype, 4>::ConstTensor input_tensor,
      typename TTypes<Dtype, 4>::ConstTensor filter_tensor,
      typename TTypes<Dtype, 1>::ConstTensor bias_tensor,
      typename TTypes<Dtype, 4>::ConstTensor offset_tensor,
      typename TTypes<Dtype, 4>::ConstTensor mask_tensor,
      typename TTypes<Dtype, 4>::Tensor column_buffer_tensor,
      typename TTypes<Dtype, 4>::Tensor output_tensor,
      DeformableConv2DParams _p);

  Status operator()(OpKernelContext* context);

  typename TTypes<Dtype, 4>::Tensor _output_tensor;
};

template <typename Device, typename Dtype>
struct DeformableConv2DGradFunctor
    : public DeformableConv2DFunctorBase<Device, Dtype> {
  DeformableConv2DGradFunctor(
      typename TTypes<Dtype, 4>::ConstTensor input_tensor,
      typename TTypes<Dtype, 4>::ConstTensor filter_tensor,
      typename TTypes<Dtype, 1>::ConstTensor bias_tensor,
      typename TTypes<Dtype, 4>::ConstTensor offset_tensor,
      typename TTypes<Dtype, 4>::ConstTensor mask_tensor,
      typename TTypes<Dtype, 4>::ConstTensor output_grad_tensor,
      typename TTypes<Dtype, 4>::Tensor input_grad_tensor,
      typename TTypes<Dtype, 4>::Tensor filter_grad_tensor,
      typename TTypes<Dtype, 1>::Tensor bias_grad_tensor,
      typename TTypes<Dtype, 4>::Tensor offset_grad_tensor,
      typename TTypes<Dtype, 4>::Tensor mask_grad_tensor,
      typename TTypes<Dtype, 4>::Tensor column_buffer_tensor,
      DeformableConv2DParams p);

  Status operator()(OpKernelContext* context);

  typename TTypes<Dtype, 4>::ConstTensor _output_grad_tensor;
  typename TTypes<Dtype, 4>::Tensor _input_grad_tensor;
  typename TTypes<Dtype, 4>::Tensor _filter_grad_tensor;
  typename TTypes<Dtype, 1>::Tensor _bias_grad_tensor;
  typename TTypes<Dtype, 4>::Tensor _offset_grad_tensor;
  typename TTypes<Dtype, 4>::Tensor _mask_grad_tensor;
};

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif  // TENSORFLOW_ADDONS_LAYERS_KERNELS_DEFORMABLECONV2D_OP_H_
