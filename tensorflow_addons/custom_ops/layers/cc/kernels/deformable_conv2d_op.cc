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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow_addons/custom_ops/layers/cc/kernels/deformable_conv2d_op.h"

#include <array>
#include <mutex>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/kernel_shape_util.h"

namespace tensorflow {
namespace addons {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#if GOOGLE_CUDA
#define EXTERN_TEMPLATE(T)                           \
  extern template Status Transpose<GPUDevice, T, 5>( \
      OpKernelContext * ctx, const Tensor &in,       \
      const gtl::ArraySlice<int32> perm, Tensor *out);
TF_CALL_float(EXTERN_TEMPLATE);
TF_CALL_double(EXTERN_TEMPLATE);
#undef EXTERN_TEMPLATE
#endif  // GOOGLE_CUDA

namespace functor {

#if GOOGLE_CUDA
#define EXTERN_TEMPLATE(T)                                             \
  extern template struct DeformableConv2DForwardFunctor<GPUDevice, T>; \
  extern template struct DeformableConv2DGradFunctor<GPUDevice, T>;
TF_CALL_float(EXTERN_TEMPLATE);
TF_CALL_double(EXTERN_TEMPLATE);
#undef EXTERN_TEMPLATE
#endif  // GOOGLE_CUDA

#define IM2COL(T)                                                              \
  template <>                                                                  \
  void DeformableConv2DFunctorBase<CPUDevice, T>::DeformableIm2Col(            \
      OpKernelContext *context, int32 b) {                                     \
    auto num_kernels =                                                         \
        p.input_channels * p.output_rows * p.output_cols * p.parallel_imgs;    \
                                                                               \
    const auto offset_eigen_tensor = offset_tensor.tensor<T, 8>();             \
                                                                               \
    const auto mask_eigen_tensor =                                             \
        p.use_mask ? mask_tensor.tensor<T, 7>()                                \
                   : mask_tensor.shaped<T, 7>({0, 0, 0, 0, 0, 0, 0});          \
                                                                               \
    const auto input_eigen_tensor = input_tensor.tensor<T, 5>();               \
                                                                               \
    auto column_buffer_eigen_tensor = column_buffer_tensor.tensor<T, 4>();     \
                                                                               \
    const auto cost = p.filter_rows * p.filter_cols;                           \
    const auto work = [&](Eigen::Index start, Eigen::Index end) -> void {      \
      for (Eigen::Index k = start; k < end; ++k) {                             \
        const auto current_output_col = k % p.output_cols;                     \
        const auto current_output_row = (k / p.output_cols) % p.output_rows;   \
        const auto current_batch =                                             \
            (k / (p.output_rows * p.output_cols)) % p.parallel_imgs;           \
        const auto current_input_channel =                                     \
            k / (p.output_rows * p.output_cols * p.parallel_imgs);             \
        const auto current_output_channel =                                    \
            current_input_channel * p.filter_rows * p.filter_cols;             \
                                                                               \
        const auto current_actual_batch = b * p.parallel_imgs + current_batch; \
                                                                               \
        const auto group_index =                                               \
            current_input_channel / (p.input_channels / p.offset_groups);      \
                                                                               \
        auto column_buffer_tensor_channel = current_output_channel;            \
        for (auto current_filter_row = 0; current_filter_row < p.filter_rows;  \
             current_filter_row++) {                                           \
          for (auto current_filter_col = 0;                                    \
               current_filter_col < p.filter_cols; current_filter_col++) {     \
            auto offset_h =                                                    \
                offset_eigen_tensor(b, current_batch, group_index,             \
                                    current_filter_row, current_filter_col, 0, \
                                    current_output_row, current_output_col);   \
            auto offset_w =                                                    \
                offset_eigen_tensor(b, current_batch, group_index,             \
                                    current_filter_row, current_filter_col, 1, \
                                    current_output_row, current_output_col);   \
                                                                               \
            auto mask = p.use_mask                                             \
                            ? mask_eigen_tensor(                               \
                                  b, current_batch, group_index,               \
                                  current_filter_row, current_filter_col,      \
                                  current_output_row, current_output_col)      \
                            : T(1);                                            \
                                                                               \
            auto y = (current_output_row * p.stride_rows - p.padding_rows) +   \
                     current_filter_row * p.dilation_rows + offset_h;          \
            auto x = (current_output_col * p.stride_cols - p.padding_cols) +   \
                     current_filter_col * p.dilation_cols + offset_w;          \
                                                                               \
            column_buffer_eigen_tensor(column_buffer_tensor_channel,           \
                                       current_batch, current_output_row,      \
                                       current_output_col) =                   \
                mask * BilinearInterpolate<T>(input_eigen_tensor, b,           \
                                              current_actual_batch,            \
                                              current_input_channel, y, x);    \
            column_buffer_tensor_channel++;                                    \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    };                                                                         \
    auto thread_pool =                                                         \
        context->device()->tensorflow_cpu_worker_threads()->workers;           \
    thread_pool->ParallelFor(num_kernels, cost, work);                         \
  }
TF_CALL_float(IM2COL);
TF_CALL_double(IM2COL);
#undef IM2COL

#define COL2IM_OFFSET_AND_MASK(T)                                              \
  template <>                                                                  \
  void                                                                         \
  DeformableConv2DGradFunctor<CPUDevice, T>::DeformableCol2ImForOffsetAndMask( \
      OpKernelContext *context, int32 b) {                                     \
    const auto num_kernels = p.output_rows * p.output_cols * 2 *               \
                             p.filter_rows * p.filter_cols * p.offset_groups * \
                             p.parallel_imgs;                                  \
                                                                               \
    const auto offset_eigen_tensor = offset_tensor.template tensor<T, 8>();    \
                                                                               \
    const auto mask_eigen_tensor =                                             \
        p.use_mask ? mask_tensor.template tensor<T, 7>()                       \
                   : mask_tensor.template shaped<T, 7>({0, 0, 0, 0, 0, 0, 0}); \
                                                                               \
    const auto column_buffer_eigen_tensor =                                    \
        column_buffer_tensor.template shaped<T, 6>(                            \
            {p.input_channels, p.filter_rows, p.filter_cols, p.parallel_imgs,  \
             p.output_rows, p.output_cols});                                   \
                                                                               \
    auto offset_grad_eigen_tensor = offset_grad_tensor.tensor<T, 4>();         \
    auto mask_grad_eigen_tensor = mask_grad_tensor.tensor<T, 4>();             \
                                                                               \
    const auto input_eigen_tensor = input_tensor.tensor<T, 5>();               \
                                                                               \
    const auto cost = p.input_channels / p.offset_groups;                      \
    const auto work = [&](Eigen::Index start, Eigen::Index end) -> void {      \
      for (Eigen::Index k = start; k < end; ++k) {                             \
        auto offset_grad_value = T(0);                                         \
        auto mask_grad_value = T(0);                                           \
                                                                               \
        const auto offset_channels =                                           \
            2 * p.filter_rows * p.filter_cols * p.offset_groups;               \
                                                                               \
        const auto offset_channel_step = p.filter_rows * p.filter_cols;        \
                                                                               \
        const auto current_output_col = k % p.output_cols;                     \
        const auto current_output_row = (k / p.output_cols) % p.output_rows;   \
        const auto current_filter_col =                                        \
            (k / (2 * p.output_rows * p.output_cols)) % p.filter_cols;         \
        const auto current_filter_row =                                        \
            (k / (2 * p.output_rows * p.output_cols * p.filter_cols)) %        \
            p.filter_rows;                                                     \
        const auto current_offset_channel =                                    \
            (k / (p.output_rows * p.output_cols)) % offset_channels;           \
        const auto current_batch =                                             \
            k / (p.output_rows * p.output_cols * offset_channels);             \
                                                                               \
        const auto current_actual_batch = b * p.parallel_imgs + current_batch; \
                                                                               \
        const auto current_offset_group =                                      \
            current_offset_channel / (2 * offset_channel_step);                \
                                                                               \
        const auto channels_per_offset_group =                                 \
            p.input_channels / p.offset_groups;                                \
        const auto offset_channel_diff =                                       \
            current_offset_channel -                                           \
            current_offset_group * 2 * offset_channel_step;                    \
        const auto is_y_direction = offset_channel_diff % 2 == 0;              \
                                                                               \
        for (auto selected_offset_channel = (offset_channel_diff / 2);         \
             selected_offset_channel <                                         \
             channels_per_offset_group * offset_channel_step;                  \
             selected_offset_channel += offset_channel_step) {                 \
          const auto selected_filter_col =                                     \
              selected_offset_channel % p.filter_cols;                         \
          const auto selected_filter_row =                                     \
              (selected_offset_channel / p.filter_cols) % p.filter_rows;       \
          const auto input_channel_diff =                                      \
              (selected_offset_channel / (p.filter_cols * p.filter_rows));     \
                                                                               \
          const auto offset_h = offset_eigen_tensor(                           \
              b, current_batch, current_offset_group, selected_filter_row,     \
              selected_filter_col, 0, current_output_row, current_output_col); \
          const auto offset_w = offset_eigen_tensor(                           \
              b, current_batch, current_offset_group, selected_filter_row,     \
              selected_filter_col, 1, current_output_row, current_output_col); \
          const auto mask =                                                    \
              p.use_mask ? mask_eigen_tensor(                                  \
                               b, current_batch, current_offset_group,         \
                               selected_filter_row, selected_filter_col,       \
                               current_output_row, current_output_col)         \
                         : T(1);                                               \
                                                                               \
          const auto y =                                                       \
              (current_output_row * p.stride_rows - p.padding_rows) +          \
              selected_filter_row * p.dilation_rows + offset_h;                \
          const auto x =                                                       \
              (current_output_col * p.stride_cols - p.padding_cols) +          \
              selected_filter_col * p.dilation_cols + offset_w;                \
                                                                               \
          const auto selected_input_channel =                                  \
              input_channel_diff +                                             \
              current_offset_group * channels_per_offset_group;                \
                                                                               \
          const auto filter_data = column_buffer_eigen_tensor(                 \
              selected_input_channel, selected_filter_row,                     \
              selected_filter_col, current_batch, current_output_row,          \
              current_output_col);                                             \
                                                                               \
          const auto weight = GetCoordinateWeight<T>(                          \
              input_eigen_tensor, b, current_actual_batch,                     \
              selected_input_channel, y, x, is_y_direction);                   \
                                                                               \
          offset_grad_value += mask * weight * filter_data;                    \
                                                                               \
          if (is_y_direction) {                                                \
            mask_grad_value +=                                                 \
                filter_data * BilinearInterpolate<T>(                          \
                                  input_eigen_tensor, b, current_actual_batch, \
                                  selected_input_channel, y, x);               \
          }                                                                    \
        }                                                                      \
                                                                               \
        offset_grad_eigen_tensor(current_actual_batch, current_offset_channel, \
                                 current_output_row, current_output_col) =     \
            offset_grad_value;                                                 \
                                                                               \
        if (p.use_mask && is_y_direction) {                                    \
          const auto current_mask_channel =                                    \
              (current_offset_group * p.filter_rows + current_filter_row) *    \
                  p.filter_cols +                                              \
              current_filter_col;                                              \
                                                                               \
          mask_grad_eigen_tensor(current_actual_batch, current_mask_channel,   \
                                 current_output_row, current_output_col) =     \
              mask_grad_value;                                                 \
        }                                                                      \
      }                                                                        \
    };                                                                         \
    auto thread_pool =                                                         \
        context->device()->tensorflow_cpu_worker_threads()->workers;           \
    thread_pool->ParallelFor(num_kernels, cost, work);                         \
  }
TF_CALL_float(COL2IM_OFFSET_AND_MASK);
TF_CALL_double(COL2IM_OFFSET_AND_MASK);
#undef COL2IM_OFFSET_AND_MASK

#define COL2IM_INPUT(T)                                                        \
  template <>                                                                  \
  void DeformableConv2DGradFunctor<CPUDevice, T>::DeformableCol2ImForInput(    \
      OpKernelContext *context, int32 b) {                                     \
    const auto num_kernels = p.input_channels * p.filter_rows *                \
                             p.filter_cols * p.output_rows * p.output_cols *   \
                             p.parallel_imgs;                                  \
                                                                               \
    const auto offset_eigen_tensor = offset_tensor.template tensor<T, 8>();    \
                                                                               \
    const auto mask_eigen_tensor =                                             \
        p.use_mask ? mask_tensor.template tensor<T, 7>()                       \
                   : mask_tensor.template shaped<T, 7>({0, 0, 0, 0, 0, 0, 0}); \
                                                                               \
    const auto column_buffer_tensor_flattened =                                \
        column_buffer_tensor.template shaped<T, 1>({num_kernels});             \
                                                                               \
    auto input_grad_eigen_tensor = input_grad_tensor.tensor<T, 4>();           \
                                                                               \
    const auto cost = 3 * 3;                                                   \
    std::array<std::mutex, 100> mutex_array;                                   \
                                                                               \
    const auto work = [&](Eigen::Index start, Eigen::Index end) -> void {      \
      for (Eigen::Index k = start; k < end; ++k) {                             \
        const auto current_output_col = k % p.output_cols;                     \
        const auto current_output_row = (k / p.output_cols) % p.output_rows;   \
        const auto current_batch =                                             \
            (k / (p.output_rows * p.output_cols)) % p.parallel_imgs;           \
                                                                               \
        const auto current_filter_col =                                        \
            (k / (p.output_rows * p.output_cols * p.parallel_imgs)) %          \
            p.filter_cols;                                                     \
        const auto current_filter_row =                                        \
            (k / (p.output_rows * p.output_cols * p.parallel_imgs *            \
                  p.filter_cols)) %                                            \
            p.filter_rows;                                                     \
        const auto current_channel =                                           \
            k / (p.output_rows * p.output_cols * p.parallel_imgs *             \
                 p.filter_rows * p.filter_cols);                               \
                                                                               \
        const auto current_offset_group =                                      \
            current_channel / (p.input_channels / p.offset_groups);            \
                                                                               \
        const auto mask =                                                      \
            p.use_mask                                                         \
                ? mask_eigen_tensor(b, current_batch, current_offset_group,    \
                                    current_filter_row, current_filter_col,    \
                                    current_output_row, current_output_col)    \
                : T(1);                                                        \
                                                                               \
        const auto offset_h = offset_eigen_tensor(                             \
            b, current_batch, current_offset_group, current_filter_row,        \
            current_filter_col, 0, current_output_row, current_output_col);    \
        const auto offset_w = offset_eigen_tensor(                             \
            b, current_batch, current_offset_group, current_filter_row,        \
            current_filter_col, 1, current_output_row, current_output_col);    \
                                                                               \
        const auto y = (current_output_row * p.stride_rows - p.padding_rows) + \
                       current_filter_row * p.dilation_rows + offset_h;        \
        const auto x = (current_output_col * p.stride_cols - p.padding_cols) + \
                       current_filter_col * p.dilation_cols + offset_w;        \
                                                                               \
        for (auto dy = -1; dy <= 1; dy++) {                                    \
          for (auto dx = -1; dx <= 1; dx++) {                                  \
            const auto current_input_row = int(y) + dy;                        \
            const auto current_input_col = int(x) + dx;                        \
                                                                               \
            if (p.input_rows > current_input_row && current_input_row >= 0 &&  \
                p.input_cols > current_input_col && current_input_col >= 0 &&  \
                std::abs(y - current_input_row) < 1 &&                         \
                std::abs(x - current_input_col) < 1) {                         \
              const auto weight = (1.0 - std::abs(y - current_input_row)) *    \
                                  (1.0 - std::abs(x - current_input_col));     \
                                                                               \
              const auto current_actual_batch =                                \
                  b * p.parallel_imgs + current_batch;                         \
                                                                               \
              std::lock_guard<std::mutex> lock(                                \
                  mutex_array[(current_input_row * p.input_cols +              \
                               current_input_col) %                            \
                              100]);                                           \
              input_grad_eigen_tensor(current_actual_batch, current_channel,   \
                                      current_input_row, current_input_col) += \
                  mask * weight * column_buffer_tensor_flattened(k);           \
            }                                                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    };                                                                         \
    auto thread_pool =                                                         \
        context->device()->tensorflow_cpu_worker_threads()->workers;           \
    thread_pool->ParallelFor(num_kernels, cost, work);                         \
  }
TF_CALL_float(COL2IM_INPUT);
TF_CALL_double(COL2IM_INPUT);
#undef COL2IM_INPUT

}  // end namespace functor

template <typename Device, typename T>
class DeformableConv2DOpBase : public OpKernel {
 public:
  explicit DeformableConv2DOpBase(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides));
    OP_REQUIRES_OK(context, context->GetAttr("weight_groups", &weight_groups));
    OP_REQUIRES_OK(context, context->GetAttr("offset_groups", &offset_groups));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations));
    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    FormatFromString(data_format_str, &data_format);

    p = DeformableConv2DParams{};
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input_tensor = context->input(0);
    const Tensor &filter_tensor = context->input(1);

    const Tensor &bias_tensor = context->input(2);
    const Tensor &mask_tensor = context->input(4);

    const TensorShape &input_shape = input_tensor.shape();
    const TensorShape &filter_shape = filter_tensor.shape();

    const auto input_batches = input_shape.dim_size(0);
    const auto input_channels = input_shape.dim_size(1);
    const auto input_rows = input_shape.dim_size(2);
    const auto input_cols = input_shape.dim_size(3);

    const auto output_channels = filter_shape.dim_size(0);
    const auto filter_channels = filter_shape.dim_size(1);
    const auto filter_rows = filter_shape.dim_size(2);
    const auto filter_cols = filter_shape.dim_size(3);

    const auto dilation_rows = dilations[0];
    const auto dilation_cols = dilations[1];

    const auto stride_rows = strides[0];
    const auto stride_cols = strides[1];

    const auto parallel_imgs = GetParallelImgs(input_batches);

    int64 output_rows, output_cols;
    int64 padding_rows, padding_cols;
    OP_REQUIRES_OK(
        context, GetWindowedOutputSizeV2(input_rows, filter_rows, dilation_rows,
                                         stride_rows, padding, &output_rows,
                                         &padding_rows));
    OP_REQUIRES_OK(
        context, GetWindowedOutputSizeV2(input_cols, filter_cols, dilation_cols,
                                         stride_cols, padding, &output_cols,
                                         &padding_cols));

    p.input_batches = input_batches;
    p.input_channels = input_channels;
    p.input_rows = input_rows;
    p.input_cols = input_cols;
    p.filter_channels = filter_channels;
    p.filter_rows = filter_rows;
    p.filter_cols = filter_cols;
    p.padding_rows = padding_rows;
    p.padding_cols = padding_cols;
    p.stride_rows = stride_rows;
    p.stride_cols = stride_cols;
    p.dilation_rows = dilation_rows;
    p.dilation_cols = dilation_cols;
    p.output_channels = output_channels;
    p.output_rows = output_rows;
    p.output_cols = output_cols;
    p.parallel_imgs = parallel_imgs;
    p.weight_groups = weight_groups;
    p.offset_groups = offset_groups;
    p.batches = p.input_batches / p.parallel_imgs;
    p.use_mask = mask_tensor.NumElements() > 0;
    p.use_bias = bias_tensor.NumElements() > 0;
  }

  int GetParallelImgs(int n) {
    for (auto k = kMaxParallelImgs; k > 1; --k) {
      if (n % k == 0) {
        return k;
      }
    }
    return 1;
  }

 protected:
  TensorFormat data_format;
  DeformableConv2DParams p;

 private:
  std::vector<int32> strides;
  int32 weight_groups;
  int32 offset_groups;
  Padding padding;
  std::vector<int32> dilations;
};

template <typename Device, typename T>
class DeformableConv2DForwardOp : public DeformableConv2DOpBase<Device, T> {
  using DeformableConv2DOpBase<Device, T>::data_format;
  using DeformableConv2DOpBase<Device, T>::p;

 public:
  explicit DeformableConv2DForwardOp(OpKernelConstruction *context)
      : DeformableConv2DOpBase<Device, T>(context) {}

  void Compute(OpKernelContext *context) override {
    DeformableConv2DOpBase<Device, T>::Compute(context);

    const Tensor &input_tensor = context->input(0);
    const Tensor &filter_tensor = context->input(1);
    const Tensor &bias_tensor = context->input(2);
    const Tensor &offset_tensor = context->input(3);
    const Tensor &mask_tensor = context->input(4);

    TensorShape column_buffer_shape(
        {p.input_channels * p.filter_rows * p.filter_cols, p.parallel_imgs,
         p.output_rows, p.output_cols});
    Tensor column_buffer_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   column_buffer_shape,
                                                   &column_buffer_tensor));

    TensorShape output_shape =
        ShapeFromFormat(data_format, p.input_batches, p.output_rows,
                        p.output_cols, p.output_channels);
    Tensor *output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));

    functor::DeformableConv2DForwardFunctor<Device, T> deformableConv2DFunc(
        &input_tensor, &filter_tensor, &bias_tensor, &offset_tensor,
        &mask_tensor, &column_buffer_tensor, output_tensor, &p);
    Status s = deformableConv2DFunc(context);

    OP_REQUIRES_OK(context, s);
  }
};

template <typename Device, typename T>
class DeformableConv2DGradOp : public DeformableConv2DOpBase<Device, T> {
  using DeformableConv2DOpBase<Device, T>::data_format;
  using DeformableConv2DOpBase<Device, T>::p;

 public:
  explicit DeformableConv2DGradOp(OpKernelConstruction *context)
      : DeformableConv2DOpBase<Device, T>(context) {}

  void Compute(OpKernelContext *context) override {
    DeformableConv2DOpBase<Device, T>::Compute(context);

    const Tensor &input_tensor = context->input(0);
    const Tensor &filter_tensor = context->input(1);
    const Tensor &bias_tensor = context->input(2);
    const Tensor &offset_tensor = context->input(3);
    const Tensor &mask_tensor = context->input(4);
    const Tensor &output_grad_tensor = context->input(5);

    const TensorShape &input_shape = input_tensor.shape();
    const TensorShape &filter_shape = filter_tensor.shape();
    const TensorShape &bias_shape = bias_tensor.shape();
    const TensorShape &offset_shape = offset_tensor.shape();
    const TensorShape &mask_shape = mask_tensor.shape();

    TensorShape column_buffer_shape(
        {p.input_channels * p.filter_rows * p.filter_cols, p.parallel_imgs,
         p.output_rows, p.output_cols});
    Tensor column_buffer_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   column_buffer_shape,
                                                   &column_buffer_tensor));

    Tensor output_grad_tensor_reshaped;
    CHECK(output_grad_tensor_reshaped.CopyFrom(
        output_grad_tensor,
        TensorShape({p.batches, p.parallel_imgs, p.output_channels,
                     p.output_rows, p.output_cols})));

    TensorShape output_grad_tensor_transposed_shape(
        {p.batches, p.output_channels, p.parallel_imgs, p.output_rows,
         p.output_cols});
    Tensor output_grad_tensor_transposed;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          output_grad_tensor_transposed_shape,
                                          &output_grad_tensor_transposed));
    OP_REQUIRES_OK(context,
                   Transpose<Device, T, 5>(context, output_grad_tensor_reshaped,
                                           {0, 2, 1, 3, 4},
                                           &output_grad_tensor_transposed));

    TensorShape output_shape =
        ShapeFromFormat(data_format, p.input_batches, p.output_rows,
                        p.output_cols, p.output_channels);

    Tensor *input_grad_tensor = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, input_shape, &input_grad_tensor));
    Tensor *filter_grad_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, filter_shape,
                                                     &filter_grad_tensor));
    Tensor *bias_grad_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, bias_shape, &bias_grad_tensor));
    Tensor *offset_grad_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(3, offset_shape,
                                                     &offset_grad_tensor));
    Tensor *mask_grad_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(4, mask_shape, &mask_grad_tensor));

    functor::DeformableConv2DGradFunctor<Device, T> deformableConv2DGradFunc(
        &input_tensor, &filter_tensor, &bias_tensor, &offset_tensor,
        &mask_tensor, &output_grad_tensor_transposed, input_grad_tensor,
        filter_grad_tensor, bias_grad_tensor, offset_grad_tensor,
        mask_grad_tensor, &column_buffer_tensor, &p);
    Status s = deformableConv2DGradFunc(context);

    OP_REQUIRES_OK(context, s);
  }
};

// Register the CPU kernels.
#define REGISTER_DEFORMABLECONV2D_OP_CPU(T)                        \
  REGISTER_KERNEL_BUILDER(Name("Addons>DeformableConv2D")          \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<T>("T"),             \
                          DeformableConv2DForwardOp<CPUDevice, T>) \
  REGISTER_KERNEL_BUILDER(Name("Addons>DeformableConv2DGrad")      \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<T>("T"),             \
                          DeformableConv2DGradOp<CPUDevice, T>)

TF_CALL_float(REGISTER_DEFORMABLECONV2D_OP_CPU);
TF_CALL_double(REGISTER_DEFORMABLECONV2D_OP_CPU);
#undef REGISTER_DEFORMABLECONV2D_OP_CPU

// Register the GPU kernels.
#if GOOGLE_CUDA

#define REGISTER_DEFORMABLECONV2D_OP_GPU(T)                        \
  REGISTER_KERNEL_BUILDER(Name("Addons>DeformableConv2D")          \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T"),             \
                          DeformableConv2DForwardOp<GPUDevice, T>) \
  REGISTER_KERNEL_BUILDER(Name("Addons>DeformableConv2DGrad")      \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T"),             \
                          DeformableConv2DGradOp<GPUDevice, T>)

TF_CALL_float(REGISTER_DEFORMABLECONV2D_OP_GPU);
TF_CALL_double(REGISTER_DEFORMABLECONV2D_OP_GPU);
#undef REGISTER_DEFORMABLECONV2D_OP_GPU

#endif  // GOOGLE_CUDA

}  // namespace addons
}  // namespace tensorflow
