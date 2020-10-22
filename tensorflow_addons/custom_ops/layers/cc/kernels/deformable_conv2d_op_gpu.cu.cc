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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow_addons/custom_ops/layers/cc/kernels/deformable_conv2d_op.h"

namespace tensorflow {
namespace addons {

using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
__global__ void DeformableIm2ColKernel(
    int32 b, int32 num_kernels, DeformableConv2DParams p,
    typename TTypes<T, 5>::Tensor input_eigen_tensor,
    typename TTypes<T, 8>::Tensor offset_eigen_tensor,
    typename TTypes<T, 7>::Tensor mask_eigen_tensor,
    typename TTypes<T, 4>::Tensor column_buffer_eigen_tensor) {
  CUDA_1D_KERNEL_LOOP(k, num_kernels) {
    const auto current_output_col = k % p.output_cols;
    const auto current_output_row = (k / p.output_cols) % p.output_rows;
    const auto current_batch =
        (k / (p.output_rows * p.output_cols)) % p.parallel_imgs;
    const auto current_input_channel =
        k / (p.output_rows * p.output_cols * p.parallel_imgs);
    const auto current_output_channel =
        current_input_channel * p.filter_rows * p.filter_cols;

    const auto current_actual_batch = b * p.parallel_imgs + current_batch;

    const auto group_index =
        current_input_channel / (p.input_channels / p.offset_groups);

    auto column_buffer_tensor_channel = current_output_channel;
    for (auto current_filter_row = 0; current_filter_row < p.filter_rows;
         current_filter_row++) {
      for (auto current_filter_col = 0; current_filter_col < p.filter_cols;
           current_filter_col++) {
        auto offset_h = offset_eigen_tensor(
            b, current_batch, group_index, current_filter_row,
            current_filter_col, 0, current_output_row, current_output_col);
        auto offset_w = offset_eigen_tensor(
            b, current_batch, group_index, current_filter_row,
            current_filter_col, 1, current_output_row, current_output_col);

        auto mask = p.use_mask ? mask_eigen_tensor(
                                     b, current_batch, group_index,
                                     current_filter_row, current_filter_col,
                                     current_output_row, current_output_col)
                               : T(1);

        auto y = (current_output_row * p.stride_rows - p.padding_rows) +
                 current_filter_row * p.dilation_rows + offset_h;
        auto x = (current_output_col * p.stride_cols - p.padding_cols) +
                 current_filter_col * p.dilation_cols + offset_w;

        column_buffer_eigen_tensor(column_buffer_tensor_channel, current_batch,
                                   current_output_row, current_output_col) =
            mask * BilinearInterpolate<T>(input_eigen_tensor, b,
                                          current_actual_batch,
                                          current_input_channel, y, x);
        column_buffer_tensor_channel++;
      }
    }
  }
}

template <typename T>
__global__ void DeformableCol2ImForOffsetAndMaskKernel(
    int32 b, int32 num_kernels, DeformableConv2DParams p,
    typename TTypes<T, 5>::Tensor input_eigen_tensor,
    typename TTypes<T, 8>::Tensor offset_eigen_tensor,
    typename TTypes<T, 7>::Tensor mask_eigen_tensor,
    typename TTypes<T, 4>::Tensor offset_grad_eigen_tensor,
    typename TTypes<T, 4>::Tensor mask_grad_eigen_tensor,
    typename TTypes<T, 6>::Tensor column_buffer_eigen_tensor) {
  CUDA_1D_KERNEL_LOOP(k, num_kernels) {
    auto offset_grad_value = T(0);
    auto mask_grad_value = T(0);

    const auto offset_channels =
        2 * p.filter_rows * p.filter_cols * p.offset_groups;

    const auto offset_channel_step = p.filter_rows * p.filter_cols;

    const auto current_output_col = k % p.output_cols;
    const auto current_output_row = (k / p.output_cols) % p.output_rows;
    const auto current_filter_col =
        (k / (2 * p.output_rows * p.output_cols)) % p.filter_cols;
    const auto current_filter_row =
        (k / (2 * p.output_rows * p.output_cols * p.filter_cols)) %
        p.filter_rows;
    const auto current_offset_channel =
        (k / (p.output_rows * p.output_cols)) % offset_channels;
    const auto current_batch =
        k / (p.output_rows * p.output_cols * offset_channels);

    const auto current_actual_batch = b * p.parallel_imgs + current_batch;

    const auto current_offset_group =
        current_offset_channel / (2 * offset_channel_step);

    const auto channels_per_offset_group = p.input_channels / p.offset_groups;
    const auto offset_channel_diff =
        current_offset_channel - current_offset_group * 2 * offset_channel_step;
    const auto is_y_direction = offset_channel_diff % 2 == 0;

    for (auto selected_offset_channel = (offset_channel_diff / 2);
         selected_offset_channel <
         channels_per_offset_group * offset_channel_step;
         selected_offset_channel += offset_channel_step) {
      const auto selected_filter_col = selected_offset_channel % p.filter_cols;
      const auto selected_filter_row =
          (selected_offset_channel / p.filter_cols) % p.filter_rows;
      const auto input_channel_diff =
          (selected_offset_channel / (p.filter_cols * p.filter_rows));

      const auto offset_h = offset_eigen_tensor(
          b, current_batch, current_offset_group, selected_filter_row,
          selected_filter_col, 0, current_output_row, current_output_col);
      const auto offset_w = offset_eigen_tensor(
          b, current_batch, current_offset_group, selected_filter_row,
          selected_filter_col, 1, current_output_row, current_output_col);
      const auto mask =
          p.use_mask
              ? mask_eigen_tensor(b, current_batch, current_offset_group,
                                  selected_filter_row, selected_filter_col,
                                  current_output_row, current_output_col)
              : T(1);

      const auto y = (current_output_row * p.stride_rows - p.padding_rows) +
                     selected_filter_row * p.dilation_rows + offset_h;
      const auto x = (current_output_col * p.stride_cols - p.padding_cols) +
                     selected_filter_col * p.dilation_cols + offset_w;

      const auto selected_input_channel =
          input_channel_diff + current_offset_group * channels_per_offset_group;

      const auto filter_data = column_buffer_eigen_tensor(
          selected_input_channel, selected_filter_row, selected_filter_col,
          current_batch, current_output_row, current_output_col);

      const auto weight =
          GetCoordinateWeight<T>(input_eigen_tensor, b, current_actual_batch,
                                 selected_input_channel, y, x, is_y_direction);

      offset_grad_value += mask * weight * filter_data;

      if (is_y_direction) {
        mask_grad_value +=
            filter_data * BilinearInterpolate<T>(input_eigen_tensor, b,
                                                 current_actual_batch,
                                                 selected_input_channel, y, x);
      }
    }

    offset_grad_eigen_tensor(current_actual_batch, current_offset_channel,
                             current_output_row, current_output_col) =
        offset_grad_value;

    if (p.use_mask && is_y_direction) {
      const auto current_mask_channel =
          (current_offset_group * p.filter_rows + current_filter_row) *
              p.filter_cols +
          current_filter_col;

      mask_grad_eigen_tensor(current_actual_batch, current_mask_channel,
                             current_output_row, current_output_col) =
          mask_grad_value;
    }
  }
}

template <typename T>
__global__ void DeformableCol2ImForInputKernel(
    int32 b, int32 num_kernels, DeformableConv2DParams p,
    typename TTypes<T, 8>::Tensor offset_eigen_tensor,
    typename TTypes<T, 7>::Tensor mask_eigen_tensor,
    typename TTypes<T, 4>::Tensor input_grad_eigen_tensor,
    typename TTypes<T, 1>::Tensor column_buffer_tensor_flattened) {
  CUDA_1D_KERNEL_LOOP(k, num_kernels) {
    const auto current_output_col = k % p.output_cols;
    const auto current_output_row = (k / p.output_cols) % p.output_rows;
    const auto current_batch =
        (k / (p.output_rows * p.output_cols)) % p.parallel_imgs;

    const auto current_filter_col =
        (k / (p.output_rows * p.output_cols * p.parallel_imgs)) % p.filter_cols;
    const auto current_filter_row = (k / (p.output_rows * p.output_cols *
                                          p.parallel_imgs * p.filter_cols)) %
                                    p.filter_rows;
    const auto current_channel =
        k / (p.output_rows * p.output_cols * p.parallel_imgs * p.filter_rows *
             p.filter_cols);

    const auto current_offset_group =
        current_channel / (p.input_channels / p.offset_groups);

    const auto mask =
        p.use_mask ? mask_eigen_tensor(b, current_batch, current_offset_group,
                                       current_filter_row, current_filter_col,
                                       current_output_row, current_output_col)
                   : T(1);

    const auto offset_h = offset_eigen_tensor(
        b, current_batch, current_offset_group, current_filter_row,
        current_filter_col, 0, current_output_row, current_output_col);
    const auto offset_w = offset_eigen_tensor(
        b, current_batch, current_offset_group, current_filter_row,
        current_filter_col, 1, current_output_row, current_output_col);

    const auto y = (current_output_row * p.stride_rows - p.padding_rows) +
                   current_filter_row * p.dilation_rows + offset_h;
    const auto x = (current_output_col * p.stride_cols - p.padding_cols) +
                   current_filter_col * p.dilation_cols + offset_w;

    for (auto dy = -1; dy <= 1; dy++) {
      for (auto dx = -1; dx <= 1; dx++) {
        const auto current_input_row = int(y) + dy;
        const auto current_input_col = int(x) + dx;

        if (p.input_rows > current_input_row && current_input_row >= 0 &&
            p.input_cols > current_input_col && current_input_col >= 0 &&
            std::abs(y - current_input_row) < 1 &&
            std::abs(x - current_input_col) < 1) {
          const auto weight = (1.0 - std::abs(y - current_input_row)) *
                              (1.0 - std::abs(x - current_input_col));

          const auto current_actual_batch = b * p.parallel_imgs + current_batch;

          auto *ptr = input_grad_eigen_tensor.data();

          const auto ptr_pos =
              ((current_actual_batch * p.input_channels + current_channel) *
                   p.input_rows +
               current_input_row) *
                  p.input_cols +
              current_input_col;

          GpuAtomicAdd(ptr + ptr_pos,
                       mask * weight * column_buffer_tensor_flattened(k));
        }
      }
    }
  }
}

#define IM2COL(T)                                                              \
  template <>                                                                  \
  void DeformableConv2DFunctorBase<GPUDevice, T>::DeformableIm2Col(            \
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
    auto device = context->template eigen_device<GPUDevice>();                 \
    GpuLaunchConfig config = GetGpuLaunchConfig(num_kernels, device);          \
    TF_CHECK_OK(GpuLaunchKernel(DeformableIm2ColKernel<T>, config.block_count, \
                                config.thread_per_block, 0, device.stream(),   \
                                b, num_kernels, p, input_eigen_tensor,         \
                                offset_eigen_tensor, mask_eigen_tensor,        \
                                column_buffer_eigen_tensor));                  \
  }
TF_CALL_float(IM2COL);
TF_CALL_double(IM2COL);
#undef IM2COL

#define COL2IM_OFFSET_AND_MASK(T)                                              \
  template <>                                                                  \
  void                                                                         \
  DeformableConv2DGradFunctor<GPUDevice, T>::DeformableCol2ImForOffsetAndMask( \
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
    auto device = context->template eigen_device<GPUDevice>();                 \
    GpuLaunchConfig config = GetGpuLaunchConfig(num_kernels, device);          \
    TF_CHECK_OK(GpuLaunchKernel(                                               \
        DeformableCol2ImForOffsetAndMaskKernel<T>, config.block_count,         \
        config.thread_per_block, 0, device.stream(), b, num_kernels, p,        \
        input_eigen_tensor, offset_eigen_tensor, mask_eigen_tensor,            \
        offset_grad_eigen_tensor, mask_grad_eigen_tensor,                      \
        column_buffer_eigen_tensor));                                          \
  }
TF_CALL_float(COL2IM_OFFSET_AND_MASK);
TF_CALL_double(COL2IM_OFFSET_AND_MASK);
#undef COL2IM_OFFSET_AND_MASK

#define COL2IM_INPUT(T)                                                        \
  template <>                                                                  \
  void DeformableConv2DGradFunctor<GPUDevice, T>::DeformableCol2ImForInput(    \
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
    auto device = context->template eigen_device<GPUDevice>();                 \
    GpuLaunchConfig config = GetGpuLaunchConfig(num_kernels, device);          \
    TF_CHECK_OK(GpuLaunchKernel(                                               \
        DeformableCol2ImForInputKernel<T>, config.block_count,                 \
        config.thread_per_block, 0, device.stream(), b, num_kernels, p,        \
        offset_eigen_tensor, mask_eigen_tensor, input_grad_eigen_tensor,       \
        column_buffer_tensor_flattened));                                      \
  }
TF_CALL_float(COL2IM_INPUT);
TF_CALL_double(COL2IM_INPUT);
#undef COL2IM_INPUT

#define EXPLICIT_TEMPLATE(T)                                    \
  template struct DeformableConv2DForwardFunctor<GPUDevice, T>; \
  template struct DeformableConv2DGradFunctor<GPUDevice, T>;
TF_CALL_float(EXPLICIT_TEMPLATE);
TF_CALL_double(EXPLICIT_TEMPLATE);
#undef EXPLICIT_TEMPLATE

}  // end namespace functor

#define EXPLICIT_TEMPLATE(T)                   \
  template Status Transpose<GPUDevice, T, 5>(  \
      OpKernelContext * ctx, const Tensor &in, \
      const gtl::ArraySlice<int32> perm, Tensor *out);
TF_CALL_float(EXPLICIT_TEMPLATE);
TF_CALL_double(EXPLICIT_TEMPLATE);
#undef EXPLICIT_TEMPLATE

}  // namespace addons
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
