
#include <algorithm>
#include <cstdlib>

#include "tensorflow_addons/custom_ops/layers/cc/kernels/deformable_conv_op.h"

#ifdef GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace addons {

namespace functor {

typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename DType>
__device__ DType DmcnIm2colBilinear(const DType *bottom_data,
                                    const int data_width, const int height,
                                    const int width, DType h, DType w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  DType lh = h - h_low;
  DType lw = w - w_low;
  DType hh = 1 - lh, hw = 1 - lw;

  DType v1 = 0;
  if (h_low >= 0 && w_low >= 0) v1 = bottom_data[h_low * data_width + w_low];
  DType v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  DType v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  DType v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  DType w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  DType val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}
template <typename DType>
__device__ DType DmcnGetGradientWeight(DType argmax_h, DType argmax_w,
                                       const int h, const int w,
                                       const int height, const int width) {
  /*
   * offset h, offset w, (h, w) coordinate
   */
  if (argmax_h <= -1 || argmax_w <= -1 || argmax_h >= height ||
      argmax_w >= width) {
    return 0;
  }
  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;
  DType weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
  return weight;
}
template <typename DType>
__device__ DType DmcnGetCoordinateWeight(DType argmax_h, DType argmax_w,
                                         const int height, const int width,
                                         const DType *im_data,
                                         const int data_width,
                                         const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    // empty
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  DType weight = 0;

  if (bp_dir == 0) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += -1 * (argmax_w - argmax_w_low) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += (argmax_w_low + 1 - argmax_w) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_w - argmax_w_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  } else if (bp_dir == 1) {
    if (argmax_h_low >= 0 && argmax_w_low >= 0)
      weight += -1 * (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_low];
    if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
      weight += (argmax_h_low + 1 - argmax_h) *
                im_data[argmax_h_low * data_width + argmax_w_high];
    if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
      weight += -1 * (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_low];
    if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
      weight += (argmax_h - argmax_h_low) *
                im_data[argmax_h_high * data_width + argmax_w_high];
  }

  return weight;
}

template <typename DType>
__global__ void SwapAxisKernel(const int n, const int cuda_mem_size,
                               const int min_unit_size, DType *input_data,
                               const int dim_num, const int axis_x_dims,
                               const int axis_y_dims, const int axis_x,
                               const int axis_y) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    DType *device_data = new DType[cuda_mem_size];
    DType *input_data_ptr = input_data + index * cuda_mem_size;
    for (int j = 0; j < axis_y_dims; j++) {
      for (int i = 0; i < axis_x_dims; i++) {
        DType *temp_ptr =
            input_data_ptr + (i * axis_x_dims + j) * min_unit_size;
        DType *device_data_temp_ptr =
            device_data + (j * axis_y_dims + i) * min_unit_size;
        for (int k = 0; k < min_unit_size; k++) {
          *(device_data_temp_ptr + k) = *(temp_ptr + k);
        }
      }
    }
    for (int i = 0; i < cuda_mem_size; i++) {
      *(input_data_ptr + i) = *(device_data + i);
    }
    delete[] device_data;
  }
}
template <typename DType>
__global__ void DeformableConv2DIm2ColKernel(
    const int n, const DType *data_im, const DType *data_offset,
    const DType *data_mask, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group,
    const int batch_size, const int num_channels, const int deformable_group,
    const int height_col, const int width_col, DType *data_col) {
  /*
   * channel_per_deformable_group // 输入图通道数除以deformable_group的数量,
   * //这里的batch_size代表的是im2col_step_, 一般就设为1了
   */
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;
    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;
    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    DType *data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const DType *data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const DType *data_offset_ptr =
        data_offset + (b_col * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;
    const DType *data_mask_ptr =
        data_mask + (b_col * deformable_group + deformable_group_index) *
                        kernel_h * kernel_w * height_col * width_col;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const int data_mask_hw_ptr =
            ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
        const DType offset_h = data_offset_ptr[data_offset_h_ptr];
        const DType offset_w = data_offset_ptr[data_offset_w_ptr];
        const DType mask = data_mask_ptr[data_mask_hw_ptr];
        DType val = static_cast<DType>(0);
        const DType h_im = h_in + i * dilation_h + offset_h;
        const DType w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          val =
              DmcnIm2colBilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}
template <typename T>
__global__ void DeformablePSROIPoolForwardKernel(
    const int count, const T *bottom_data, const T spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const T *bottom_rois,
    const T *bottom_trans, const int no_trans, const T trans_std,
    const int sample_per_part, const int output_dim, const int group_size,
    const int part_size, const int num_classes, const int channels_each_class,
    T *top_data, T *top_count) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;
    // [start, end) interval for spatial sampling
    const T *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    T roi_start_w = (T)(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    T roi_start_h = (T)(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
    T roi_end_w = (T)(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
    T roi_end_h = (T)(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;
    // Force too small ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, static_cast<T>(0.1));  // avoid 0
    T roi_height = max(roi_end_h - roi_start_h, static_cast<T>(0.1));
    // Compute w and h at bottom
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);
    T sub_bin_size_h = bin_size_h / static_cast<T>(sample_per_part);
    T sub_bin_size_w = bin_size_w / static_cast<T>(sample_per_part);
    int part_h = floor(static_cast<T>(ph) / pooled_height * part_size);
    int part_w = floor(static_cast<T>(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;
    T trans_x =
        no_trans
            ? (T)(0)
            : bottom_trans[(((n * num_classes + class_id) * 2) * part_size +
                            part_h) *
                               part_size +
                           part_w] *
                  (T)trans_std;
    T trans_y =
        no_trans
            ? (T)(0)
            : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size +
                            part_h) *
                               part_size +
                           part_w] *
                  (T)trans_std;
    T wstart = static_cast<T>(pw) * bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;
    T sum = 0;
    int total = 0;
    int gw = floor(static_cast<T>(pw) * group_size / pooled_width);
    int gh = floor(static_cast<T>(ph) * group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);
    const T *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels) * height * width;
    for (int ih = 0; ih < sample_per_part; ++ih) {
      for (int iw = 0; iw < sample_per_part; ++iw) {
        T w = wstart + iw * sub_bin_size_w;
        T h = hstart + ih * sub_bin_size_h;
        // bilinear interpolation
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5) {
          continue;
        }
        w = min(max(w, static_cast<T>(0.)), static_cast<T>(width - 1.));
        h = min(max(h, static_cast<T>(0.)), static_cast<T>(height - 1.));
        int c = (ctop * group_size + gh) * group_size + gw;
        T val = DmcnIm2colBilinear(offset_bottom_data + c * height * width, w,
                                   h, w, (T)height, (T)width);
        sum += val;
        total++;
      }
    }
    top_data[index] = total == 0 ? (T)(0) : sum / total;
    top_count[index] = total;
  }
}
template <typename T>
__global__ void DeformablePSROIPoolBackwardAccKernel(
    const int count, const T *top_diff, const T *top_count, const int num_rois,
    const T spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int output_dim, T *bottom_data_diff, T *bottom_trans_diff,
    const T *bottom_data, const T *bottom_rois, const T *bottom_trans,
    const int no_trans, const T trans_std, const int sample_per_part,
    const int group_size, const int part_size, const int num_classes,
    const int channels_each_class) {
  CUDA_1D_KERNEL_LOOP(index, count) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;
    // [start, end) interval for spatial sampling
    const T *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    T roi_start_w = (T)(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    T roi_start_h = (T)(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
    T roi_end_w = (T)(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
    T roi_end_h = (T)(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;
    // Force too small ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, static_cast<T>(0.1));  // avoid 0
    T roi_height = max(roi_end_h - roi_start_h, static_cast<T>(0.1));

    // Compute w and h at bottom
    T bin_size_h = roi_height / (T)(pooled_height);
    T bin_size_w = roi_width / (T)(pooled_width);

    T sub_bin_size_h = bin_size_h / (T)(sample_per_part);
    T sub_bin_size_w = bin_size_w / (T)(sample_per_part);

    int part_h = floor((T)(ph) / pooled_height * part_size);
    int part_w = floor((T)(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;
    T trans_x =
        no_trans
            ? (T)(0)
            : bottom_trans[(((n * num_classes + class_id) * 2) * part_size +
                            part_h) *
                               part_size +
                           part_w] *
                  (T)trans_std;
    T trans_y =
        no_trans
            ? (T)(0)
            : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size +
                            part_h) *
                               part_size +
                           part_w] *
                  (T)trans_std;

    T wstart = (T)(pw)*bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    T hstart = (T)(ph)*bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;

    if (top_count[index] <= 0) {
      continue;
    }
    T diff_val = top_diff[index] / top_count[index];
    const T *offset_bottom_data =
        bottom_data + roi_batch_ind * channels * height * width;
    T *offset_bottom_data_diff =
        bottom_data_diff + roi_batch_ind * channels * height * width;
    int gw = floor((T)(pw)*group_size / pooled_width);
    int gh = floor((T)(ph)*group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);
    for (int ih = 0; ih < sample_per_part; ih++) {
      for (int iw = 0; iw < sample_per_part; iw++) {
        T w = wstart + iw * sub_bin_size_w;
        T h = hstart + ih * sub_bin_size_h;
        // bilinear interpolation
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5) {
          continue;
        }
        w = min(max(w, 0.), width - 1.);
        h = min(max(h, 0.), height - 1.);
        int c = (ctop * group_size + gh) * group_size + gw;
        // backward on feature
        int x0 = floor(w);
        int x1 = ceil(w);
        int y0 = floor(h);
        int y1 = ceil(h);
        T dist_x = w - x0, dist_y = h - y0;
        T q00 = (1 - dist_x) * (1 - dist_y);
        T q01 = (1 - dist_x) * dist_y;
        T q10 = dist_x * (1 - dist_y);
        T q11 = dist_x * dist_y;
        int bottom_index_base = c * height * width;
        CudaAtomicAdd(
            offset_bottom_data_diff + bottom_index_base + y0 * width + x0,
            q00 * diff_val);
        CudaAtomicAdd(
            offset_bottom_data_diff + bottom_index_base + y1 * width + x0,
            q01 * diff_val);
        CudaAtomicAdd(
            offset_bottom_data_diff + bottom_index_base + y0 * width + x1,
            q10 * diff_val);
        CudaAtomicAdd(
            offset_bottom_data_diff + bottom_index_base + y1 * width + x1,
            q11 * diff_val);

        if (no_trans) {
          continue;
        }
        T U00 = offset_bottom_data[bottom_index_base + y0 * width + x0];
        T U01 = offset_bottom_data[bottom_index_base + y1 * width + x0];
        T U10 = offset_bottom_data[bottom_index_base + y0 * width + x1];
        T U11 = offset_bottom_data[bottom_index_base + y1 * width + x1];
        T diff_x = (U11 * dist_y + U10 * (1 - dist_y) - U01 * dist_y -
                    U00 * (1 - dist_y)) *
                   trans_std * diff_val;
        diff_x *= roi_width;
        T diff_y = (U11 * dist_x + U01 * (1 - dist_x) - U10 * dist_x -
                    U00 * (1 - dist_x)) *
                   trans_std * diff_val;
        diff_y *= roi_height;

        CudaAtomicAdd(
            bottom_trans_diff +
                (((n * num_classes + class_id) * 2) * part_size + part_h) *
                    part_size +
                part_w,
            diff_x);
        CudaAtomicAdd(
            bottom_trans_diff +
                (((n * num_classes + class_id) * 2 + 1) * part_size + part_h) *
                    part_size +
                part_w,
            diff_y);
      }
    }
  }
}
template <typename DType>
__global__ void DeformableConv2DCol2ImKernel(
    const int n, const DType *data_col, const DType *data_offset,
    const DType *data_mask, const int channels, const int height,
    const int width, const int kernel_h, const int kernel_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int deformable_group, const int height_col, const int width_col,
    DType *grad_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i =
        (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c =
        index / width_col / height_col / batch_size / kernel_w / kernel_h;
    // compute the start and end of the output
    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const DType *data_offset_ptr =
        data_offset + (b * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;
    const DType *data_mask_ptr =
        data_mask + (b * deformable_group + deformable_group_index) * kernel_h *
                        kernel_w * height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const int data_mask_hw_ptr =
        ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    const DType offset_h = data_offset_ptr[data_offset_h_ptr];
    const DType offset_w = data_offset_ptr[data_offset_w_ptr];
    const DType mask = data_mask_ptr[data_mask_hw_ptr];
    const DType cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const DType cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const DType cur_top_grad = data_col[index] * mask;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          DType weight =
              DmcnGetGradientWeight(cur_inv_h_data, cur_inv_w_data, cur_h + dy,
                                    cur_w + dx, height, width);
          CudaAtomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}
template <typename DType>
__global__ void DeformableConv2DCol2ImCoordGPUKernel(
    const int n, const DType *data_col, const DType *data_im,
    const DType *data_offset, const DType *data_mask, const int channels,
    const int height, const int width,  // 输入的C, H, W
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group,
    const int batch_size, const int offset_channels, const int deformable_group,
    const int height_col, const int width_col, DType *grad_offset,
    DType *grad_mask) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    DType val = 0, mval = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;
    // compute the start and end of the output

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const DType *data_col_ptr =
        data_col + deformable_group_index * channel_per_deformable_group *
                       batch_size * width_col * height_col;
    const DType *data_im_ptr =
        data_im + (b * deformable_group + deformable_group_index) *
                      channel_per_deformable_group / kernel_h / kernel_w *
                      height * width;
    const DType *data_offset_ptr =
        data_offset + (b * deformable_group + deformable_group_index) * 2 *
                          kernel_h * kernel_w * height_col * width_col;
    const DType *data_mask_ptr =
        data_mask + (b * deformable_group + deformable_group_index) * kernel_h *
                        kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group;
         col_c += col_step) {
      const int col_pos =
          (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i =
          (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
           w_out);
      const int data_mask_hw_ptr =
          (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const DType offset_h = data_offset_ptr[data_offset_h_ptr];
      const DType offset_w = data_offset_ptr[data_offset_w_ptr];
      const DType mask = data_mask_ptr[data_mask_hw_ptr];
      DType inv_h = h_in + i * dilation_h + offset_h;
      DType inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      } else {
        mval += data_col_ptr[col_pos] *
                DmcnIm2colBilinear(data_im_ptr + cnt * height * width, width,
                                   height, width, inv_h, inv_w);
      }
      const DType weight = DmcnGetCoordinateWeight(
          inv_h, inv_w, height, width, data_im_ptr + cnt * height * width,
          width, bp_dir);
      val += weight * data_col_ptr[col_pos] * mask;
      cnt += 1;
    }

    grad_offset[index] = val;
    // KERNEL_ASSIGN(grad_offset[index], offset_req, val);
    if (offset_c % 2 == 0) {
      grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h *
                      kernel_w +
                  offset_c / 2) *
                     height_col +
                 h) *
                    width_col +
                w] = mval;
      // KERNEL_ASSIGN(grad_mask[(((b * deformable_group +
      // deformable_group_index) * kernel_h * kernel_w + offset_c / 2) *
      // height_col + h) * width_col + w], mask_req, mval);
    }
  }
}
template <typename DType>
__global__ void PureAddToKernel(const int n, DType *result_data,
                                const DType *right_data) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    CudaAtomicAdd(result_data + index, right_data[index]);
  }
}
template <typename DType>
__global__ void SetZeroKernel(const int n, DType *result_data) {
  CUDA_1D_KERNEL_LOOP(index, n) { *(result_data + index) = DType(0); }
}
template <typename DType>
__global__ void SetOneKernel(const int n, DType *result_data) {
  CUDA_1D_KERNEL_LOOP(index, n) { *(result_data + index) = DType(1); }
}
template <typename DType>
__global__ void SetNumAtIndexKernel(DType num, int index, DType *data) {
  *(data + index) = num;
}
template <typename DType>
void DeformableConv2DCol2ImCoord<GPUDevice, DType>::operator()(
    const Eigen::GpuDevice &d, const DType *data_col, const DType *data_im,
    const DType *data_offset, const DType *data_mask, const TShape &im_shape,
    const TShape &col_shape, const TShape &kernel_shape, const TShape &pad,
    const TShape &stride, const TShape &dilation,
    const int32_t deformable_group, DType *grad_offset, DType *grad_mask) {
  int num_spatial_axes = kernel_shape.size();
  int num_kernels = col_shape[1] * col_shape[2] * col_shape[3] * 2 *
                    kernel_shape[0] * kernel_shape[1] * deformable_group;
  int channel_per_deformable_group = col_shape[0] / deformable_group;
  // num_axes should be smaller than block size
  CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels, d);
  CHECK_LT(num_spatial_axes, config.thread_per_block);
  switch (num_spatial_axes) {
    case 2:
      // To avoid involving atomic operations, we will launch one kernel per
      // bottom dimension, and then in the kernel add up the top dimensions.
      // NOLINT_NEXT_LINE(whitespace/operators)

      DeformableConv2DCol2ImCoordGPUKernel<DType>
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
              num_kernels, data_col, data_im, data_offset, data_mask,
              im_shape[1], im_shape[2], im_shape[3], kernel_shape[0],
              kernel_shape[1], pad[0], pad[1], stride[0], stride[1],
              dilation[0], dilation[1], channel_per_deformable_group,
              col_shape[1],
              2 * kernel_shape[0] * kernel_shape[1] * deformable_group,
              deformable_group, col_shape[2], col_shape[3], grad_offset,
              grad_mask);
      // MSHADOW_CUDA_POST_KERNEL_CHECK(DeformableConv2DCol2ImCoordGPUKernel);
      break;
    default:
      LOG(FATAL) << "col2im_nd_gpu does not support computation with "
                 << num_spatial_axes << " spatial axes";
  }
}
template <typename DType>
void DeformableConv2DCol2Im<GPUDevice, DType>::operator()(
    const GPUDevice &d, const DType *data_col, const DType *data_offset,
    const DType *data_mask, const TShape &im_shape, const TShape &col_shape,
    const TShape &kernel_shape, const TShape &pad, const TShape &stride,
    const TShape &dilation, const int32_t deformable_group, DType *grad_im) {
  int num_spatial_axes = kernel_shape.size();
  int im_size = ProdShape(im_shape, 1, im_shape.size());
  int channel_per_deformable_group = im_shape[1] / deformable_group;
  int num_kernels = ProdShape(col_shape, 0, col_shape.size());
  // num_axes should be smaller than block size
  CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels, d);
  CHECK_LT(num_spatial_axes, config.thread_per_block);
  //   using namespace mxnet_op;
  switch (num_spatial_axes) {
    case 2:
      // To avoid involving atomic operations, we will launch one kernel per
      // bottom dimension, and then in the kernel add up the top dimensions.
      // NOLINT_NEXT_LINE(whitespace/operators)
      DeformableConv2DCol2ImKernel<DType>
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
              num_kernels, data_col, data_offset, data_mask, im_shape[1],
              im_shape[2], im_shape[3], kernel_shape[0], kernel_shape[1],
              pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1],
              channel_per_deformable_group, col_shape[1], deformable_group,
              col_shape[2], col_shape[3], grad_im);
      // MSHADOW_CUDA_POST_KERNEL_CHECK(modulated_deformable_col2im_gpu_kernel);
      break;
    default:
      LOG(FATAL) << "col2im_nd_gpu does not support computation with "
                 << num_spatial_axes << " spatial axes";
  }
}

template <typename DType>
void DeformableConv2DIm2Col<GPUDevice, DType>::operator()(
    const GPUDevice &d, const DType *data_im, const DType *data_offset,
    const DType *data_mask, const TShape &im_shape, const TShape &col_shape,
    const TShape &kernel_shape, const TShape &pad, const TShape &stride,
    const TShape &dilation, const int32_t deformable_group, DType *data_col) {
  int num_spatial_axes = kernel_shape.size();
  int channel_per_deformable_group =
      im_shape[1] / deformable_group;  // imshape[1] = 输入图的通道数
  int num_kernels =
      im_shape[1] *
      ProdShape(col_shape, 1,
                col_shape.size());  // K * N / k.Size(), k = filter, col_shape =
                                    // [K, im2col_step_, H, W]
  CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels, d);
  CHECK_LT(num_spatial_axes, config.thread_per_block);
  switch (num_spatial_axes) {
    case 2:
      DeformableConv2DIm2ColKernel<
          DType>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<config.block_count,
             config
                 .thread_per_block,  // 注意这里申请的block的个数是num_kernel个,
             0, d.stream()>>>(
              // CUDA对device(GPU
              // )的内存管理主要通过cudaMalloc()、cudaFree()、cudaMemcpy()
              // 进行管理。另外，从上述代码我们可以看到， add()
              // 函数的调用比较奇怪相对于C语言来说，需要用add<<<M，N>>>
              // 这种形式表明这是一个从host(CPU)代码调用device的代码，
              //并且括号中的数值表明，M个block，每个block有 N个线程,
              //所以这个函数总共有M*N个线程。
              num_kernels, data_im, data_offset, data_mask, im_shape[2],
              im_shape[3], kernel_shape[0], kernel_shape[1], pad[0], pad[1],
              stride[0], stride[1], dilation[0], dilation[1],
              channel_per_deformable_group, col_shape[1], im_shape[1],
              deformable_group, col_shape[2], col_shape[3], data_col);
      // MSHADOW_CUDA_POST_KERNEL_CHECK(modulated_deformable_im2col_gpu_kernel);
      break;
    default:
      LOG(FATAL) << "im2col_nd_gpu does not support computation with "
                 << num_spatial_axes << " spatial axes";
  }
}

template <typename DType>
void SetZeros<GPUDevice, DType>::operator()(const GPUDevice &d, int n,
                                            DType *result_data) {
  CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
  SetZeroKernel<DType>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          n, result_data);
}

template <typename DType>
void PureAddTo<GPUDevice, DType>::operator()(const GPUDevice &d, const int n,
                                             DType *result_data,
                                             const DType *right_data) {
  CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
  PureAddToKernel<DType>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          n, result_data, right_data);
}

template <typename DType>
void SetOne<GPUDevice, DType>::operator()(const GPUDevice &d, int n,
                                          DType *result_data) {
  CudaLaunchConfig config = GetCudaLaunchConfig(n, d);
  SetOneKernel<DType>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          n, result_data);
}

template <typename DType>
void SetNumAtIndex<GPUDevice, DType>::operator()(const GPUDevice &d, DType num,
                                                 int index, DType *data) {
  CudaLaunchConfig config = GetCudaLaunchConfig(1, d);
  SetNumAtIndexKernel<DType>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          num, index, data);
}

// 如果没有在这里实例化的话, 生成的.so会报类似于 undefined symbol:
// _ZN10tensorflow13setNumAtIndexIN5Eigen9GpuDeviceEfEclERKS2_fiPf的错误 I guess
// the reason for instancing the functional structure below is that certifying
// single functor instance for every functor.
template struct DeformableConv2DIm2Col<GPUDevice, double>;
template struct DeformableConv2DCol2Im<GPUDevice, double>;
template struct DeformableConv2DCol2ImCoord<GPUDevice, double>;
template struct PureAddTo<GPUDevice, double>;
template struct SetOne<GPUDevice, double>;
template struct SetZeros<GPUDevice, double>;
template struct SwapAxis<GPUDevice, double>;
template struct SetNumAtIndex<GPUDevice, double>;

template struct DeformableConv2DIm2Col<GPUDevice, float>;
template struct DeformableConv2DCol2Im<GPUDevice, float>;
template struct DeformableConv2DCol2ImCoord<GPUDevice, float>;
template struct PureAddTo<GPUDevice, float>;
template struct SetOne<GPUDevice, float>;
template struct SetZeros<GPUDevice, float>;
template struct SwapAxis<GPUDevice, float>;
template struct SetNumAtIndex<GPUDevice, float>;
template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T *cuda_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T *>(cuda_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}

class CublasScratchAllocator : public se::ScratchAllocator {
 public:
  using Stream = se::Stream;
  using DeviceMemoryBytes = se::DeviceMemory<uint8>;

  CublasScratchAllocator(OpKernelContext *context) : context_(context) {}

  int64 GetMemoryLimitInBytes() override { return -1; }

  se::port::StatusOr<DeviceMemoryBytes> AllocateBytes(
      int64 byte_size) override {
    Tensor temporary_memory;

    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory));
    if (!allocation_status.ok()) {
      return se::port::StatusOr<DeviceMemoryBytes>(
          DeviceMemoryBytes::MakeFromByteSize(nullptr, 0));
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    return se::port::StatusOr<DeviceMemoryBytes>(
        DeviceMemoryBytes::MakeFromByteSize(
            temporary_memory.flat<uint8>().data(),
            temporary_memory.flat<uint8>().size()));
  }

  se::port::StatusOr<DeviceMemoryBytes> AllocateBytes(Stream *stream,
                                                      int64 byte_size) {
    Tensor temporary_memory;

    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory));
    if (!allocation_status.ok()) {
      return se::port::StatusOr<DeviceMemoryBytes>(
          DeviceMemoryBytes::MakeFromByteSize(nullptr, 0));
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    return se::port::StatusOr<DeviceMemoryBytes>(
        DeviceMemoryBytes::MakeFromByteSize(
            temporary_memory.flat<uint8>().data(),
            temporary_memory.flat<uint8>().size()));
  }

 private:
  OpKernelContext *context_;
  std::vector<Tensor> allocated_tensors_;
};

template <typename Scalar>
void LaunchBatchMatMul<GPUDevice, Scalar>::launch(
    OpKernelContext *context, const TensorShape &in_x_shape,
    const TensorShape &in_y_shape, const Scalar *in_x_ptr,
    const Scalar *in_y_ptr, bool adj_x, bool adj_y, Scalar *out) {
  constexpr se::blas::Transpose kTranspose =
      is_complex<Scalar>::value ? se::blas::Transpose::kConjugateTranspose
                                : se::blas::Transpose::kTranspose;
  se::blas::Transpose trans[] = {se::blas::Transpose::kNoTranspose, kTranspose};

  const uint64 m = in_x_shape.dim_size(adj_x ? 2 : 1);
  const uint64 k = in_x_shape.dim_size(adj_x ? 1 : 2);
  const uint64 n = in_y_shape.dim_size(adj_y ? 1 : 2);
  const uint64 batch_size = in_x_shape.dim_size(0);
  auto blas_transpose_a = trans[adj_x];
  auto blas_transpose_b = trans[adj_y];

  auto *stream = context->op_device_context()->stream();
  OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

  typedef se::DeviceMemory<Scalar> DeviceMemoryType;
  std::vector<DeviceMemoryType> a_device_memory;
  std::vector<DeviceMemoryType> b_device_memory;
  std::vector<DeviceMemoryType> c_device_memory;
  std::vector<DeviceMemoryType *> a_ptrs;
  std::vector<DeviceMemoryType *> b_ptrs;
  std::vector<DeviceMemoryType *> c_ptrs;
  a_device_memory.reserve(batch_size);
  b_device_memory.reserve(batch_size);
  c_device_memory.reserve(batch_size);
  a_ptrs.reserve(batch_size);
  b_ptrs.reserve(batch_size);
  c_ptrs.reserve(batch_size);
  auto *a_base_ptr = in_x_ptr;
  auto *b_base_ptr = in_y_ptr;
  auto *c_base_ptr = out;
  for (int64 i = 0; i < batch_size; ++i) {
    a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
    b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
    c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
    a_ptrs.push_back(&a_device_memory.back());
    b_ptrs.push_back(&b_device_memory.back());
    c_ptrs.push_back(&c_device_memory.back());
  }

  typedef Scalar Coefficient;

  // Cublas does
  // C = A x B
  // where A, B and C are assumed to be in column major.
  // We want the output to be in row-major, so we can compute
  // C' = B' x A', where ' stands for transpose (not adjoint).
  // TODO(yangzihao): Choose the best of the three strategies using autotune.
  if (batch_size == 1) {
    // This is a regular matrix*matrix or matrix*vector multiply. Avoid the
    // overhead of the scratch allocator and the batch interface.
    if (n == 1 &&
        blas_transpose_b != se::blas::Transpose::kConjugateTranspose &&
        blas_transpose_a != se::blas::Transpose::kConjugateTranspose) {
      // This is a matrix*vector multiply so use GEMV to compute A * b.
      // Here we are multiplying in the natural order, so we have to flip
      // the transposition flag to compensate for the tensor being stored
      // row-major. Since GEMV doesn't provide a way to just conjugate an
      // argument, we have to defer those cases to GEMM below.
      auto gemv_trans_a = blas_transpose_a == se::blas::Transpose::kTranspose
                              ? se::blas::Transpose::kNoTranspose
                              : se::blas::Transpose::kTranspose;
      bool blas_launch_status =
          stream
              ->ThenBlasGemv(gemv_trans_a, adj_x ? m : k, adj_x ? k : m,
                             static_cast<Coefficient>(1.0), *(a_ptrs[0]),
                             adj_x ? m : k, *(b_ptrs[0]), 1,
                             static_cast<Coefficient>(0.0), c_ptrs[0], 1)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal(
            "Blas xGEMV launch failed : a.shape=", in_x_shape.DebugString(),
            ", b.shape=", in_y_shape.DebugString(), ", m=", m, ", n=", n,
            ", k=", k));
      }
    } else {
      bool blas_launch_status =
          stream
              ->ThenBlasGemm(blas_transpose_b, blas_transpose_a, n, m, k,
                             static_cast<Coefficient>(1.0), *(b_ptrs[0]),
                             adj_y ? k : n, *(a_ptrs[0]), adj_x ? m : k,
                             static_cast<Coefficient>(0.0), c_ptrs[0], n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal(
            "Blas xGEMM launch failed : a.shape=", in_x_shape.DebugString(),
            ", b.shape=", in_y_shape.DebugString(), ", m=", m, ", n=", n,
            ", k=", k));
      }
    }
  } else {
    CublasScratchAllocator scratch_allocator(context);
    bool blas_launch_status =
        stream
            ->ThenBlasGemmBatchedWithScratch(
                blas_transpose_b, blas_transpose_a, n, m, k,
                static_cast<Coefficient>(1.0), b_ptrs, adj_y ? k : n, a_ptrs,
                adj_x ? m : k, static_cast<Coefficient>(0.0), c_ptrs, n,
                batch_size, &scratch_allocator)
            .ok();
    if (!blas_launch_status) {
      context->SetStatus(errors::Internal(
          "Blas xGEMMBatched launch failed : a.shape=",
          in_x_shape.DebugString(), ", b.shape=", in_y_shape.DebugString(),
          ", m=", m, ", n=", n, ", k=", k, ", batch_size=", batch_size));
    }
  }
}
template <typename T>
void DeformablePSROIPoolForward<GPUDevice, T>::operator()(
    const GPUDevice &d, const int count, const T *bottom_data,
    const T spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const T *bottom_rois, const T *bottom_trans, const int no_trans,
    const T trans_std, const int sample_per_part, const int output_dim,
    const int group_size, const int part_size, const int num_classes,
    const int channels_each_class, T *top_data, T *top_count) {
  auto config = GetCudaLaunchConfig(count, d);
  DeformablePSROIPoolForwardKernel<T>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          count, bottom_data, spatial_scale, channels, height, width,
          pooled_height, pooled_width, bottom_rois, bottom_trans, no_trans,
          trans_std, sample_per_part, output_dim, group_size, part_size,
          num_classes, channels_each_class, top_data, top_count);
}
template <typename T>
void DeformablePSROIPoolBackwardKernel<GPUDevice, T>::operator()(
    const GPUDevice &d, const int count, const T *top_diff, const T *top_count,
    const int num_rois, const T spatial_scale, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int output_dim, T *bottom_data_diff,
    T *bottom_trans_diff, const T *bottom_data, const T *bottom_rois,
    const T *bottom_trans, const int no_trans, const T trans_std,
    const int sample_per_part, const int group_size, const int part_size,
    const int num_classes, const int channels_each_class) {
  auto config = GetCudaLaunchConfig(count, d);
  DeformablePSROIPoolBackwardAccKernel<T>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          count, top_diff, top_count, num_rois, spatial_scale, channels, height,
          width, pooled_height, pooled_width, output_dim, bottom_data_diff,
          bottom_trans_diff, bottom_data, bottom_rois, bottom_trans, no_trans,
          trans_std, sample_per_part, group_size, part_size, num_classes,
          channels_each_class);
}
template struct LaunchBatchMatMul<GPUDevice, float>;
template struct LaunchBatchMatMul<GPUDevice, double>;
template struct DeformablePSROIPoolForward<GPUDevice, float>;
template struct DeformablePSROIPoolForward<GPUDevice, double>;
template struct DeformablePSROIPoolBackwardKernel<GPUDevice, float>;
template struct DeformablePSROIPoolBackwardKernel<GPUDevice, double>;
}  // namespace functor
}  // namespace addons
}  // namespace tensorflow
#endif
