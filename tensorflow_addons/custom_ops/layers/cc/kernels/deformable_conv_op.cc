
#include "tensorflow_addons/custom_ops/layers/cc/kernels/deformable_conv_op.h"

#include <algorithm>
#include <atomic>
#include <mutex>

#include "tensorflow_addons/custom_ops/layers/cc/kernels/deformable_conv2d_utils.h"

namespace tensorflow {
namespace addons {

namespace functor {

template <typename DType>
DType DmcnIm2colBilinear(const DType *bottom_data, const int data_width,
                         const int height, const int width, DType h, DType w) {
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
DType DmcnGetGradientWeight(DType argmax_h, DType argmax_w, const int h,
                            const int w, const int height, const int width) {
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
DType DmcnGetCoordinateWeight(DType argmax_h, DType argmax_w, const int height,
                              const int width, const DType *im_data,
                              const int data_width, const int bp_dir) {
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

using ull = unsigned long long int;
using uInt = unsigned int;
typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

Eigen::IndexPair<Eigen::DenseIndex> ContractionDims(bool adj_x, bool adj_y) {
  return {adj_x ? 0 : 1, adj_y ? 1 : 0};
}
#if PLATFORM_WINDOWS
template <typename T>
void AtomicAdd(T *address, T val) {
  static std::mutex mu;
  std::lock_guard<std::mutex> lk(mu);
  *address += val;
}
#else
void AtomicAdd(float *address, float val) {
  auto *address_as_ull = reinterpret_cast<uInt *>(address);
  uInt old = *address_as_ull;
  uInt assumed;
  float desired;
  do {
    assumed = old;
    desired = *reinterpret_cast<float *>(&assumed) + static_cast<float>(val);
    old = __sync_val_compare_and_swap(address_as_ull, assumed,
                                      *reinterpret_cast<uInt *>(&desired));
  } while (assumed != old);
}

void AtomicAdd(double *address, double val) {
  auto *address_as_ull = reinterpret_cast<ull *>(address);
  ull old = *address_as_ull;
  ull assumed;
  double desired;
  do {
    assumed = old;
    desired = *reinterpret_cast<double *>(&assumed) + static_cast<double>(val);
    old = __sync_val_compare_and_swap(address_as_ull, assumed,
                                      *reinterpret_cast<ull *>(&desired));
  } while (assumed != old);
}
#endif

template <typename DType>
void SwapAxisKernel(const CPUDevice &d, const int n, const int cuda_mem_size,
                    const int min_unit_size, DType *input_data,
                    const int dim_num, const int axis_x_dims,
                    const int axis_y_dims, const int axis_x, const int axis_y) {
  d.parallelFor(n,
                Eigen::TensorOpCost(cuda_mem_size, cuda_mem_size,
                                    cuda_mem_size * axis_y_dims * axis_x_dims),
                [min_unit_size, input_data, dim_num, axis_x_dims, axis_y_dims,
                 axis_x, axis_y, cuda_mem_size](int64 start, int64 end) {
                  for (int64 index = start; index < end; index++) {
                    auto *device_data = new DType[cuda_mem_size];
                    DType *input_data_ptr = input_data + index * cuda_mem_size;
                    for (int j = 0; j < axis_y_dims; j++) {
                      for (int i = 0; i < axis_x_dims; i++) {
                        DType *temp_ptr = input_data_ptr +
                                          (i * axis_x_dims + j) * min_unit_size;
                        DType *device_data_temp_ptr =
                            device_data + (j * axis_y_dims + i) * min_unit_size;
                        for (int k = 0; k < min_unit_size; k++) {
                          *(device_data_temp_ptr + k) = *(temp_ptr + k);
                        }
                      }
                    }
                    for (int idx = 0; idx < cuda_mem_size; idx++) {
                      *(input_data_ptr + idx) = *(device_data + idx);
                    }
                    delete[] device_data;
                  }
                });
}

template <typename T>
void DeformablePSROIPoolBackwardCpuAccKernel(
    const CPUDevice &d, const int count, const T *top_diff, const T *top_count,
    const int num_rois, const T spatial_scale, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int output_dim, T *bottom_data_diff,
    T *bottom_trans_diff, const T *bottom_data, const T *bottom_rois,
    const T *bottom_trans, const int no_trans, const T trans_std,
    const int sample_per_part, const int group_size, const int part_size,
    const int num_classes, const int channels_each_class) {
  auto f = [count, top_diff, top_count, num_rois, spatial_scale, channels,
            height, width, pooled_height, pooled_width, output_dim,
            bottom_data_diff, bottom_trans_diff, bottom_data, bottom_rois,
            bottom_trans, no_trans, trans_std, sample_per_part, group_size,
            part_size, num_classes,
            channels_each_class](int64 start, int64 end) {
    for (int64 index = start; index < end; ++index) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int ctop = (index / pooled_width / pooled_height) % output_dim;
      int n = index / pooled_width / pooled_height / output_dim;
      // [start, end) interval for spatial sampling
      const T *offset_bottom_rois = bottom_rois + n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      T roi_start_w = (T)(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
      T roi_start_h = (T)(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
      T roi_end_w =
          (T)(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
      T roi_end_h =
          (T)(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;
      // Force too small ROIs to be 1x1
      T roi_width =
          std::max(roi_end_w - roi_start_w, static_cast<T>(0.1));  // avoid 0
      T roi_height = std::max(roi_end_h - roi_start_h, static_cast<T>(0.1));

      // Compute w and h at bottom
      T bin_size_h = roi_height / static_cast<T>(pooled_height);
      T bin_size_w = roi_width / static_cast<T>(pooled_width);

      T sub_bin_size_h = bin_size_h / static_cast<T>(sample_per_part);
      T sub_bin_size_w = bin_size_w / static_cast<T>(sample_per_part);

      int part_h = floor((T)(ph) / pooled_height * part_size);
      int part_w = floor((T)(pw) / pooled_width * part_size);
      int class_id = ctop / channels_each_class;
      T trans_x =
          no_trans
              ? static_cast<T>(0)
              : bottom_trans[(((n * num_classes + class_id) * 2) * part_size +
                              part_h) *
                                 part_size +
                             part_w] *
                    (T)trans_std;
      T trans_y = no_trans
                      ? (T)(0)
                      : bottom_trans[(((n * num_classes + class_id) * 2 + 1) *
                                          part_size +
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
      gw = std::min(std::max(gw, 0), group_size - 1);
      gh = std::min(std::max(gh, 0), group_size - 1);
      for (int ih = 0; ih < sample_per_part; ih++) {
        for (int iw = 0; iw < sample_per_part; iw++) {
          T w = wstart + iw * sub_bin_size_w;
          T h = hstart + ih * sub_bin_size_h;
          // bilinear interpolation
          if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5) {
            continue;
          }
          w = std::min(std::max(w, static_cast<T>(0.)),
                       static_cast<T>(width - 1.));
          h = std::min(std::max(h, static_cast<T>(0.)),
                       static_cast<T>(height - 1.));
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
          AtomicAdd(
              offset_bottom_data_diff + bottom_index_base + y0 * width + x0,
              q00 * diff_val);
          AtomicAdd(
              offset_bottom_data_diff + bottom_index_base + y1 * width + x0,
              q01 * diff_val);
          AtomicAdd(
              offset_bottom_data_diff + bottom_index_base + y0 * width + x1,
              q10 * diff_val);
          AtomicAdd(
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

          AtomicAdd(
              bottom_trans_diff +
                  (((n * num_classes + class_id) * 2) * part_size + part_h) *
                      part_size +
                  part_w,
              diff_x);
          AtomicAdd(bottom_trans_diff +
                        (((n * num_classes + class_id) * 2 + 1) * part_size +
                         part_h) *
                            part_size +
                        part_w,
                    diff_y);
        }
      }
    }
  };
  d.parallelFor(count, Eigen::TensorOpCost(count, count, count), f);
}

template <typename T>
void DeformablePSROIPoolForwardCpuKernel(
    const CPUDevice &d, const int count, const T *bottom_data,
    const T spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const T *bottom_rois, const T *bottom_trans, const int no_trans,
    const T trans_std, const int sample_per_part, const int output_dim,
    const int group_size, const int part_size, const int num_classes,
    const int channels_each_class, T *top_data, T *top_count) {
  auto f = [count, bottom_data, spatial_scale, channels, height, width,
            pooled_height, pooled_width, bottom_rois, bottom_trans, no_trans,
            trans_std, sample_per_part, output_dim, group_size, part_size,
            num_classes, channels_each_class, top_data,
            top_count](int64 start, int64 end) {
    for (int64 index = start; index < end; ++index) {
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
      T roi_end_w =
          (T)(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
      T roi_end_h =
          (T)(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;
      // Force too small ROIs to be 1x1
      T roi_width =
          std::max(roi_end_w - roi_start_w, static_cast<T>(0.1));  // avoid 0
      T roi_height = std::max(roi_end_h - roi_start_h, static_cast<T>(0.1));
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
      T trans_y = no_trans
                      ? (T)(0)
                      : bottom_trans[(((n * num_classes + class_id) * 2 + 1) *
                                          part_size +
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
      gw = std::min(std::max(gw, 0), group_size - 1);
      gh = std::min(std::max(gh, 0), group_size - 1);
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
          w = std::min(std::max(w, static_cast<T>(0.)),
                       static_cast<T>(width - 1.));
          h = std::min(std::max(h, static_cast<T>(0.)),
                       static_cast<T>(height - 1.));
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
  };
  d.parallelFor(count, Eigen::TensorOpCost(count, count, count), f);
}
template <typename DType>
void DeformableConv2DIm2ColCPUKernel(
    const CPUDevice &d, const int n, const DType *data_im,
    const DType *data_offset, const DType *data_mask,

    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,

    const int
        channel_per_deformable_group,  // 输入图通道数除以deformable_group的数量,
    const int batch_size, const int num_channels,
    const int
        deformable_group,  //这里的batch_size代表的是im2col_step_, 一般就设为1了
    const int height_col, const int width_col, DType *data_col) {
  auto f = [n, data_im, data_offset, data_mask, height, width, kernel_h,
            kernel_w, pad_h, pad_w, stride_w, stride_h, dilation_w, dilation_h,
            channel_per_deformable_group, batch_size, num_channels,
            deformable_group, height_col, width_col,
            data_col](int64 start, int64 end) {
    for (int64 index = start; index < end; index++) {
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
          ((c_col * batch_size + b_col) * height_col + h_col) * width_col +
          w_col;
      const DType *data_im_ptr =
          data_im + (b_col * num_channels + c_im) * height * width;
      const DType *data_offset_ptr =
          data_offset + (b_col * deformable_group + deformable_group_index) *
                            2 * kernel_h * kernel_w * height_col *
                            width_col;  //

      const DType *data_mask_ptr =
          data_mask + (b_col * deformable_group + deformable_group_index) *
                          kernel_h * kernel_w * height_col * width_col;  //
      for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
          const int data_offset_h_ptr =
              ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col +
              w_col;
          const int data_offset_w_ptr =
              ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
              w_col;
          const int data_mask_hw_ptr =
              ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;
          const DType offset_h = data_offset_ptr[data_offset_h_ptr];
          const DType offset_w = data_offset_ptr[data_offset_w_ptr];
          const DType mask = data_mask_ptr[data_mask_hw_ptr];
          auto val = static_cast<DType>(0);
          const DType h_im = h_in + i * dilation_h + offset_h;
          const DType w_im = w_in + j * dilation_w + offset_w;
          if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
            val = DmcnIm2colBilinear(data_im_ptr, width, height, width, h_im,
                                     w_im);
          }
          *data_col_ptr = val * mask;
          data_col_ptr += batch_size * height_col * width_col;
        }
      }
    }
  };
  d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}

template <typename DType>
void DeformableConv2DCol2ImCPUKernel(
    const CPUDevice &d, const int n, const DType *data_col,
    const DType *data_offset, const DType *data_mask, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int channel_per_deformable_group, const int batch_size,
    const int deformable_group, const int height_col, const int width_col,
    DType *grad_im) {
  auto f = [n, data_col, data_offset, data_mask, channels, height, width,
            kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
            dilation_w, channel_per_deformable_group, batch_size,
            deformable_group, height_col, width_col,
            grad_im](int64 start, int64 end) {
    for (int64 index = start; index < end; ++index) {
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
          data_mask + (b * deformable_group + deformable_group_index) *
                          kernel_h * kernel_w * height_col * width_col;
      const int data_offset_h_ptr =
          ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
      const int data_offset_w_ptr =
          ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
          w_out;
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
                DmcnGetGradientWeight(cur_inv_h_data, cur_inv_w_data,
                                      cur_h + dy, cur_w + dx, height, width);
            AtomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
            //                      *(grad_im + cur_bottom_grad_pos) += weight *
            //                      cur_top_grad;
          }
        }
      }
    }
  };
  d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}
template <typename DType>
void DeformableConv2DCol2ImCoordCPUKernel(
    const CPUDevice &d, const int n, const DType *data_col,
    const DType *data_im, const DType *data_offset, const DType *data_mask,
    const int channels, const int height, const int width,  // 输入的C, H, W
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int channel_per_deformable_group,
    const int batch_size, const int offset_channels, const int deformable_group,
    const int height_col, const int width_col, DType *grad_offset,
    DType *grad_mask) {
  auto f = [n, data_col, data_im, data_offset, data_mask, channels, height,
            width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, channel_per_deformable_group, batch_size,
            offset_channels, deformable_group, height_col, width_col,
            grad_offset, grad_mask](int64 start, int64 end) {
    for (int64 index = start; index < end; index++) {
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
          data_mask + (b * deformable_group + deformable_group_index) *
                          kernel_h * kernel_w * height_col * width_col;

      const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

      for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group;
           col_c += col_step) {
        const int col_pos =
            (((col_c * batch_size + b) * height_col) + h) * width_col + w;
        const int bp_dir = offset_c % 2;

        int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
        int i = (col_pos / width_col / height_col / batch_size / kernel_w) %
                kernel_h;
        int w_out = col_pos % width_col;
        int h_out = (col_pos / width_col) % height_col;
        int w_in = w_out * stride_w - pad_w;
        int h_in = h_out * stride_h - pad_h;
        const int data_offset_h_ptr =
            (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col +
             w_out);
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
  };
  d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}
template <typename DType>
void PureAddToKernel(const CPUDevice &d, const int n, DType *result_data,
                     const DType *right_data) {
  auto f = [n, result_data, right_data](int64 start, int64 end) {
    for (int64 index = start; index < end; index++) {
      *(result_data + index) += (right_data[index]);
    }
  };
  d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}
template <typename DType>
void SetZeroKernel(const CPUDevice &d, const int n, DType *result_data) {
  auto f = [n, result_data](int64 start, int64 end) {
    for (int64 index = start; index < end; ++index) {
      *(result_data + index) = DType(0);
    }
  };
  d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}
template <typename DType>
void SetOneKernel(const CPUDevice &d, const int n, DType *result_data) {
  auto f = [n, result_data](int64 start, int64 end) {
    for (int64 index = start; index < end; ++index) {
      *(result_data + index) = DType(1);
    }
  };
  d.parallelFor(n, Eigen::TensorOpCost(n, n, n), f);
}

template <typename DType>
void DeformableConv2DCol2ImCoord<CPUDevice, DType>::operator()(
    const Eigen::ThreadPoolDevice &d, const DType *data_col,
    const DType *data_im, const DType *data_offset, const DType *data_mask,
    const TShape &im_shape, const TShape &col_shape, const TShape &kernel_shape,
    const TShape &pad, const TShape &stride, const TShape &dilation,
    const int32_t deformable_group, DType *grad_offset, DType *grad_mask) {
  int num_spatial_axes = kernel_shape.size();
  int num_kernels = col_shape[1] * col_shape[2] * col_shape[3] * 2 *
                    kernel_shape[0] * kernel_shape[1] * deformable_group;
  int channel_per_deformable_group = col_shape[0] / deformable_group;
  switch (num_spatial_axes) {
    case 2:
      DeformableConv2DCol2ImCoordCPUKernel<DType>(
          d, num_kernels, data_col, data_im, data_offset, data_mask,
          im_shape[1], im_shape[2], im_shape[3], kernel_shape[0],
          kernel_shape[1], pad[0], pad[1], stride[0], stride[1], dilation[0],
          dilation[1], channel_per_deformable_group, col_shape[1],
          2 * kernel_shape[0] * kernel_shape[1] * deformable_group,
          deformable_group, col_shape[2], col_shape[3], grad_offset, grad_mask);
      break;
    default:
      LOG(FATAL) << "col2im_nd_gpu does not support computation with "
                 << num_spatial_axes << "spatial axes";
  }
}

template <typename DType>
void DeformableConv2DCol2Im<CPUDevice, DType>::operator()(
    const Eigen::ThreadPoolDevice &d, const DType *data_col,
    const DType *data_offset, const DType *data_mask, const TShape &im_shape,
    const TShape &col_shape, const TShape &kernel_shape, const TShape &pad,
    const TShape &stride, const TShape &dilation,
    const int32_t deformable_group, DType *grad_im) {
  int num_spatial_axes = kernel_shape.size();
  int channel_per_deformable_group = im_shape[1] / deformable_group;
  int num_kernels = ProdShape(col_shape, 0, col_shape.size());
  // num_axes should be smaller than block size
  //   using namespace mxnet_op;
  switch (num_spatial_axes) {
    case 2:
      // To avoid involving atomic operations, we will launch one kernel per
      // bottom dimension, and then in the kernel add up the top dimensions.
      // NOLINT_NEXT_LINE(whitespace/operators)
      DeformableConv2DCol2ImCPUKernel<DType>(
          d, num_kernels, data_col, data_offset, data_mask, im_shape[1],
          im_shape[2], im_shape[3], kernel_shape[0], kernel_shape[1], pad[0],
          pad[1], stride[0], stride[1], dilation[0], dilation[1],
          channel_per_deformable_group, col_shape[1], deformable_group,
          col_shape[2], col_shape[3], grad_im);
      break;
    default:
      LOG(FATAL) << "col2im_nd_gpu does not support computation with "
                 << num_spatial_axes << " spatial axes";
  }
}

template <typename DType>
void DeformableConv2DIm2Col<CPUDevice, DType>::operator()(
    const Eigen::ThreadPoolDevice &d, const DType *data_im,
    const DType *data_offset, const DType *data_mask, const TShape &im_shape,
    const TShape &col_shape, const TShape &kernel_shape, const TShape &pad,
    const TShape &stride, const TShape &dilation,
    const int32_t deformable_group, DType *data_col) {
  int num_spatial_axes = kernel_shape.size();
  int channel_per_deformable_group =
      im_shape[1] / deformable_group;  // imshape[1] = 输入图的通道数
  int num_kernels =
      im_shape[1] *
      ProdShape(col_shape, 1,
                col_shape.size());  // K * N / k.Size(), k = filter, col_shape =
                                    // [K, im2col_step_, H, W]
  switch (num_spatial_axes) {
    case 2:
      DeformableConv2DIm2ColCPUKernel<DType>(
          d, num_kernels, data_im, data_offset, data_mask, im_shape[2],
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
void SetZeros<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d,
                                            int n, DType *result_data) {
  SetZeroKernel(d, n, result_data);
}
template <typename DType>
void PureAddTo<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d,
                                             const int n, DType *result_data,
                                             const DType *right_data) {
  PureAddToKernel(d, n, result_data, right_data);
}
template <typename DType>
void SetOne<CPUDevice, DType>::operator()(const Eigen::ThreadPoolDevice &d,
                                          int n, DType *result_data) {
  SetOneKernel(d, n, result_data);
}
template <typename DType>
void SetNumAtIndex<CPUDevice, DType>::operator()(
    const Eigen::ThreadPoolDevice &d, DType num, int index, DType *data) {
  *(data + index) = num;
}

template <typename T>
void LaunchBatchMatMul<CPUDevice, T>::launch(OpKernelContext *context,
                                             const TensorShape &in_x_shape,
                                             const TensorShape &in_y_shape,
                                             const T *in_x_ptr,
                                             const T *in_y_ptr, bool adj_x,
                                             bool adj_y, T *out) {
  const int64 m = in_x_shape.dim_size(adj_x ? 2 : 1);
  const int64 k = in_x_shape.dim_size(adj_x ? 1 : 2);
  const int64 n = in_y_shape.dim_size(adj_y ? 1 : 2);
  const uint64 batch_size = in_x_shape.dim_size(0);
  Eigen::TensorMap<Eigen::Tensor<const T, 3, Eigen::RowMajor>> t_in_x(
      in_x_ptr, in_x_shape.AsEigenDSizes<3, Eigen::DenseIndex>());
  Eigen::TensorMap<Eigen::Tensor<const T, 3, Eigen::RowMajor>> t_in_y(
      in_y_ptr, in_y_shape.AsEigenDSizes<3, Eigen::DenseIndex>());
  Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> t_out(out, batch_size,
                                                               m, n);
  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
  contract_pairs[0] = ContractionDims(adj_x, adj_y);
  auto &device = context->eigen_device<CPUDevice>();
  for (int i = 0; i < t_out.dimension(0); ++i) {
    t_out.template chip<0>(i).device(device) =
        (t_in_x.template chip<0>(i))
            .template contract(t_in_y.template chip<0>(i), contract_pairs);
  }
}

template <typename T>
void DeformablePSROIPoolForward<CPUDevice, T>::operator()(
    const CPUDevice &d, const int count, const T *bottom_data,
    const T spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const T *bottom_rois, const T *bottom_trans, const int no_trans,
    const T trans_std, const int sample_per_part, const int output_dim,
    const int group_size, const int part_size, const int num_classes,
    const int channels_each_class, T *top_data, T *top_count) {
  DeformablePSROIPoolForwardCpuKernel<T>(
      d, count, bottom_data, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_rois, bottom_trans, no_trans,
      trans_std, sample_per_part, output_dim, group_size, part_size,
      num_classes, channels_each_class, top_data, top_count);
}

template <typename T>
void DeformablePSROIPoolBackwardKernel<CPUDevice, T>::operator()(
    const CPUDevice &d, const int count, const T *top_diff, const T *top_count,
    const int num_rois, const T spatial_scale, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int output_dim, T *bottom_data_diff,
    T *bottom_trans_diff, const T *bottom_data, const T *bottom_rois,
    const T *bottom_trans, const int no_trans, const T trans_std,
    const int sample_per_part, const int group_size, const int part_size,
    const int num_classes, const int channels_each_class) {
  DeformablePSROIPoolBackwardCpuAccKernel<T>(
      d, count, top_diff, top_count, num_rois, spatial_scale, channels, height,
      width, pooled_height, pooled_width, output_dim, bottom_data_diff,
      bottom_trans_diff, bottom_data, bottom_rois, bottom_trans, no_trans,
      trans_std, sample_per_part, group_size, part_size, num_classes,
      channels_each_class);
}
template struct DeformableConv2DIm2Col<CPUDevice, double>;
template struct DeformableConv2DCol2Im<CPUDevice, double>;
template struct DeformableConv2DCol2ImCoord<CPUDevice, double>;
template struct PureAddTo<CPUDevice, double>;
template struct SetOne<CPUDevice, double>;
template struct SetZeros<CPUDevice, double>;
template struct SwapAxis<CPUDevice, double>;
template struct SetNumAtIndex<CPUDevice, double>;

template struct DeformableConv2DIm2Col<CPUDevice, float>;
template struct DeformableConv2DCol2Im<CPUDevice, float>;
template struct DeformableConv2DCol2ImCoord<CPUDevice, float>;
template struct PureAddTo<CPUDevice, float>;
template struct SetOne<CPUDevice, float>;
template struct SetZeros<CPUDevice, float>;
template struct SwapAxis<CPUDevice, float>;
template struct SetNumAtIndex<CPUDevice, float>;

template struct LaunchBatchMatMul<CPUDevice, float>;
template struct LaunchBatchMatMul<CPUDevice, double>;
template struct DeformablePSROIPoolForward<CPUDevice, float>;
template struct DeformablePSROIPoolForward<CPUDevice, double>;
template struct DeformablePSROIPoolBackwardKernel<CPUDevice, float>;
template struct DeformablePSROIPoolBackwardKernel<CPUDevice, double>;

template <typename Device, typename T>
class DeformableConv2DOp : public OpKernel {
 public:
  explicit DeformableConv2DOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, InitDeformableConv2DParameters(ctx, &params_));
  }
  void Compute(OpKernelContext *context) override {
    // Input tensor's shape
    // [batch, channels, height, weight]
    const Tensor &input = context->input(0);
    const TensorShape &input_shape = input.shape();
    // [out_channels, in_channels, filter_height, filter_weight]
    const Tensor &filter = context->input(1);
    const TensorShape &filter_shape = filter.shape();
    // [batch, 2 * filter.Size(), out_height, out_weight]
    const Tensor &offset = context->input(2);
    const TensorShape &offset_shape = offset.shape();
    // [batch, filter.Size(), out_height, out_weight]
    const Tensor &mask = context->input(3);
    const TensorShape &mask_shape = mask.shape();

    DeformableConv2DDimensions dimensions;
    OP_REQUIRES_OK(context, ComputeDeformableConv2DDimension(
                                params_, input, filter, &dimensions, 0));
    // data_format = NCHW
    // 这个地方我出了bug,原因是shapefromformat的参数必须是data_format, N, H, W,
    // C,因为其内部是根据data_format来决定是否需要进行transpose,
    // 如何第三个参数给了C, 且第一个参数为NCHW,那最后得到的结果会是NWCH
    TensorShape out_shape = ShapeFromFormat(
        params_.data_format, dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_depth, out_rows, out_cols]
    // Tensor* output = nullptr;
    // OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    VLOG(2) << "DeformableConv2D: in_depth = " << dimensions.in_depth
            << ", patch_depth = " << dimensions.patch_depth
            << ", input_cols = " << dimensions.input_cols
            << ", filter_cols = " << dimensions.filter_cols
            << ", input_rows = " << dimensions.input_rows
            << ", filter_rows = " << dimensions.filter_rows
            << ", stride_rows = " << dimensions.stride_rows
            << ", stride_cols = " << dimensions.stride_cols
            << ", dilation_rows = " << dimensions.dilation_rows
            << ", dilation_cols = " << dimensions.dilation_cols
            << ", out_depth = " << dimensions.out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    /**
     * from here i stop use the traditional convolution implement of the
     * official code which was defined in conv_ops.cc and began to use the
     * implement of the deformable conv2d of the msra version
     * **/
    LayerSetUp(input_shape, filter_shape, offset_shape, mask_shape, out_shape);
    // notice the fact that the flat function return a reference of a pointer,
    // but in fact we only need a pointer
    const T *in_data_ptr = input.template flat<T>().data();
    const T *offset_ptr = offset.template flat<T>().data();
    const T *mask_ptr = mask.template flat<T>().data();
    const Device &d = context->eigen_device<Device>();
    int col_buffer_shape_temp[4];  // calculate the shape of col_buffer,
                                   // mxnet源码是 + 1, 多了一个im2col_step_
    col_buffer_shape_temp[0] = ProdShape(
        filter_shape, 1,
        filter_shape
            .dims());  // 卷积核的参数个数,注意卷积核的形状应该是[out_depth,
                       // in_depth, height, weight]
    col_buffer_shape_temp[1] = im2col_step_;
    col_buffer_shape_temp[2] = out_shape.dim_size(2);
    col_buffer_shape_temp[3] = out_shape.dim_size(3);
    TensorShape col_buffer_shape =
        TensorShape({col_buffer_shape_temp[0], col_buffer_shape_temp[1],
                     col_buffer_shape_temp[2], col_buffer_shape_temp[3]});

    Tensor col_buffer;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          col_buffer_shape, &col_buffer));
    T *col_buffer_ptr = col_buffer.template flat<T>().data();

    int32_t M = conv_out_channels_ / group_;  // filter的数量
    int32_t N = im2col_step_ * conv_out_spatial_dim_;
    int32_t K = kernel_dim_;  // 卷积的参数个数

    Tensor weight_3d;
    TensorShape weight_3d_shape = TensorShape({group_, M, K});
    OP_REQUIRES(context, weight_3d.CopyFrom(filter, weight_3d_shape),
                errors::InvalidArgument("shape doesn't match"));
    T *weight_3d_ptr = weight_3d.template flat<T>().data();

    Tensor *output_temp_4d = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output_temp_4d));
    auto output_temp_4d_ptr = output_temp_4d->template flat<T>().data();
    //      auto output__ptr = output_temp_4d->flat<T>();
    /**
     * 这样的话下面计算矩阵乘法的时候直接就写到这个输出里了
     * 但是注意的是作者实现的时候划分ｓｔｅｐ，这个时候其实是往ｓｈａｐｅ为｛num_
     * / im2col_step_, group_, M,
     * N｝的输出里写的，所以最后一定要置换一下维度的位置
     * **/
    SetZeros<Device, T>()(d, ProdShape(out_shape, 0, out_shape.dims()),
                          output_temp_4d_ptr);
    TShape pads;
    pads.push_back(dimensions.pad_rows);
    pads.push_back(dimensions.pad_cols);
    for (int32_t n = 0; n < num_ / im2col_step_; ++n) {  // 分batch进行
      // transform image to col_buffer in order to use gemm
      DeformableConv2DIm2Col<Device, T>()(
          d,
          in_data_ptr +
              n * im2col_step_ *
                  input_dim_,  // dptr是获取输入数据的指针 + n * im2col_step_*
                               // input_dim 是让指针向后移动 一张图片的数据
          offset_ptr + n * im2col_step_ * input_offset_dim_,  //
          mask_ptr + n * im2col_step_ * input_mask_dim_, ToVector(input_shape),
          ToVector(col_buffer_shape), SubVector(filter_shape, 2, 4), pads,
          SubVector(params_.strides, 2, 4), SubVector(params_.dilations, 2, 4),
          params_.deformable_groups, col_buffer_ptr);
      TensorShape col_buffer_3d_shape = TensorShape({group_, K, N});

      auto output_temp_group_ptr = output_temp_4d_ptr + (n * group_ * M * N);

      LaunchBatchMatMul<Device, T>::launch(
          context, weight_3d_shape, col_buffer_3d_shape, weight_3d_ptr,
          col_buffer_ptr, false, false, output_temp_group_ptr);
    }
  }

 private:
  DeformableConv2DParameters params_;
  bool use_cudnn_;
  bool cudnn_use_autotune_;
  int32_t channel_axis_;       // channel axis of the input
  int32_t channels_;           // number of channels of input image
  int32_t num_spatial_axes_;   // number of spatial axes
  int32_t num_;                // batch size
  int32_t group_;              // number of groups
  int32_t conv_out_channels_;  // number of output channels (num_filter)
  int32_t
      conv_out_spatial_dim_;  // number of pixels of output images per channel
  int32_t conv_in_channels_;  // number of input channels
  int32_t kernel_dim_;     // number of input channels per group * kernel size
  int32_t weight_offset_;  // number of output channels per group * kernel_dim_
  int32_t col_offset_;
  int32_t output_offset_;
  int32_t col_buffer_size_;
  int32_t input_dim_;
  int32_t input_offset_dim_;
  int32_t input_mask_dim_;
  int32_t output_dim_;
  int32_t num_kernels_im2col_;
  int32_t num_kernels_col2im_;
  int32_t im2col_step_;
  bool bias_term_;  // has bias term?
  bool is_1x1_;
  void LayerSetUp(const TensorShape &ishape, const TensorShape &filter_shape,
                  const TensorShape &offset_shape,
                  const TensorShape &mask_shape, const TensorShape &oshape) {
    channel_axis_ = 1;  // hard code channel axis, fixed the input data_format
    const int32_t first_spatial_axis = channel_axis_ + 1;
    const int32_t num_axes = filter_shape.dims();
    num_spatial_axes_ =
        num_axes -
        first_spatial_axis;  //表示的是空间坐标个数,比如说2维卷积里,就是2,
                             // 3维卷积里就是3
    is_1x1_ = true;  //  判断是否为1x1卷积
    for (int32_t i = 2; i < filter_shape.dims(); ++i) {
      // is_1x1_ &= filter_shape.dim_size(i) == 1 && params_.stride[i] == 1 &&
      // params_.pad[i] == 0;
      is_1x1_ &=
          filter_shape.dim_size(i) == 1;  // only judge by the filter's shape
      if (!is_1x1_) break;
    }
    num_ = ishape.dim_size(0);                      // batch size
    channels_ = ishape.dim_size(1);                 // number of input channels
    group_ = params_.num_groups;                    //
    conv_out_channels_ = filter_shape.dim_size(0);  // output channel nums
    conv_in_channels_ = channels_;                  // input channel nums
    bias_term_ = !params_.no_bias;                  //
    kernel_dim_ =
        conv_in_channels_ / group_ * filter_shape.dim_size(2) *
        filter_shape.dim_size(
            3);  // Size()返回tensor中元素个数，即各维度大小的乘积，所以这里的kernel_dim的意思是卷积核的参数个数了．
    conv_out_spatial_dim_ = ProdShape(
        oshape, 2,
        oshape
            .dims());  // ProdShape(dimstart, dimend)返回指定维度大小乘积,
                       // 这个变量代表每个通道的像素点个数,
                       // oshape.ndim()返回这个shape的维度，假设是NCHW那么返回4,则为
                       // H * W，
    //        col_offset_ = kernel_dim_ *
    //        conv_out_spatial_dim_;//kernel_dim代表一个卷积核参数的个数，conv_out_spatial_dim_相当于特征图上的坐标个数，那这个变量相当于总共需要的偏移量
    //        weight_offset_ = conv_out_channels_ * kernel_dim_ /
    //        group_;//这里应该是所有的权重的个数，也就是需要求的权重偏移的个数
    //        output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ /
    //        group_;//这里是输出通道数乘上每个通道的像素点的个数，所以结果应该是输出的总维度，就是C*H*W
    im2col_step_ = std::min(params_.im2col_step, num_);
    col_buffer_size_ =
        kernel_dim_ * group_ * im2col_step_ *
        conv_out_spatial_dim_;  // 开辟的缓存大小// size of the column buffer
                                // used for storing im2col-ed pixels

    input_dim_ = ProdShape(
        ishape, 1,
        ishape.dims());  // input image size (#channels * height * width)
    input_offset_dim_ =
        ProdShape(offset_shape, 1, offset_shape.dims());  // 18 * H * W
    input_mask_dim_ = ProdShape(mask_shape, 1, mask_shape.dims());  // 9 * H * W
    output_dim_ = ProdShape(oshape, 1, oshape.dims());  //输出的元素个数

    num_kernels_im2col_ =
        conv_in_channels_ *
        conv_out_spatial_dim_;  //如果输出和输入的分辨率不变的话，代表输入数据的dim,我个人觉得就是把整个输入展开为一个一维向量,在求其维度大小
    num_kernels_col2im_ = input_dim_;  //输入数据的dim
  }
};

template <typename Device, typename T>
class DeformableConv2DBackPropOp : public OpKernel {
 public:
  explicit DeformableConv2DBackPropOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, InitDeformableConv2DParameters(context, &params_));
  }
  void Compute(OpKernelContext *ctx) override {
    const Tensor &x = ctx->input(0);
    const TensorShape &x_shape = x.shape();
    const Tensor &filter = ctx->input(1);
    const TensorShape &filter_shape = filter.shape();
    const Tensor &offset = ctx->input(2);
    const TensorShape &offset_shape = offset.shape();
    const Tensor &mask = ctx->input(3);
    const TensorShape &mask_shape = mask.shape();
    const Tensor &out_grad = ctx->input(4);
    const TensorShape &out_grad_shape = out_grad.shape();
    DeformableConv2DDimensions dimensions;
    OP_REQUIRES_OK(ctx, ComputeDeformableConv2DDimension(params_, x, filter,
                                                         &dimensions, 1));
    LayerSetUp(x_shape, filter_shape, offset_shape, mask_shape, out_grad_shape);
    const Device &d = ctx->eigen_device<Device>();
    int col_buffer_shape_temp[4];
    col_buffer_shape_temp[0] = ProdShape(filter_shape, 1, filter_shape.dims());
    col_buffer_shape_temp[1] = im2col_step_;
    col_buffer_shape_temp[2] = out_grad_shape.dim_size(2);
    col_buffer_shape_temp[3] = out_grad_shape.dim_size(3);
    TensorShape col_buffer_shape =
        TensorShape({col_buffer_shape_temp[0], col_buffer_shape_temp[1],
                     col_buffer_shape_temp[2], col_buffer_shape_temp[3]});
    int32_t M = kernel_dim_;
    int32_t N = im2col_step_ * conv_out_spatial_dim_;
    int32_t K = conv_out_channels_ / group_;
    const auto x_ptr = x.template flat<T>().data();
    const auto offset_ptr = offset.template flat<T>().data();
    const auto mask_ptr = mask.template flat<T>().data();
    const auto weight_3d_ptr = filter.template flat<T>().data();
    TensorShape weight_3d_shape = TensorShape({group_, K, M});
    Tensor out_grad_4d;
    TensorShape out_grad_4d_shape =
        TensorShape({num_ / im2col_step_, im2col_step_, conv_out_channels_,
                     conv_out_spatial_dim_});
    OP_REQUIRES(ctx, out_grad_4d.CopyFrom(out_grad, out_grad_4d_shape),
                errors::InvalidArgument("shape doesn't match"));
    auto out_grad_4d_ptr = out_grad_4d.template flat<T>().data();
    out_grad_4d_shape = TensorShape({num_ / im2col_step_, group_, K, N});
    Tensor col_buffer;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           col_buffer_shape, &col_buffer));
    auto col_buffer_3d_ptr = col_buffer.template flat<T>().data();
    TensorShape col_buffer_3d_shape = TensorShape({group_, M, N});
    Tensor *dweight_3d = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, filter_shape, &dweight_3d));
    T *dweight_3d_ptr = dweight_3d->template flat<T>().data();
    Tensor *x_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x_shape, &x_grad));
    T *x_grad_ptr = x_grad->template flat<T>().data();
    Tensor *offset_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, offset_shape, &offset_grad));
    T *offset_grad_ptr = offset_grad->template flat<T>().data();

    Tensor *mask_grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, mask_shape, &mask_grad));
    T *mask_grad_ptr = mask_grad->template flat<T>().data();
    TShape pads;
    pads.push_back(dimensions.pad_rows);
    pads.push_back(dimensions.pad_cols);
    TShape kernel_shape = SubVector(filter_shape, 2, 4);
    TShape stride_shape = SubVector(params_.strides, 2, 4);
    TShape dilation_shape = SubVector(params_.dilations, 2, 4);
    Tensor dweight_3d_temp;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           filter_shape, &dweight_3d_temp));
    T *dweight_3d_temp_ptr = dweight_3d_temp.template flat<T>().data();
    SetZeros<Device, T>()(d, group_ * M * N, col_buffer_3d_ptr);
    SetZeros<Device, T>()(d, ProdShape(x_shape, 0, x_shape.dims()), x_grad_ptr);
    SetZeros<Device, T>()(d, ProdShape(filter_shape, 0, filter_shape.dims()),
                          dweight_3d_ptr);
    SetZeros<Device, T>()(d, ProdShape(filter_shape, 0, filter_shape.dims()),
                          dweight_3d_temp_ptr);
    for (int n = 0; n < num_ / im2col_step_; ++n) {
      TensorShape out_grad_3d_shape = TensorShape({group_, K, N});
      T *out_grad_3d_ptr = out_grad_4d_ptr + n * group_ * K * N;
      LaunchBatchMatMul<Device, T>::launch(
          ctx, weight_3d_shape, out_grad_3d_shape, weight_3d_ptr,
          out_grad_3d_ptr, true, false, col_buffer_3d_ptr);
      DeformableConv2DCol2ImCoord<Device, T>()(
          d, col_buffer_3d_ptr, x_ptr + n * im2col_step_ * input_dim_,
          offset_ptr + n * im2col_step_ * input_offset_dim_,
          mask_ptr + n * im2col_step_ * input_mask_dim_, ToVector(x_shape),
          ToVector(col_buffer_shape), kernel_shape, pads, stride_shape,
          dilation_shape, params_.deformable_groups,
          offset_grad_ptr + n * im2col_step_ * input_offset_dim_,
          mask_grad_ptr + n * im2col_step_ * input_mask_dim_);
      DeformableConv2DCol2Im<Device, T>()(
          d, col_buffer_3d_ptr,
          offset_ptr + n * im2col_step_ * input_offset_dim_,
          mask_ptr + n * im2col_step_ * input_mask_dim_, ToVector(x_shape),
          ToVector(col_buffer_shape), kernel_shape, pads, stride_shape,
          dilation_shape, params_.deformable_groups,
          x_grad_ptr + n * im2col_step_ * input_dim_);
      DeformableConv2DIm2Col<Device, T>()(
          d, x_ptr + n * im2col_step_ * input_dim_,
          offset_ptr + n * im2col_step_ * input_offset_dim_,
          mask_ptr + n * im2col_step_ * input_mask_dim_, ToVector(x_shape),
          ToVector(col_buffer_shape), kernel_shape, pads, stride_shape,
          dilation_shape, params_.deformable_groups, col_buffer_3d_ptr);
      if (n == 0) {
        LaunchBatchMatMul<Device, T>::launch(
            ctx, out_grad_3d_shape, col_buffer_3d_shape, out_grad_3d_ptr,
            col_buffer_3d_ptr, false, true, dweight_3d_ptr);
      } else {
        LaunchBatchMatMul<Device, T>::launch(
            ctx, out_grad_3d_shape, col_buffer_3d_shape, out_grad_3d_ptr,
            col_buffer_3d_ptr, false, true, dweight_3d_temp_ptr);
        PureAddTo<Device, T>()(d,
                               ProdShape(filter_shape, 0, filter_shape.dims()),
                               dweight_3d_ptr, dweight_3d_temp_ptr);
      }
    }
  }

 private:
  DeformableConv2DParameters params_;
  // bool use_cudnn_;
  // bool cudnn_use_autotune_;
  int32_t channel_axis_;       // channel axis of the input
  int32_t channels_;           // number of channels of input image
  int32_t num_spatial_axes_;   // number of spatial axes
  int32_t num_;                // batch size
  int32_t group_;              // number of groups
  int32_t conv_out_channels_;  // number of output channels (num_filter)
  int32_t
      conv_out_spatial_dim_;  // number of pixels of output images per channel
  int32_t conv_in_channels_;  // number of input channels
  int32_t kernel_dim_;     // number of input channels per group * kernel size
  int32_t weight_offset_;  // number of output channels per group * kernel_dim_
  int32_t col_offset_;
  int32_t output_offset_;
  int32_t col_buffer_size_;
  int32_t input_dim_;
  int32_t input_offset_dim_;
  int32_t input_mask_dim_;
  int32_t output_dim_;
  int32_t num_kernels_im2col_;
  int32_t num_kernels_col2im_;
  int32_t im2col_step_;
  bool bias_term_;  // has bias term?
  bool is_1x1_;
  void LayerSetUp(const TensorShape &ishape, const TensorShape &filter_shape,
                  const TensorShape &offset_shape,
                  const TensorShape &mask_shape, const TensorShape &oshape) {
    channel_axis_ = 1;  // hard code channel axis, fixed the input data_format
    const int32_t first_spatial_axis = channel_axis_ + 1;
    const int32_t num_axes = filter_shape.dims();
    num_spatial_axes_ =
        num_axes -
        first_spatial_axis;  //表示的是空间坐标个数,比如说2维卷积里,就是2,
                             // 3维卷积里就是3
    is_1x1_ = true;  //  判断是否为1x1卷积
    for (int32_t i = 2; i < filter_shape.dims(); ++i) {
      // is_1x1_ &= filter_shape.dim_size(i) == 1 && params_.stride[i] == 1 &&
      // params_.pad[i] == 0;
      is_1x1_ &=
          filter_shape.dim_size(i) == 1;  // only judge by the filter's shape
      if (!is_1x1_) break;
    }
    num_ = ishape.dim_size(0);                      // batch size
    channels_ = ishape.dim_size(1);                 // number of input channels
    group_ = params_.num_groups;                    //
    conv_out_channels_ = filter_shape.dim_size(0);  // output channel nums
    conv_in_channels_ = channels_;                  // input channel nums
    bias_term_ = !params_.no_bias;                  //
    kernel_dim_ =
        conv_in_channels_ / group_ * filter_shape.dim_size(2) *
        filter_shape.dim_size(
            3);  // Size()返回tensor中元素个数，即各维度大小的乘积，所以这里的kernel_dim的意思是卷积核的参数个数了．
    conv_out_spatial_dim_ = ProdShape(
        oshape, 2,
        oshape
            .dims());  // ProdShape(dimstart, dimend)返回指定维度大小乘积,
                       // 这个变量代表每个通道的像素点个数,
                       // oshape.ndim()返回这个shape的维度，假设是NCHW那么返回4,则为
                       // H * W，
    col_offset_ =
        kernel_dim_ *
        conv_out_spatial_dim_;  // kernel_dim代表一个卷积核参数的个数，conv_out_spatial_dim_相当于特征图上的坐标个数，那这个变量相当于总共需要的偏移量
    weight_offset_ =
        conv_out_channels_ * kernel_dim_ /
        group_;  //这里应该是所有的权重的个数，也就是需要求的权重偏移的个数
    output_offset_ =
        conv_out_channels_ * conv_out_spatial_dim_ /
        group_;  //这里是输出通道数乘上每个通道的像素点的个数，所以结果应该是输出的总维度，就是C*H*W
    im2col_step_ = std::min(params_.im2col_step, num_);
    col_buffer_size_ =
        kernel_dim_ * group_ * im2col_step_ *
        conv_out_spatial_dim_;  // 开辟的缓存大小// size of the column buffer
                                // used for storing im2col-ed pixels

    input_dim_ = ProdShape(
        ishape, 1,
        ishape.dims());  // input image size (#channels * height * width)
    input_offset_dim_ =
        ProdShape(offset_shape, 1, offset_shape.dims());  // 18 * H * W
    input_mask_dim_ = ProdShape(mask_shape, 1, mask_shape.dims());  // 9 * H * W
    output_dim_ = ProdShape(oshape, 1, oshape.dims());  //输出的元素个数

    num_kernels_im2col_ =
        conv_in_channels_ *
        conv_out_spatial_dim_;  //如果输出和输入的分辨率不变的话，代表输入数据的dim,我个人觉得就是把整个输入展开为一个一维向量,在求其维度大小
    num_kernels_col2im_ = input_dim_;  //输入数据的dim
  };
};

template <typename Device, typename Type>
class DeformablePSROIPoolOp : public OpKernel {
 public:
  explicit DeformablePSROIPoolOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pooled_size", &pool_size));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("no_trans", &no_trans));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("spatial_scale", &spatial_scale));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_dim", &output_dim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("group_size", &group_size));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("part_size", &part_size));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sample_per_part", &sample_per_part));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("trans_std", &trans_std));
  }
  void Compute(OpKernelContext *ctx) override {
    const Tensor &data = ctx->input(0);
    const Tensor &bbox = ctx->input(1);
    const Tensor &trans = ctx->input(2);
    const int batch = data.dim_size(0);
    const int channels = data.dim_size(1);
    const int height = data.dim_size(2);
    const int width = data.dim_size(3);
    const int channels_trans = no_trans ? 2 : trans.dim_size(1);
    const int num_bbox = bbox.dim_size(0);
    Tensor *output;
    Tensor *top_count;
    const int pooled_width = pool_size;
    const int pooled_height = pool_size;
    const int count = num_bbox * output_dim * pooled_height * pooled_width;
    const int num_classes = no_trans ? 1 : channels_trans / 2;
    const int channels_each_class =
        no_trans ? output_dim : output_dim / num_classes;
    TensorShape output_shape{num_bbox, output_dim, pooled_height, pooled_width};
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, output_shape, &top_count));
    const Type *bottom_data = data.flat<Type>().data();
    const Type *bottom_rois = bbox.flat<Type>().data();
    const Type *bottom_trans = no_trans ? nullptr : trans.flat<Type>().data();
    Type *top_data = output->flat<Type>().data();
    Type *top_count_data = top_count->flat<Type>().data();
    const Device &d = ctx->eigen_device<Device>();
    DeformablePSROIPoolForward<Device, Type>()(
        d, count, bottom_data, spatial_scale, channels, height, width,
        pooled_height, pooled_width, bottom_rois, bottom_trans, no_trans,
        trans_std, sample_per_part, output_dim, group_size, part_size,
        num_classes, channels_each_class, top_data, top_count_data);
  }

 private:
  int pool_size;
  int no_trans;
  float spatial_scale;
  int output_dim;
  int group_size;
  int part_size;
  int sample_per_part;
  float trans_std;
};

template <typename Device, typename Type>
class DeformablePSROIPoolBackPropOp : public OpKernel {
 public:
  explicit DeformablePSROIPoolBackPropOp(OpKernelConstruction *ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("pooled_size", &pool_size));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("no_trans", &no_trans));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("spatial_scale", &spatial_scale));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_dim", &output_dim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("group_size", &group_size));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("part_size", &part_size));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sample_per_part", &sample_per_part));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("trans_std", &trans_std));
  }
  void Compute(OpKernelContext *ctx) override {
    const Tensor &data = ctx->input(0);
    const Tensor &bbox = ctx->input(1);
    const Tensor &trans = ctx->input(2);
    const Tensor &top_count = ctx->input(3);
    const Tensor &out_grad = ctx->input(4);
    const int batch = data.dim_size(0);
    const int channels = data.dim_size(1);
    const int height = data.dim_size(2);
    const int width = data.dim_size(3);
    const int channels_trans = no_trans ? 2 : trans.dim_size(1);
    const int num_bbox = bbox.dim_size(0);
    const int num_rois = num_bbox;
    const int pooled_height = pool_size;
    const int pooled_width = pool_size;
    const int count = num_bbox * output_dim * pooled_height * pooled_width;
    const int num_classes = no_trans ? 1 : channels_trans / 2;
    const int channels_each_class =
        no_trans ? output_dim : output_dim / num_classes;
    Tensor *in_grad = nullptr;
    Tensor *trans_grad = nullptr;
    const TensorShape &in_grad_shape = data.shape();
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in_grad_shape, &in_grad));
    TensorShape trans_grad_shape;
    const Type *top_diff = out_grad.flat<Type>().data();
    const Type *bottom_data = data.flat<Type>().data();
    const Type *bottom_rois = bbox.flat<Type>().data();
    trans_grad_shape = trans.shape();
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, trans_grad_shape, &trans_grad));
    const Type *bottom_trans = no_trans ? nullptr : trans.flat<Type>().data();
    Type *bottom_data_diff = in_grad->flat<Type>().data();
    Type *bottom_trans_diff =
        no_trans ? nullptr : trans_grad->flat<Type>().data();
    const Type *top_count_data = top_count.flat<Type>().data();
    const Device &d = ctx->eigen_device<Device>();
    DeformablePSROIPoolBackwardKernel<Device, Type>()(
        d, count, top_diff, top_count_data, num_rois, spatial_scale, channels,
        height, width, pooled_height, pooled_width, output_dim,
        bottom_data_diff, bottom_trans_diff, bottom_data, bottom_rois,
        bottom_trans, no_trans, trans_std, sample_per_part, group_size,
        part_size, num_classes, channels_each_class);
  }

 private:
  int pool_size;
  int no_trans;
  float spatial_scale;
  int output_dim;
  int group_size;
  int part_size;
  int sample_per_part;
  float trans_std;
};

#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(Name("AddonsDeformableConv2D")         \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<T>("T"),           \
                          DeformableConv2DOp<CPUDevice, T>);     \
  REGISTER_KERNEL_BUILDER(Name("AddonsDeformableConv2DBackProp") \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<T>("T"),           \
                          DeformableConv2DBackPropOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
#undef REGISTER_CPU
#define REGISTER_CPU(T)                                             \
  REGISTER_KERNEL_BUILDER(Name("AddonsDeformablePsroiPool")         \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T"),              \
                          DeformablePSROIPoolOp<CPUDevice, T>);     \
  REGISTER_KERNEL_BUILDER(Name("AddonsDeformablePsroiPoolBackProp") \
                              .Device(DEVICE_CPU)                   \
                              .TypeConstraint<T>("T"),              \
                          DeformablePSROIPoolBackPropOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
#undef REGISTER_CPU

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(Name("AddonsDeformableConv2D")         \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<T>("T"),           \
                          DeformableConv2DOp<GPUDevice, T>);     \
  REGISTER_KERNEL_BUILDER(Name("AddonsDeformableConv2DBackProp") \
                              .Device(DEVICE_GPU)                \
                              .TypeConstraint<T>("T"),           \
                          DeformableConv2DBackPropOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#undef REGISTER_GPU
#define REGISTER_GPU(T)                                             \
  REGISTER_KERNEL_BUILDER(Name("AddonsDeformablePsroiPool")         \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T"),              \
                          DeformablePSROIPoolOp<GPUDevice, T>);     \
  REGISTER_KERNEL_BUILDER(Name("AddonsDeformablePsroiPoolBackProp") \
                              .Device(DEVICE_GPU)                   \
                              .TypeConstraint<T>("T"),              \
                          DeformablePSROIPoolBackPropOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow
