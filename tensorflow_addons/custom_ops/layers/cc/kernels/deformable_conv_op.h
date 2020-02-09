
#ifndef TF_OPS_DEFORMABLE_CONV2D_H
#define TF_OPS_DEFORMABLE_CONV2D_H

#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__

// This is slightly mental, but gets it to properly index device function calls
// like __popc and whatever.
//#define __CUDACC__

// These headers are all implicitly present when you compile CUDA with clang.
// Clion doesn't know that, so we include them explicitly to make the indexer
// happy. Doing this when you actually build is, obviously, a terrible idea :D
//#include <__clang_cuda_builtin_vars.h>
//#include <__clang_cuda_intrinsics.h>
//#include <__clang_cuda_math_forward_declares.h>
//#include <__clang_cuda_complex_builtins.h>
//#include <__clang_cuda_cmath.h>
#endif  // __JETBRAINS_IDE__

#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU

#include <iostream>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace addons {

namespace functor {

using TShape = std::vector<int>;

typedef Eigen::GpuDevice GPUDevice;
typedef Eigen::ThreadPoolDevice CPUDevice;

inline int ProdShape(const TShape &shape, int start, int end) {
  int res = 1;
  for (int i = start; i < end; ++i) {
    res *= shape[i];
  }
  return res;
}
inline int ProdShape(const TensorShape &shape, int start, int end) {
  int res = 1;
  for (int i = start; i < end; ++i) {
    res *= shape.dim_size(i);
  }
  return res;
}

template <typename Device, typename DType>
struct PureAddTo {
  void operator()(const Device &d, const int n, DType *result_data,
                  const DType *right_data);
};
struct DeformableConv2DParameters {
  TShape dilations;
  TShape strides;
  Padding padding;
  int32_t num_groups;
  int32_t deformable_groups;
  int32_t im2col_step;
  bool no_bias;
  TensorFormat data_format;
};
struct DeformableConv2DDimensions {
  int batch;
  int input_rows;
  int input_cols;
  int in_depth;
  int filter_rows;
  int filter_cols;
  int patch_depth;
  int out_depth;
  int stride_rows;
  int stride_cols;
  int dilation_rows;
  int dilation_cols;
  int out_rows;
  int out_cols;
  int pad_rows;
  int pad_cols;
};
template <typename Device, typename T>
struct LaunchBatchMatMul;

template <typename Device, typename DType>
struct DeformableConv2DCol2ImCoord {
  void operator()(const Device &d, const DType *data_col, const DType *data_im,
                  const DType *data_offset, const DType *data_mask,
                  const TShape &im_shape, const TShape &col_shape,
                  const TShape &kernel_shape, const TShape &pad,
                  const TShape &stride, const TShape &dilation,
                  const int32_t deformable_group, DType *grad_offset,
                  DType *grad_mask);
};
template <typename Device, typename DType>
struct SwapAxis {
  void operator()(const Device &d, DType *input_data,
                  const TShape &origin_shape, const int axis_x,
                  const int axis_y);
};
template <typename Device, typename DType>
struct DeformableConv2DCol2Im {
  void operator()(const Device &d, const DType *data_col,
                  const DType *data_offset, const DType *data_mask,
                  const TShape &im_shape, const TShape &col_shape,
                  const TShape &kernel_shape, const TShape &pad,
                  const TShape &stride, const TShape &dilation,
                  const int32_t deformable_group, DType *grad_im);
};
template <typename Device, typename DType>
struct DeformableConv2DIm2Col {
  void operator()(const Device &d, const DType *data_im,
                  const DType *data_offset, const DType *data_mask,
                  const TShape &im_shape, const TShape &col_shape,
                  const TShape &kernel_shape, const TShape &pad,
                  const TShape &stride, const TShape &dilation,
                  const int32_t deformable_group, DType *data_col);
};
template <typename Device, typename DType>
struct SetZeros {
  void operator()(const Device &d, int n, DType *result_data);
};
template <typename Device, typename DType>
struct SetOne {
  void operator()(const Device &d, int n, DType *result_data);
};
template <typename Device, typename DType>
struct SetNumAtIndex {
  void operator()(const Device &d, DType num, int index, DType *data);
};
#ifdef GOOGLE_CUDA
template <typename DType>
struct DeformableConv2DIm2Col<Eigen::GpuDevice, DType> {
  void operator()(const Eigen::GpuDevice &d, const DType *data_im,
                  const DType *data_offset, const DType *data_mask,
                  const TShape &im_shape, const TShape &col_shape,
                  const TShape &kernel_shape, const TShape &pad,
                  const TShape &stride, const TShape &dilation,
                  const int32_t deformable_group, DType *data_col);
};
template <typename DType>
struct DeformableConv2DCol2Im<Eigen::GpuDevice, DType> {
  void operator()(const Eigen::GpuDevice &d, const DType *data_col,
                  const DType *data_offset, const DType *data_mask,
                  const TShape &im_shape, const TShape &col_shape,
                  const TShape &kernel_shape, const TShape &pad,
                  const TShape &stride, const TShape &dilation,
                  const int32_t deformable_group, DType *grad_im);
};
template <typename DType>
struct DeformableConv2DCol2ImCoord<Eigen::GpuDevice, DType> {
  void operator()(const Eigen::GpuDevice &d, const DType *data_col,
                  const DType *data_im, const DType *data_offset,
                  const DType *data_mask, const TShape &im_shape,
                  const TShape &col_shape, const TShape &kernel_shape,
                  const TShape &pad, const TShape &stride,
                  const TShape &dilation, const int32_t deformable_group,
                  DType *grad_offset, DType *grad_mask);
};
template <typename DType>
struct SetNumAtIndex<Eigen::GpuDevice, DType> {
  void operator()(const Eigen::GpuDevice &d, DType num, int index, DType *data);
};
template <typename DType>
struct SetZeros<Eigen::GpuDevice, DType> {
  void operator()(const Eigen::GpuDevice &d, int n, DType *result_data);
};
template <typename DType>
struct SetOne<Eigen::GpuDevice, DType> {
  void operator()(const Eigen::GpuDevice &d, int n, DType *result_data);
};
template <typename DType>
struct PureAddTo<Eigen::GpuDevice, DType> {
  void operator()(const Eigen::GpuDevice &d, const int n, DType *result_data,
                  const DType *right_data);
};
#endif
template <typename DType>
struct DeformableConv2DIm2Col<Eigen::ThreadPoolDevice, DType> {
  void operator()(const Eigen::ThreadPoolDevice &d, const DType *data_im,
                  const DType *data_offset, const DType *data_mask,
                  const TShape &im_shape, const TShape &col_shape,
                  const TShape &kernel_shape, const TShape &pad,
                  const TShape &stride, const TShape &dilation,
                  const int32_t deformable_group, DType *data_col);
};
template <typename DType>
struct DeformableConv2DCol2Im<Eigen::ThreadPoolDevice, DType> {
  void operator()(const Eigen::ThreadPoolDevice &d, const DType *data_col,
                  const DType *data_offset, const DType *data_mask,
                  const TShape &im_shape, const TShape &col_shape,
                  const TShape &kernel_shape, const TShape &pad,
                  const TShape &stride, const TShape &dilation,
                  const int32_t deformable_group, DType *grad_im);
};
template <typename DType>
struct DeformableConv2DCol2ImCoord<Eigen::ThreadPoolDevice, DType> {
  void operator()(const Eigen::ThreadPoolDevice &d, const DType *data_col,
                  const DType *data_im, const DType *data_offset,
                  const DType *data_mask, const TShape &im_shape,
                  const TShape &col_shape, const TShape &kernel_shape,
                  const TShape &pad, const TShape &stride,
                  const TShape &dilation, const int32_t deformable_group,
                  DType *grad_offset, DType *grad_mask);
};
template <typename DType>
struct SetNumAtIndex<Eigen::ThreadPoolDevice, DType> {
  void operator()(const Eigen::ThreadPoolDevice &d, DType num, int index,
                  DType *data);
};
template <typename DType>
struct SetZeros<Eigen::ThreadPoolDevice, DType> {
  void operator()(const Eigen::ThreadPoolDevice &d, int n, DType *result_data);
};
template <typename DType>
struct SetOne<Eigen::ThreadPoolDevice, DType> {
  void operator()(const Eigen::ThreadPoolDevice &d, int n, DType *result_data);
};
template <typename DType>
struct PureAddTo<Eigen::ThreadPoolDevice, DType> {
  void operator()(const Eigen::ThreadPoolDevice &d, const int n,
                  DType *result_data, const DType *right_data);
};

template <typename T>
struct LaunchBatchMatMul<GPUDevice, T> {
  static void launch(OpKernelContext *context, const TensorShape &in_x_shape,
                     const TensorShape &in_y_shape, const T *in_x_ptr,
                     const T *in_y_ptr, bool adj_x, bool adj_y, T *out);
};
template <typename T>
struct LaunchBatchMatMul<CPUDevice, T> {
  static void launch(OpKernelContext *context, const TensorShape &in_x_shape,
                     const TensorShape &in_y_shape, const T *in_x_ptr,
                     const T *in_y_ptr, bool adj_x, bool adj_y, T *out);
};

template <typename Device, typename T>
struct DeformablePSROIPoolForward {};
template <typename T>
struct DeformablePSROIPoolForward<CPUDevice, T> {
  void operator()(const CPUDevice &d, const int count, const T *bottom_data,
                  const T spatial_scale, const int channels, const int height,
                  const int width, const int pooled_height,
                  const int pooled_width, const T *bottom_rois,
                  const T *bottom_trans, const int no_trans, const T trans_std,
                  const int sample_per_part, const int output_dim,
                  const int group_size, const int part_size,
                  const int num_classes, const int channels_each_class,
                  T *top_data, T *top_count);
};
template <typename T>
struct DeformablePSROIPoolForward<GPUDevice, T> {
  void operator()(const GPUDevice &d, const int count, const T *bottom_data,
                  const T spatial_scale, const int channels, const int height,
                  const int width, const int pooled_height,
                  const int pooled_width, const T *bottom_rois,
                  const T *bottom_trans, const int no_trans, const T trans_std,
                  const int sample_per_part, const int output_dim,
                  const int group_size, const int part_size,
                  const int num_classes, const int channels_each_class,
                  T *top_data, T *top_count);
};

template <typename Device, typename T>
struct DeformablePSROIPoolBackwardKernel {};

template <typename T>
struct DeformablePSROIPoolBackwardKernel<GPUDevice, T> {
  void operator()(const GPUDevice &d, const int count, const T *top_diff,
                  const T *top_count, const int num_rois, const T spatial_scale,
                  const int channels, const int height, const int width,
                  const int pooled_height, const int pooled_width,
                  const int output_dim, T *bottom_data_diff,
                  T *bottom_trans_diff, const T *bottom_data,
                  const T *bottom_rois, const T *bottom_trans,
                  const int no_trans, const T trans_std,
                  const int sample_per_part, const int group_size,
                  const int part_size, const int num_classes,
                  const int channels_each_class);
};

template <typename T>
struct DeformablePSROIPoolBackwardKernel<CPUDevice, T> {
  void operator()(const CPUDevice &d, const int count, const T *top_diff,
                  const T *top_count, const int num_rois, const T spatial_scale,
                  const int channels, const int height, const int width,
                  const int pooled_height, const int pooled_width,
                  const int output_dim, T *bottom_data_diff,
                  T *bottom_trans_diff, const T *bottom_data,
                  const T *bottom_rois, const T *bottom_trans,
                  const int no_trans, const T trans_std,
                  const int sample_per_part, const int group_size,
                  const int part_size, const int num_classes,
                  const int channels_each_class);
};
}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif  // TF_OPS_DEFORMABLE_CONV2D_H
