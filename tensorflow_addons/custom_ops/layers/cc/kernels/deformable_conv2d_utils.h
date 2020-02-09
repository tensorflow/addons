
#ifndef TF_OPS_DEFORMABLE_CONV2D_UTILS_H
#define TF_OPS_DEFORMABLE_CONV2D_UTILS_H

#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_addons/custom_ops/layers/cc/kernels/deformable_conv_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/ThreadPool"

namespace tensorflow {
namespace addons {

namespace functor {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

using namespace tensorflow::shape_inference;
Status CheckFormatConstraintsOnShape(const TensorFormat tensor_format,
                                     const ShapeHandle shape_handle,
                                     const string &tensor_name,
                                     InferenceContext *ctx) {
  if (tensor_format == FORMAT_NCHW_VECT_C) {
    const int num_dims = ctx->Rank(shape_handle);
    DimensionHandle vect_dim = ctx->Dim(
        shape_handle, GetTensorInnerFeatureDimIndex(num_dims, tensor_format));
    DimensionHandle unused_vect_dim;
    TF_RETURN_IF_ERROR(ctx->WithValue(vect_dim, 4, &unused_vect_dim));
  }
  return Status::OK();
}
Status DimensionsFromShape(ShapeHandle shape, TensorFormat format,
                           DimensionHandle *batch_dim,
                           gtl::MutableArraySlice<DimensionHandle> spatial_dims,
                           DimensionHandle *filter_dim, InferenceContext *ctx) {
  const int32 rank = GetTensorDimsFromSpatialDims(spatial_dims.size(), format);
  *batch_dim = ctx->Dim(shape, GetTensorBatchDimIndex(rank, format));
  for (int spatial_dim_index = 0; spatial_dim_index < spatial_dims.size();
       ++spatial_dim_index) {
    spatial_dims[spatial_dim_index] = ctx->Dim(
        shape, GetTensorSpatialDimIndex(rank, format, spatial_dim_index));
  }
  *filter_dim = ctx->Dim(shape, GetTensorFeatureDimIndex(rank, format));
  if (format == FORMAT_NCHW_VECT_C) {
    TF_RETURN_IF_ERROR(ctx->Multiply(
        *filter_dim,
        ctx->Dim(shape, GetTensorInnerFeatureDimIndex(rank, format)),
        filter_dim));
  }
  return Status::OK();
}
Status ShapeFromDimensions(DimensionHandle batch_dim,
                           gtl::ArraySlice<DimensionHandle> spatial_dims,
                           DimensionHandle filter_dim, TensorFormat format,
                           InferenceContext *ctx, ShapeHandle *shape) {
  const int32 rank = GetTensorDimsFromSpatialDims(spatial_dims.size(), format);
  std::vector<DimensionHandle> out_dims(rank);
  out_dims[GetTensorBatchDimIndex(rank, format)] = batch_dim;
  for (int spatial_dim_index = 0; spatial_dim_index < spatial_dims.size();
       ++spatial_dim_index) {
    out_dims[GetTensorSpatialDimIndex(rank, format, spatial_dim_index)] =
        spatial_dims[spatial_dim_index];
  }
  if (format == FORMAT_NCHW_VECT_C) {
    TF_RETURN_IF_ERROR(
        ctx->Divide(filter_dim, 4, true,
                    &out_dims[GetTensorFeatureDimIndex(rank, format)]));
    out_dims[GetTensorInnerFeatureDimIndex(rank, format)] = ctx->MakeDim(4);
  } else {
    out_dims[GetTensorFeatureDimIndex(rank, format)] = filter_dim;
  }
  *shape = ctx->MakeShape(out_dims);
  return Status::OK();
}
template <typename Ta, typename Tb>
EIGEN_ALWAYS_INLINE EIGEN_DEVICE_FUNC bool FastBoundsCheck(const Ta index,
                                                           const Tb limit) {
  static_assert(std::is_integral<Ta>::value && std::is_integral<Tb>::value,
                "FastBoundsCheck can only be used on integer types.");
  typedef typename std::make_unsigned<decltype(index + limit)>::type UIndex;
  return TF_PREDICT_TRUE(static_cast<UIndex>(index) <
                         static_cast<UIndex>(limit));
}

#define TF_REQUIRES(EXP, STATUS)                \
  do {                                          \
    if (!TF_PREDICT_TRUE(EXP)) return (STATUS); \
  } while (false)

Status InitDeformableConv2DParameters(const OpKernelConstruction *ctx,
                                      DeformableConv2DParameters *params) {
  TF_RETURN_IF_ERROR(ctx->GetAttr("dilations", &params->dilations));
  TF_RETURN_IF_ERROR(ctx->GetAttr("strides", &params->strides));
  TF_RETURN_IF_ERROR(ctx->GetAttr("padding", &params->padding));
  string data_format_string;
  TF_RETURN_IF_ERROR(ctx->GetAttr("data_format", &data_format_string));
  TF_RETURN_IF_ERROR(ctx->GetAttr("num_groups", &params->num_groups));
  TF_RETURN_IF_ERROR(
      ctx->GetAttr("deformable_groups", &params->deformable_groups));
  TF_RETURN_IF_ERROR(ctx->GetAttr("im2col_step", &params->im2col_step));
  TF_RETURN_IF_ERROR(ctx->GetAttr("no_bias", &params->no_bias));
  TF_REQUIRES(FormatFromString(data_format_string, &params->data_format),
              errors::InvalidArgument("Invalid data format"));
  const auto &strides = params->strides;
  const auto &dilations = params->dilations;
  const auto &data_format = params->data_format;
  TF_REQUIRES(dilations.size() == 4,
              errors::InvalidArgument("Sliding window dilations field must "
                                      "specify 4 dimensions"));
  TF_REQUIRES(strides.size() == 4,
              errors::InvalidArgument("Sliding window strides field must "
                                      "specify 4 dimensions"));
  const int64 stride_n = GetTensorDim(strides, data_format, 'N');
  const int64 stride_c = GetTensorDim(strides, data_format, 'C');
  const int64 stride_h = GetTensorDim(strides, data_format, 'H');
  const int64 stride_w = GetTensorDim(strides, data_format, 'W');
  TF_REQUIRES(
      stride_n == 1 && stride_c == 1,
      errors::InvalidArgument("Current implementation does not yet support "
                              "strides in the batch and depth dimensions."));
  TF_REQUIRES(stride_h > 0 && stride_w > 0,
              errors::InvalidArgument(
                  "Row and column strides should be larger than 0."));

  const int64 dilation_n = GetTensorDim(dilations, data_format, 'N');
  const int64 dilation_c = GetTensorDim(dilations, data_format, 'C');
  const int64 dilation_h = GetTensorDim(dilations, data_format, 'H');
  const int64 dilation_w = GetTensorDim(dilations, data_format, 'W');
  TF_REQUIRES(
      dilation_n == 1 && dilation_c == 1,
      errors::InvalidArgument("Current implementation does not yet support "
                              "dilations in the batch and depth dimensions."));
  TF_REQUIRES(
      dilation_h > 0 && dilation_w > 0,
      errors::InvalidArgument("Dilated rates should be larger than 0."));

  return Status::OK();
}
Status ComputeDeformableConv2DDimension(
    const DeformableConv2DParameters &params, const Tensor &input,
    const Tensor &filter, DeformableConv2DDimensions *dimensions, int flag) {
  TF_REQUIRES(input.dims() == 4,
              errors::InvalidArgument("input must be 4-dimensional",
                                      input.shape().DebugString()));
  TF_REQUIRES(filter.dims() == 4,
              errors::InvalidArgument("filter must be 4-dimensional: ",
                                      filter.shape().DebugString()));
  for (int i = 3; i > 0; i--) {
    TF_REQUIRES(
        FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
        errors::InvalidArgument("filter too large"));
  }
  const int64 in_depth_raw = GetTensorDim(input, params.data_format, 'C');
  const int64 patch_depth_raw = filter.dim_size(1);
  TF_REQUIRES(FastBoundsCheck(in_depth_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input depth too large"));
  TF_REQUIRES(FastBoundsCheck(patch_depth_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Patch depth too large"));
  const int in_depth = static_cast<int>(in_depth_raw);
  const int patch_depth = static_cast<int>(patch_depth_raw);
  TF_REQUIRES(in_depth % patch_depth == 0,
              errors::InvalidArgument(
                  "input depth must be evenly divisible by filter depth: ",
                  in_depth, " vs ", patch_depth, " flag: ", flag));

  // The first dimension for filter is out_depth.
  const int out_depth = static_cast<int>(filter.dim_size(0));
  const int64 input_rows_raw = GetTensorDim(input, params.data_format, 'H');
  TF_REQUIRES(FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input rows too large"));
  const int input_rows = static_cast<int>(input_rows_raw);
  const int filter_rows = static_cast<int>(filter.dim_size(2));
  const int64 input_cols_raw = GetTensorDim(input, params.data_format, 'W');
  TF_REQUIRES(FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input cols too large"));
  const int input_cols = static_cast<int>(input_cols_raw);
  const int filter_cols = static_cast<int>(filter.dim_size(3));
  const int64 batch_raw = GetTensorDim(input, params.data_format, 'N');
  TF_REQUIRES(FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("batch is too large"));
  const int batch = static_cast<int>(batch_raw);
  const int stride_rows = GetTensorDim(params.strides, params.data_format, 'H');
  const int stride_cols = GetTensorDim(params.strides, params.data_format, 'W');
  const int dilation_rows =
      GetTensorDim(params.dilations, params.data_format, 'H');
  const int dilation_cols =
      GetTensorDim(params.dilations, params.data_format, 'W');

  // Compute windowed output sizes for rows and columns.
  int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeV2(
      input_rows, filter_rows, dilation_rows, stride_rows, params.padding,
      &out_rows, &pad_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeV2(
      input_cols, filter_cols, dilation_cols, stride_cols, params.padding,
      &out_cols, &pad_cols));

  dimensions->batch = batch;
  dimensions->input_rows = input_rows;
  dimensions->input_cols = input_cols;
  dimensions->in_depth = in_depth;
  dimensions->filter_rows = filter_rows;
  dimensions->filter_cols = filter_cols;
  dimensions->patch_depth = patch_depth;
  dimensions->out_depth = out_depth;
  dimensions->stride_rows = stride_rows;
  dimensions->stride_cols = stride_cols;
  dimensions->dilation_rows = dilation_rows;
  dimensions->dilation_cols = dilation_cols;
  dimensions->out_rows = out_rows;
  dimensions->out_cols = out_cols;
  dimensions->pad_rows = pad_rows;
  dimensions->pad_cols = pad_cols;
  return Status::OK();
}

inline TShape ToVector(const TShape &shape) { return shape; }

inline std::vector<int> ToVector(const TensorShape &shape) {
  std::vector<int> res;
  for (int i = 0; i < shape.dims(); ++i) {
    res.push_back(shape.dim_size(i));
  }
  return res;
}

inline std::vector<int> SubVector(const TensorShape &shape, int start,
                                  int end) {
  std::vector<int> res;
  for (int i = start; i < end; i++) {
    res.push_back(shape.dim_size(i));
  }
  return res;
}

inline TShape SubVector(const TShape &shape, int start, int end) {
  TShape res;
  for (int i = start; i < end; i++) {
    res.push_back(shape[i]);
  }
  return res;
}

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow

#endif  // TF_OPS_DEFORMABLE_CONV2D_UTILS_H
