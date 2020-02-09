//
// Created by 孙嘉禾 on 2019/12/31.
//

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace addons {

namespace functor {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("AddonsDeformableConv2D")
    .Input("input: T")
    .Input("filter: T")
    .Input("offset: T")
    .Input("mask: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("strides: list(int)")
    // .Attr("use_cudnn_on_gpu: bool = true")
    .Attr("num_groups: int")
    .Attr("deformable_groups: int")
    .Attr("im2col_step: int")
    .Attr("no_bias: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("data_format: {'NCHW' } = 'NCHW' ")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext *c) {
      ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
      ShapeHandle filter_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &filter_shape));
      ShapeHandle offset_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &offset_shape));
      ShapeHandle mask_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &mask_shape));

      std::vector<int32> strides;
      TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
      if (strides.size() != 4) {
        return errors::InvalidArgument(
            "Deformconv requires the stride attribute to contain 4 values, but "
            "got: ",
            strides.size());
      }

      std::vector<int32> rates;
      TF_RETURN_IF_ERROR(c->GetAttr("dilations", &rates));
      if (rates.size() != 4) {
        return errors::InvalidArgument(
            "Deformconv requires the dilations attribute to contain 4 values, "
            "but "
            "got: ",
            rates.size());
      }
      string data_format;
      TensorFormat data_format_;
      TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format));
      FormatFromString(data_format, &data_format_);
      const int32 stride_rows = GetTensorDim(strides, data_format_, 'H');
      const int32 stride_cols = GetTensorDim(strides, data_format_, 'W');

      const int32 rate_rows = GetTensorDim(rates, data_format_, 'H');
      const int32 rate_cols = GetTensorDim(rates, data_format_, 'W');

      int groups;
      TF_RETURN_IF_ERROR(c->GetAttr("num_groups", &groups));
      int deform_groups;
      TF_RETURN_IF_ERROR(c->GetAttr("deformable_groups", &deform_groups));

      DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
      DimensionHandle in_depths_dim = c->Dim(input_shape, 1);
      DimensionHandle in_rows_dim = c->Dim(input_shape, 2);
      DimensionHandle in_cols_dim = c->Dim(input_shape, 3);
      DimensionHandle filter_rows_dim = c->Dim(filter_shape, 2);
      DimensionHandle filter_cols_dim = c->Dim(filter_shape, 3);
      DimensionHandle filter_depth_dim = c->Dim(filter_shape, 1);
      DimensionHandle output_depth_dim = c->Dim(filter_shape, 0);
      DimensionHandle multiplied_depth;
      DimensionHandle depth_per_dfgps;
      auto filter_row = c->Value(filter_rows_dim);
      auto filter_col = c->Value(filter_cols_dim);
      auto offset_dpt = c->Value(c->Dim(offset_shape, 1));
      if ((offset_dpt % (filter_row * filter_col) != 0) ||
          (offset_dpt / (2 * filter_row * filter_col) != deform_groups)) {
        return errors::InvalidArgument(
            "Deformconv requires the offset compatible with filter, but "
            "got: ",
            c->DebugString(offset_shape));
      }

      auto mask_dpt = c->Value(c->Dim(mask_shape, 1));
      if ((mask_dpt % (filter_row * filter_col) != 0) ||
          (mask_dpt / (filter_row * filter_col) != deform_groups)) {
        return errors::InvalidArgument(
            "Deformconv requires the mask compatible with filter, but "
            "got: ",
            c->DebugString(offset_shape));
      }

      TF_RETURN_IF_ERROR(
          c->Multiply(filter_depth_dim, groups, &multiplied_depth));
      TF_RETURN_IF_ERROR(
          c->Divide(filter_depth_dim, deform_groups, true, &depth_per_dfgps));
      TF_RETURN_IF_ERROR(
          c->Divide(in_depths_dim, deform_groups, true, &depth_per_dfgps));

      if (!c->ValueKnown(in_rows_dim) || !c->ValueKnown(in_cols_dim) ||
          !c->ValueKnown(filter_rows_dim) || !c->ValueKnown(filter_cols_dim)) {
        ShapeHandle output_shape = c->MakeShape(
            {batch_size_dim, output_depth_dim, InferenceContext::kUnknownDim,
             InferenceContext::kUnknownDim});
        c->set_output(0, output_shape);
        return Status::OK();
      }
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(input_shape, 1), multiplied_depth, &unused));

      auto in_rows = c->Value(in_rows_dim);
      auto in_cols = c->Value(in_cols_dim);
      auto filter_rows = c->Value(filter_rows_dim);
      auto filter_cols = c->Value(filter_cols_dim);
      auto filter_rows_eff = filter_rows + (filter_rows - 1) * (rate_rows - 1);
      auto filter_cols_eff = filter_cols + (filter_cols - 1) * (rate_cols - 1);

      Padding padding;
      TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

      int64 output_rows, output_cols;
      int64 padding_before, padding_after;
      TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
          in_rows, filter_rows_eff, stride_rows, padding, &output_rows,
          &padding_before, &padding_after));
      TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
          in_cols, filter_cols_eff, stride_cols, padding, &output_cols,
          &padding_before, &padding_after));

      ShapeHandle output_shape = c->MakeShape(
          {batch_size_dim, output_depth_dim, output_rows, output_cols});
      c->set_output(0, output_shape);
      // shape_inference::ShapeHandle offset_shape = c->input(2);
      // shape_inference::ShapeHandle mask_shape = c->input(3);
      shape_inference::DimensionHandle offset_batch = c->Dim(offset_shape, 0);
      shape_inference::DimensionHandle offset_channel = c->Dim(offset_shape, 1);
      shape_inference::DimensionHandle offset_height = c->Dim(offset_shape, 2);
      shape_inference::DimensionHandle offset_weight = c->Dim(offset_shape, 3);
      shape_inference::DimensionHandle mask_channel = c->Dim(mask_shape, 1);
      shape_inference::DimensionHandle mask_height = c->Dim(mask_shape, 2);
      shape_inference::DimensionHandle mask_weight = c->Dim(mask_shape, 3);
      shape_inference::DimensionHandle mask_batch = c->Dim(mask_shape, 0);
      TF_RETURN_IF_ERROR(c->WithRank(offset_shape, 4, &offset_shape));
      TF_RETURN_IF_ERROR(c->WithRank(mask_shape, 4, &mask_shape));
      TF_RETURN_IF_ERROR(
          c->WithValue(offset_batch, c->Value(batch_size_dim), &offset_batch));
      TF_RETURN_IF_ERROR(c->WithValue(
          offset_channel,
          2 * c->Value(filter_rows_dim) * c->Value(filter_cols_dim),
          &offset_channel));
      TF_RETURN_IF_ERROR(
          c->WithValue(offset_height, output_rows, &offset_height));
      TF_RETURN_IF_ERROR(
          c->WithValue(offset_weight, output_cols, &offset_weight));
      TF_RETURN_IF_ERROR(
          c->WithValue(mask_batch, c->Value(batch_size_dim), &mask_batch));
      TF_RETURN_IF_ERROR(c->WithValue(
          mask_channel, c->Value(filter_rows_dim) * c->Value(filter_cols_dim),
          &mask_channel));
      TF_RETURN_IF_ERROR(c->WithValue(mask_height, output_rows, &mask_height));
      TF_RETURN_IF_ERROR(c->WithValue(mask_weight, output_cols, &mask_weight));
      return Status::OK();
    })
    .Doc(R"doc(
        DeformableConv2D is a new convolution operation with the deformable kernel locations.
        The inputs should have format NCHW, which is faster on GPUS.
        The offset and mask should have same input spatial resolution.
        Also, the output's shape depends on the stride, and I only consider the situation of dilation rate = 1.
    )doc");

// Opkernel defination.
// template parameter <T> is the datatype of the tensors
// in my opnion, the deformable convolution op ought to be implemented by
// extending the Conv2DOp, however, we can not get the conv_ops.h file if we
// choose to dynamic link the op

REGISTER_OP("AddonsDeformableConv2DBackProp")
    .Input("input: T")
    .Input("filter: T")
    .Input("offset: T")
    .Input("mask: T")
    .Input("out_grad: T")
    .Output("x_grad: T")
    .Output("filter_grad: T")
    .Output("offset_grad: T")
    .Output("mask_grad: T")
    .Attr("T: {float, double}")
    .Attr("strides: list(int)")
    // .Attr("use_cudnn_on_gpu: bool = true")
    .Attr("num_groups: int")
    .Attr("deformable_groups: int")
    .Attr("im2col_step: int")
    .Attr("no_bias: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("data_format: { 'NCHW' } = 'NCHW' ")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](InferenceContext *c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      c->set_output(2, c->input(2));
      c->set_output(3, c->input(3));
      return Status::OK();
    })
    .Doc(R"doc(only support NCHW now)doc");

REGISTER_OP("AddonsDeformablePsroiPool")
    .Input("input: T")
    .Input("bbox: T")
    .Input("trans: T")
    .Output("output: T")
    .Output("top_count: T")
    .Attr("T: {float, double}")
    .Attr("pooled_size: int")
    .Attr("no_trans: int")
    .Attr("spatial_scale: float")
    .Attr("output_dim: int")
    .Attr("group_size: int")
    .Attr("part_size: int")
    .Attr("sample_per_part: int")
    .Attr("trans_std: float")
    .SetShapeFn([](InferenceContext *ctx) {
      int pooled_size;
      int output_dim;
      TF_RETURN_IF_ERROR(ctx->GetAttr("pooled_size", &pooled_size));
      TF_RETURN_IF_ERROR(ctx->GetAttr("output_dim", &output_dim));
      auto input_handle = ctx->input(0);
      auto batch = ctx->Dim(input_handle, 0);
      auto output_dim_handle = ctx->MakeDim(output_dim);
      auto pooled_size_handle = ctx->MakeDim(pooled_size);
      ctx->set_output(
          0, ctx->MakeShape({batch, output_dim_handle, pooled_size_handle,
                             pooled_size_handle}));
      ctx->set_output(
          1, ctx->MakeShape({batch, output_dim_handle, pooled_size_handle,
                             pooled_size_handle}));
      return Status::OK();
    })
    .Doc(
        R"doc(DeformablePsROIPool is a new pooling operation with the deformable
kernel locations. The inpus should have format NCHW, which is faster on GPUS.)doc");
REGISTER_OP("AddonsDeformablePsroiPoolBackProp")
    .Input("data: T")
    .Input("bbox: T")
    .Input("trans: T")
    .Input("top_count: T")
    .Input("out_grad: T")
    .Output("in_grad: T")
    .Output("trans_grad: T")
    .Attr("pooled_size: int")
    .Attr("T: {float, double}")
    .Attr("no_trans: int")
    .Attr("spatial_scale: float")
    .Attr("output_dim: int")
    .Attr("group_size: int")
    .Attr("part_size: int")
    .Attr("sample_per_part: int")
    .Attr("trans_std: float")
    .SetShapeFn([](InferenceContext *ctx) {
      ctx->set_output(0, ctx->input(0));
      ctx->set_output(1, ctx->input(2));
      return Status::OK();
    })
    .Doc(R"doc("BackProp operation for DeformablePSROIPool")doc");

}  // namespace functor
}  // namespace addons
}  // namespace tensorflow
