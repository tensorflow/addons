// Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace addons {

using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

// --------------------------------------------------------------------------

REGISTER_OP("Addons>CorrelationCost")
    .Input("input_a: T")
    .Input("input_b: T")
    .Output("output: T")
    .Attr("kernel_size: int")
    .Attr("max_displacement: int")
    .Attr("stride_1: int")
    .Attr("stride_2: int")
    .Attr("pad: int")
    .Attr("data_format: {'NHWC', 'NCHW'} = 'NHWC'")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input_a, input_b, input;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_a));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &input_b));
      TF_RETURN_IF_ERROR(c->Merge(input_a, input_b, &input));

      // get input shapes
      int32 B, H, W;
      B = c->Value(c->Dim(input, 0));
      string data_format;
      Status s = c->GetAttr("data_format", &data_format);
      if (s.ok() && data_format == "NCHW") {
        H = c->Value(c->Dim(input, 2));
        W = c->Value(c->Dim(input, 3));
      } else {
        H = c->Value(c->Dim(input, 1));
        W = c->Value(c->Dim(input, 2));
      }

      int32 kernel_size;
      int32 max_displacement;
      int32 stride_1;
      int32 stride_2;
      int32 pad;

      TF_RETURN_IF_ERROR(c->GetAttr("kernel_size", &kernel_size));
      TF_RETURN_IF_ERROR(c->GetAttr("max_displacement", &max_displacement));
      // stride in input
      TF_RETURN_IF_ERROR(c->GetAttr("stride_1", &stride_1));
      // stride in patch
      TF_RETURN_IF_ERROR(c->GetAttr("stride_2", &stride_2));
      TF_RETURN_IF_ERROR(c->GetAttr("pad", &pad));

      // output channels are d**2 where, d = 2r + 1
      const int32 r = max_displacement / stride_2;
      const int32 d = 2 * r + 1;
      const int32 border = max_displacement + (kernel_size - 1) / 2;

      const int32 Cout = d * d;
      // for spatial dimensions, we pad the inputs
      const int32 Hout = static_cast<int>(
          ceil(static_cast<float>(((H + 2 * pad) - border * 2)) /
               static_cast<float>(stride_1)));
      const int32 Wout = static_cast<int>(
          ceil(static_cast<float>(((W + 2 * pad) - border * 2)) /
               static_cast<float>(stride_1)));

      // Note, the output is always NCHW (even when input is NHWC)
      c->set_output(0, c->MakeShape({B, Cout, Hout, Wout}));
      return Status();
    })
    .Doc(R"Doc(
Compute Correlation costs.

This layer implements the correlation operation from
FlowNet Learning Optical Flow with Convolutional Networks (Fischer et al.)

input_a: A `Tensor` of the format specified by `data_format`.
input_b: A `Tensor` of the format specified by `data_format`.
kernel_size: An integer specifying the height and width of the
    patch used to compute the per-patch costs.
max_displacement: An integer specifying the maximum search radius
    for each position.
stride_1: An integer specifying the stride length in the input.
stride_2: An integer specifying the stride length in the patch.
pad: An integer specifying the paddings in height and width.
data_format: Specifies the data format.
    Possible values are:
    "NHWC" float [batch, height, width, channels]
    "NCHW" float [batch, channels, height, width]
    Defaults to `"NHWC"`.
)Doc");

REGISTER_OP("Addons>CorrelationCostGrad")
    .Input("orig_input_a: T")
    .Input("orig_input_b: T")
    .Input("top_diff: T")
    .Output("bottom_diff_a: T")
    .Output("bottom_diff_b: T")
    .Attr("T: realnumbertype")
    .Attr("kernel_size: int")
    .Attr("max_displacement: int")
    .Attr("stride_1: int")
    .Attr("stride_2: int")
    .Attr("pad: int")
    .Attr("data_format: {'NHWC', 'NCHW'} = 'NHWC'")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle shp_hnd;
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &shp_hnd));
      c->set_output(0, shp_hnd);
      c->set_output(1, shp_hnd);
      return Status();
    })
    .Doc(R"doc(CorrelationCostGrad op.)doc");

}  // namespace addons
}  // namespace tensorflow