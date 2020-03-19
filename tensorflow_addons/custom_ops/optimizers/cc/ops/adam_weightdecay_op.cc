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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace addons {

using shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

static ShapeHandle ShapeOrHandleShape(InferenceContext* c, int input) {
  auto* handle_data = c->input_handle_shapes_and_types(input);
  if (handle_data != nullptr && !handle_data->empty() &&
      (*handle_data)[0].dtype != DT_INVALID) {
    return (*handle_data)[0].shape;
  }
  return c->input(input);
}

// Handle the gradient and, if <sparse>, indices inputs.
// <s> is an input+output parameter, containing the current known input shape to
// the gradient.
static Status HandleGradAndIndicesInputs(InferenceContext* c, bool sparse,
                                         int grad_idx, ShapeHandle* s) {
  ShapeHandle grad = ShapeOrHandleShape(c, grad_idx);
  if (!sparse) {
    TF_RETURN_IF_ERROR(c->Merge(*s, grad, s));
    return Status::OK();
  }
  // Indices is a vector where indices.dim[0].rank == grad[0].rank.
  ShapeHandle indices;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(grad_idx + 1), 1, &indices));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(indices, 0), c->Dim(grad, 0), &unused));

  // Trailing part of grad matches trailing part of *s.
  ShapeHandle grad_unknown_first;
  TF_RETURN_IF_ERROR(
      c->ReplaceDim(grad, 0, c->UnknownDim(), &grad_unknown_first));
  TF_RETURN_IF_ERROR(c->Merge(*s, grad_unknown_first, s));

  return Status::OK();
}

static Status ApplyAdShapeFn(InferenceContext* c, bool sparse) {
  ShapeHandle unused;
  ShapeHandle s = ShapeOrHandleShape(c, 0);                       // var
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 1), &s));  // m
  TF_RETURN_IF_ERROR(c->Merge(s, ShapeOrHandleShape(c, 2), &s));  // v
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));       // beta1_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));       // beta2_power
  TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));       // wd
  TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));       // beta1
  TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));       // beta2
  TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));       // epsilon
  TF_RETURN_IF_ERROR(
      HandleGradAndIndicesInputs(c, sparse, 9 /* grad_idx */, &s));
  if (c->num_outputs() > 0) {
    c->set_output(0, s);
  }
  return Status::OK();
}

REGISTER_OP("Addons>ApplyAdamWithWeightdecay")
    .Input("var: T")
    .Input("m: T")
    .Input("v: T")
    .Input("beta1_power: T")
    .Input("beta2_power: T")
    .Input("wd: T")
    .Input("beta1: T")
    .Input("beta2: T")
    .Input("epsilon: T")
    .Input("grad: T")
    .Output("var_update: T")
    .Output("out_m: T")
    .Output("out_v: T")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .Attr("use_nesterov: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      return ApplyAdShapeFn(c, false /* sparse */);
    });
}  // namespace addons
}  // namespace tensorflow
