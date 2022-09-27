/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {
namespace addons {

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("Addons>EmbeddingBag")
    .Input("indices: Tindices")
    .Input("params: T")
    .Input("weights: T")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tindices: {int32, int64}")
    .Attr("combiner: {'SUM', 'MEAN'} = 'SUM'")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle indices, params, weights, unused, output;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &params));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &weights));
      DimensionHandle output_dim = c->Dim(params, 1);
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(indices, c->Rank(indices) - 1, output_dim, &output));
      TF_RETURN_IF_ERROR(c->Merge(indices, weights, &unused));
      c->set_output(0, output);
      return Status();
    });

REGISTER_OP("Addons>EmbeddingBagGrad")
    .Input("indices: Tindices")
    .Input("params: T")
    .Input("weights: T")
    .Input("grads: T")
    .Output("params_grads: T")
    .Output("weights_grads: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tindices: {int32, int64}")
    .Attr("combiner: {'SUM', 'MEAN'} = 'SUM'")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle indices, params, weights, unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &params));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &weights));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &unused));
      TF_RETURN_IF_ERROR(c->Merge(indices, weights, &unused));
      c->set_output(0, c->input(1));
      c->set_output(1, c->input(2));
      return Status();
    });

}  // namespace addons
}  // namespace tensorflow
