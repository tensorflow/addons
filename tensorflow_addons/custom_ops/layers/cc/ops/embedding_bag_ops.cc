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
    .Input("values: T")
    .Input("weights: T")
    .Output("output: T")
    .Attr("T: {half, float, double}")
    .Attr("Tindices: {int32, int64}")
    .Attr("combiner: {'SUM', 'MEAN'} = 'MEAN'")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle indices;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices));
      ShapeHandle values;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &values));
      ShapeHandle weights;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &weights));
      DimensionHandle output_dim = c->Dim(values, 1);
      ShapeHandle output;
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(indices, c->Rank(indices) - 1, output_dim, &output));

      // Validate if indices and weights have same shape.
      if (c->RankKnown(indices) && c->RankKnown(weights)) {
        DimensionHandle unused;
        for (int32 i = 0; i < 2; ++i) {
          TF_RETURN_IF_ERROR(
              c->Merge(c->Dim(indices, i), c->Dim(weights, i), &unused));
        }
      }

      c->set_output(0, output);
      return Status::OK();
    });

REGISTER_OP("Addons>EmbeddingBagGrad")
    .Input("indices: Tindices")
    .Input("values: T")
    .Input("weights: T")
    .Input("grads: T")
    .Output("value_grads: T")
    .Output("weight_grads: T")
    .Attr("T: {half, float, double}")
    .Attr("Tindices: {int32, int64}")
    .Attr("combiner: {'SUM', 'MEAN'} = 'MEAN'")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle indices;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &indices));
      ShapeHandle values;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &values));
      ShapeHandle weights;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &weights));
      ShapeHandle grads;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &grads));

      // Validate if indices and weights have same shape.
      if (c->RankKnown(indices) && c->RankKnown(weights)) {
        DimensionHandle unused;
        for (int32 i = 0; i < 2; ++i) {
          TF_RETURN_IF_ERROR(
              c->Merge(c->Dim(indices, i), c->Dim(weights, i), &unused));
        }
      }

      c->set_output(0, values);
      c->set_output(1, weights);
      return Status::OK();
    });

}  // namespace addons
}  // namespace tensorflow
