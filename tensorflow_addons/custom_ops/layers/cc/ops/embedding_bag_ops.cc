/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
    .Attr("combiner: string = 'sum'")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle indices_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &indices_shape));

      ShapeHandle values_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &values_shape));

      ShapeHandle weights_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 2, &weights_shape));
      // TODO Confirm that indices and weights have the same shape

      DimensionHandle valuesDim = c->Dim(values_shape, 1);
      ShapeHandle output;
      TF_RETURN_IF_ERROR(c->ReplaceDim(
          indices_shape, c->Rank(indices_shape) - 1, valuesDim, &output));

      c->set_output(0, output);
      return Status::OK();
    });

REGISTER_OP("Addons>EmbeddingBagGrad")
    .Attr("T_indices: {int32, int64}")
    .Input("indices: T_indices")
    .Input("values: float")
    .Input("weights: float")
    .Input("dloss: float")
    .Output("values_grad: float")
    .Output("weights_grad: float")
    .Output("dummy1: T_indices")
    .Output("dummy2: T_indices")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle indices_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 2, &indices_shape));

      ::tensorflow::shape_inference::ShapeHandle values_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &values_shape));

      ::tensorflow::shape_inference::ShapeHandle weights_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 2, &weights_shape));

      ::tensorflow::shape_inference::ShapeHandle dloss_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(3), 2, &dloss_shape));

      c->set_output(0, values_shape);
      c->set_output(1, weights_shape);
      c->set_output(2, indices_shape);
      c->set_output(3, indices_shape);
      return Status::OK();
    });

}  // namespace addons
}  // namespace tensorflow
