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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace addons {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("Addons>FastNonMaxSuppression")
    .Input("boxes: float")
    .Input("scores: float")
    .Input("iou_threshold: float=0.5")
    .Input("score_threshold: float=0.0")
    .Output("selected_indices: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &boxes));
      ShapeHandle scores;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &scores));
      ShapeHandle iou_threshold;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &iou_threshold));
      ShapeHandle score_threshold;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &score_threshold));
      // The boxes is a 2-D float Tensor of shape [num_boxes, 4].
      DimensionHandle unused;
      // The boxes[0] and scores[0] are both num_boxes.
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(boxes, 0), c->Dim(scores, 0), &unused));
      // The boxes[1] is 4.
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 4, &unused));

      c->set_output(0, c->Vector(c->UnknownDim()));
      return Status::OK();
    });
}  // namespace addons
}  // namespace tensorflow