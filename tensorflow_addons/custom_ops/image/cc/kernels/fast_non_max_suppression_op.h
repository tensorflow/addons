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

#ifndef TENSORFLOW_ADDONS_FAST_NON_MAX_SUPPRESSION_OP_H_
#define TENSORFLOW_ADDONS_FAST_NON_MAX_SUPPRESSION_OP_H_

#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace addons {

template <typename T>
Eigen::Tensor<T, 3, Eigen::RowMajor> jaccard(
    typename TTypes<T, 3>::ConstTensor box_a,
    typename TTypes<T, 3>::ConstTensor box_b) {
  const int num_classes = box_a.dimension(0);
  const int per_class_boxes_a = box_a.dimension(1);
  const int per_class_boxes_b = box_b.dimension(1);
  const int box_edges = box_a.dimension(2);

  const Eigen::array<int, 4> expand_shapes_a = {num_classes, per_class_boxes_a,
                                                1, box_edges};
  const Eigen::array<int, 4> expand_shapes_b = {num_classes, 1,
                                                per_class_boxes_b, box_edges};

  auto box_a_ymin =
      box_a.slice(Eigen::array<int, 3>({0, 0, 0}),
                  Eigen::array<int, 3>({num_classes, per_class_boxes_a, 1}));
  auto expand_box_a_ymin = box_a_ymin.reshape(expand_shapes_a);
  auto box_a_xmin =
      box_a.slice(Eigen::array<int, 3>({0, 0, 1}),
                  Eigen::array<int, 3>({num_classes, per_class_boxes_a, 1}));
  auto expand_box_a_xmin = box_a_xmin.reshape(expand_shapes_a);
  auto box_a_ymax =
      box_a.slice(Eigen::array<int, 3>({0, 0, 2}),
                  Eigen::array<int, 3>({num_classes, per_class_boxes_a, 1}));
  auto expand_box_a_ymax = box_a_ymax.reshape(expand_shapes_a);
  auto box_a_xmax =
      box_a.slice(Eigen::array<int, 3>({0, 0, 3}),
                  Eigen::array<int, 3>({num_classes, per_class_boxes_a, 1}));
  auto expand_box_a_xmax = box_a_xmax.reshape(expand_shapes_a);
  auto box_b_ymin =
      box_b.slice(Eigen::array<int, 3>({0, 0, 0}),
                  Eigen::array<int, 3>({num_classes, per_class_boxes_b, 1}));
  auto expand_box_b_ymin = box_b_ymin.reshape(expand_shapes_b);
  auto box_b_xmin =
      box_b.slice(Eigen::array<int, 3>({0, 0, 1}),
                  Eigen::array<int, 3>({num_classes, per_class_boxes_b, 1}));
  auto expand_box_b_xmin = box_b_xmin.reshape(expand_shapes_b);
  auto box_b_ymax =
      box_b.slice(Eigen::array<int, 3>({0, 0, 2}),
                  Eigen::array<int, 3>({num_classes, per_class_boxes_b, 1}));
  auto expand_box_b_ymax = box_b_ymax.reshape(expand_shapes_b);
  auto box_b_xmax =
      box_b.slice(Eigen::array<int, 3>({0, 0, 3}),
                  Eigen::array<int, 3>({num_classes, per_class_boxes_b, 1}));
  auto expand_box_b_xmax = box_b_xmax.reshape(expand_shapes_b);

  auto intersect_ymin = (expand_box_a_ymin > expand_box_b_ymin)
                            .select(expand_box_a_ymin, expand_box_b_ymin);
  auto intersect_xmin = (expand_box_a_xmin > expand_box_b_xmin)
                            .select(expand_box_a_xmin, expand_box_b_xmin);
  auto intersect_ymax = (expand_box_a_ymax < expand_box_b_ymax)
                            .select(expand_box_a_ymax, expand_box_b_ymax);
  auto intersect_xmax = (expand_box_a_xmax < expand_box_b_xmax)
                            .select(expand_box_a_xmax, expand_box_b_xmax);

  Eigen::Tensor<T, 4, Eigen::RowMajor> intersect_h =
      intersect_ymax - intersect_ymin;
  intersect_h =
      (intersect_h >= static_cast<T>(0))
          .select(intersect_h, intersect_h.constant(static_cast<T>(0)));

  Eigen::Tensor<T, 4, Eigen::RowMajor> intersect_w =
      intersect_xmax - intersect_xmin;
  intersect_w =
      (intersect_w >= static_cast<T>(0))
          .select(intersect_w, intersect_w.constant(static_cast<T>(0)));
  auto intersect_area =
      (intersect_h * intersect_w)
          .reshape(Eigen::array<int, 3>({intersect_h.dimension(0),
                                         intersect_h.dimension(1),
                                         intersect_h.dimension(2)}));
  auto area_a = (box_a_ymax - box_a_ymin) * (box_a_xmax - box_a_xmin);
  auto area_b = (box_b_ymax - box_b_ymin) * (box_b_xmax - box_b_xmin);
  return (intersect_area / (area_a + area_b - intersect_area));
}

template <typename Device, typename T>
void DoFastNonMaxSuppression(OpKernelContext* context,
                             typename TTypes<T, 3>::ConstTensor boxes,
                             typename TTypes<T, 2>::ConstTensor scores,
                             const T iou_threshold, const T score_threshold) {
  // const int num_classes = scores.dimension(0);
  auto iou = jaccard<T>(boxes, boxes);

  auto iou_max = iou.maximum(Eigen::array<int, 1>({1}));
  auto iou_mask = (iou_max <= iou_max.constant(iou_threshold))
                      .select(iou_max.constant(static_cast<T>(1)),
                              iou_max.constant(static_cast<T>(0)));
  auto scores_mask = (scores > scores.constant(score_threshold))
                         .select(scores.constant(static_cast<T>(1)),
                                 scores.constant(static_cast<T>(0)));

  Eigen::Tensor<T, 3, Eigen::RowMajor> selected_indices =
      iou_mask * scores_mask;

  Tensor* selected_indices_tensor = nullptr;
  TensorShape selected_indices_shape(
      {static_cast<int>(selected_indices.size())});
  OP_REQUIRES_OK(context, context->allocate_output(0, selected_indices_shape,
                                                   &selected_indices_tensor));
  selected_indices_tensor->tensor<int, 2>().device(
      context->eigen_device<Device>()) = selected_indices.cast<int>();
}

template <typename Device, typename T>
class FastNonMaxSuppressionOp : public OpKernel {
 public:
  explicit FastNonMaxSuppressionOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& boxes = context->input(0);
    const Tensor& scores = context->input(1);

    const Tensor& iou_threshold = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const T iou_threshold_val = iou_threshold.scalar<T>()();
    OP_REQUIRES(context, iou_threshold_val >= static_cast<T>(0.0) &&
                             iou_threshold_val <= static_cast<T>(1.0),
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    const Tensor& score_threshold = context->input(3);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const T score_threshold_val = score_threshold.scalar<T>()();

    typename TTypes<T, 3>::ConstTensor boxes_data = boxes.tensor<T, 3>();
    typename TTypes<T, 2>::ConstTensor scores_data = scores.tensor<T, 2>();

    DoFastNonMaxSuppression<Device, T>(context, boxes_data, scores_data,
                                       iou_threshold_val, score_threshold_val);
  }
};
}  // namespace addons
}  // namespace tensorflow
#endif