import tensorflow as tf


class MAP(tf.metrics.Metric):
    def __init__(self, num_classes, iou_threshold=0.5, name='mAP'):
        super(MAP, self).__init__(name=name)
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError(
                "iou threshold value should be greater equal than zero and less equal than one"
            )

        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.ground_truth = []
        self.detection_result = []
        for i in range(self.num_classes):
            self.detection_result.append([])
        self.index = 0

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        self.ground_truth.append([])
        for cls in range(self.num_classes):
            for x in y_pred:
                if tf.equal(x[0], cls):
                    self.detection_result[cls].append(
                        [x[1], self.index, x[2:]])
        for x in y_true:
            self.ground_truth[self.index].append([x[0], x[2:], x[1], 0])
        self.index += 1

    @tf.function
    def result(self):
        APs = tf.TensorArray(tf.float32, self.num_classes)
        APs.write(0, 0)
        for cls in tf.range(self.num_classes):
            npos = 0

            nd = len(self.detection_result[cls])
            tp = [0.] * nd
            fp = [0.] * nd
            idx = 0
            for x in self.detection_result[cls]:
                bbox = x[2]
                ovmax = -float('inf')
                gt_match = -1
                for obj in self.ground_truth[x[1]]:
                    if tf.equal(obj[0], cls):
                        npos += 1
                        bbox = tf.cast(bbox, tf.float32)
                        BBGT = tf.cast(obj[1], tf.float32)
                        if tf.shape(BBGT)[0] > 0:
                            ixmin = tf.maximum(BBGT[0], bbox[0])
                            iymin = tf.maximum(BBGT[1], bbox[1])
                            ixmax = tf.minimum(BBGT[2], bbox[2])
                            iymax = tf.minimum(BBGT[3], bbox[3])
                            iw = tf.maximum(ixmax - ixmin + 1., 0.)
                            ih = tf.maximum(iymax - iymin + 1., 0.)
                            inters = iw * ih

                            # union
                            uni = ((bbox[2] - bbox[0] + 1.) *
                                   (bbox[3] - bbox[1] + 1.) +
                                   (BBGT[2] - BBGT[0] + 1.) *
                                   (BBGT[3] - BBGT[1] + 1.) - inters)

                            overlaps = inters / uni
                            if overlaps > ovmax:
                                ovmax = overlaps
                                gt_match = obj

                if ovmax > self.iou_threshold:
                    if tf.equal(gt_match[2], 0):
                        if tf.equal(gt_match[3], 0):
                            tp[idx] = 1.
                            gt_match[3] = 1
                        else:
                            fp[idx] = 1.
                else:
                    fp[idx] = 1.
                idx += 1
            fp = tf.cumsum(fp)
            tp = tf.cumsum(tp)
            rec = tp / tf.maximum(tf.cast(npos, tf.float32), 1e-8)
            prec = tp / tf.maximum(tp + fp, 1e-8)

            ap = self._voc_ap(rec, prec)

            APs.write(cls, ap)
        return APs.stack()

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "iou_threshold": self.iou_threshold,
        }
        base_config = super(MAP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        self.ground_truth = []
        self.detection_result = []
        for i in range(self.num_classes):
            self.detection_result.append([])
        self.index = 0

    def _voc_ap(self, recall, precision):
        mrec = tf.concat([[0.], recall, [1.]], 0)
        mpre = tf.concat([[0.], precision, [0.]], 0)
        mpre_arr = tf.TensorArray(tf.float32, tf.shape(mpre)[0])
        last_idx = tf.shape(mpre)[0] - 1
        mpre_arr.write(last_idx, mpre[last_idx])
        for i in tf.range(last_idx, 0, -1):
            mpre_arr.write(i - 1, tf.maximum(mpre[i - 1], mpre[i]))
        mpre = mpre_arr.stack()
        idxs = tf.where(mrec[1:] != mrec[:-1])[:, 0]
        ap = 0
        for i in idxs:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
        return ap
