import tensorflow as tf
from tensorflow.keras.metrics import Metric
import numpy as np


class F1_score(Metric):
    """
    Computes F1 macro score
    """
    def __init__(self, num_classes, name='f1-score'):
        super(F1_score, self).__init__(name=name)
        self.num_classes = num_classes
        self.true_positives_col = self.add_weight('TP-class',
                                                  shape=[self.num_classes],
                                                  initializer='zeros',
                                                  dtype=tf.float32)
        self.false_positives_col = self.add_weight('FP-class',
                                                   shape=[self.num_classes],
                                                   initializer='zeros',
                                                   dtype=tf.float32)
        self.false_negatives_col = self.add_weight('FN-class',
                                                   shape=[self.num_classes],
                                                   initializer='zeros',
                                                   dtype=tf.float32)

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        # true positive across column
        self.true_positives_col.assign_add(tf.cast(tf.math.count_nonzero(
            y_pred * y_true, axis=0), tf.float32))
        # false positive across column
        self.false_positives_col.assign_add(tf.cast(tf.math.count_nonzero(
            y_pred * (y_true - 1), axis=0), tf.float32))
        # false negative across column
        self.false_negatives_col.assign_add(tf.cast(
            tf.math.count_nonzero((y_pred - 1) * y_true, axis=0), tf.float32))

    def result(self):
        p_sum = tf.cast(self.true_positives_col + self.false_positives_col,
                        tf.float32)
        precision_macro = tf.cast(tf.compat.v1.div_no_nan(
            self.true_positives_col, p_sum), tf.float32)

        r_sum = tf.cast(self.true_positives_col + self.false_negatives_col,
                        tf.float32)
        recall_macro = tf.cast(tf.compat.v1.div_no_nan(
            self.true_positives_col, r_sum), tf.float32)

        mul_value = 2 * precision_macro * recall_macro
        add_value = precision_macro + recall_macro
        f1_macro = tf.cast(tf.compat.v1.div_no_nan(mul_value, add_value),
                           tf.float32)

        f1_macro = tf.reduce_mean(f1_macro)

        return f1_macro

    def reset_states(self):
        self.true_positives_col.assign(np.zeros(self.num_classes), np.float32)
        self.false_positives_col.assign(np.zeros(self.num_classes), np.float32)
        self.false_negatives_col.assign(np.zeros(self.num_classes), np.float32)
