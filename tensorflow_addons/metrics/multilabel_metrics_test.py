"""Tests for Multilabel Metrics."""

import tensorflow as tf
from tensorflow_addons.metrics import (
    MultiLabelMacroRecall,
    MultiLabelMacroSensitivity,
    MultiLabelMacroSpecificity
)
from tensorflow_addons.utils import test_utils

@test_utils.run_all_in_graph_and_eager_modes
class MultiLabelMacroRecallTest(tf.test.TestCase):
    _class = MultiLabelMacroRecall
    _name = 'multi_label_macro_recall'
    def test_config(self):
        obj_a = self._class(name=self._name)
        self.assertEqual(obj_a.name, self._name)

        # Check save and restore config
        obj_b = _class.from_config(obj_a.get_config())
        self.assertEqual(obj_b.name, self._name)

    def initialize_vars(self):
        obj = self._class()
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        return obj

    def update_obj_states(self, obj, y_true, y_pred):
        update_op = obj.update_state(y_true, y_pred)
        self.evaluate(update_op)

    def check_results(self, obj, value):
        self.assertAllClose(value, self.evaluate(obj.result()), atol=1e-5)


    def test_recall_score(self):
        # num labels 3, num examples 5, here testing a single
        # example so shape is (3, 5)
        y_true = tf.cast(
            tf.constant(
                [
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 0, 1],
                  [1, 0, 0, 0, 1],
                ], dtype=tf.float32
            ), dtype=tf.float32
        )
        y_pred = tf.cast(
            tf.constant(
                [
                  [0, 0, 1, 0, 0],
                  [1, 0, 1, 1, 0],
                  [1, 0, 0, 0, 1],
                ], dtype=tf.float32
            ), dtype=tf.float32
        )

        obj = self.initialize_vars()
        self.update_obj_states(obj, y_true, y_preds)
        self.check_results(obj, 0.5)

        obj.reset_states()
        expected_single_results = [
            0.5, 0.0, 1.0, 0.0, 0.33333334
        ]
        for i, res in enumerate(xpected_single_results):
            self.update_obj_states(obj, y_true[:, i], y_preds[:, i])
            self.check_results(obj, res)
            obj.reset_states()

        pass





if __name__ == '__main__':
    tf.test.main()
