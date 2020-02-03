"""Tests for Multilabel Metrics."""

import tensorflow as tf
from tensorflow_addons.metrics import (
    MultiLabelMacroRecall,
    MultiLabelMacroSensitivity,
    MultiLabelMacroSpecificity,
)
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class MultiLabelMacroRecallTest(tf.test.TestCase):
    def test_config(self):
        _class = MultiLabelMacroRecall
        _name = "multi_label_macro_recall"

        obj_a = _class(name=_name)
        self.assertEqual(obj_a._name, _name)

        # Check save and restore config
        obj_b = _class.from_config(obj_a.get_config())
        self.assertEqual(obj_b._name, _name)

    def initialize_vars(self):
        _class = MultiLabelMacroRecall
        obj = _class()
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        return obj

    def update_obj_states(self, obj, y_true, y_pred):
        update_op = obj.update_state(y_true, y_pred)
        self.evaluate(update_op)

    def check_results(self, obj, value):
        self.assertAllClose(value, self.evaluate(obj.result()), atol=1e-5)

    def test_recall_score(self):
        obj = self.initialize_vars()

        y_true = tf.cast(
            tf.constant(
                [[0, 0, 1, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 0, 1],], dtype=tf.float32
            ),
            dtype=tf.float32,
        )

        y_pred = tf.cast(
            tf.constant(
                [[0, 0, 1, 0, 0], [1, 0, 1, 1, 0], [1, 0, 0, 0, 1],], dtype=tf.float32
            ),
            dtype=tf.float32,
        )

        self.update_obj_states(obj, y_true, y_pred)
        self.check_results(obj, 0.6666667)
        #
        # obj.reset_states()
        # expected_single_results = [0.5, 0.0, 1.0, 0.0, 1.0]
        # for i, res in enumerate(xpected_single_results):
        #     self.update_obj_states(obj, y_true[:, i], y_pred[:, i])
        #     self.check_results(obj, res)
        #     obj.reset_states()


@test_utils.run_all_in_graph_and_eager_modes
class MultiLabelMacroSensitivityTest(tf.test.TestCase):
    def test_config(self):
        _class = MultiLabelMacroSensitivity
        _name = "multi_label_macro_sensitivity"
        _threshold = 0.5
        _from_logits = True
        _activation = "sigmoid"

        obj_a = _class(
            name=_name,
            threshold=_threshold,
            from_logits=_from_logits,
            activation=_activation,
        )
        self.assertEqual(obj_a._name, _name)
        # self.assertEqual(obj_a._threshold, tf.constant(_threshold, dtype=tf.float32))
        # self.assertEqual(obj_a._from_logits, _from_logits)
        # self.assertEqual(obj_a._activation, _activation)

        # Check save and restore config
        obj_b = _class.from_config(obj_a.get_config())
        self.assertEqual(obj_b._name, _name)
        # self.assertEqual(obj_b._threshold, _threshold)
        # self.assertEqual(obj_b._from_logits, _from_logits)
        # self.assertEqual(obj_b._activation, _activation)

    def initialize_vars(self):
        _class = MultiLabelMacroSensitivity
        _name = "multi_label_macro_sensitivity"
        _threshold = 0.5
        _from_logits = True
        _activation = "sigmoid"
        obj = _class()
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        return obj

    def update_obj_states(self, obj, y_true, y_pred):
        update_op = obj.update_state(y_true, y_pred)
        self.evaluate(update_op)

    def check_results(self, obj, value):
        self.assertAllClose(value, self.evaluate(obj.result()), atol=1e-5)

    def test_sensitivity_score(self):
        obj = self.initialize_vars()
        y_true = tf.cast(
            tf.constant(
                [[0, 0, 1, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 0, 1],], dtype=tf.float32
            ),
            dtype=tf.float32,
        )

        y_pred = tf.cast(
            tf.constant(
                [[0, 0, 1, 0, 0], [1, 0, 1, 1, 0], [1, 0, 0, 0, 1],], dtype=tf.float32
            ),
            dtype=tf.float32,
        )
        self.update_obj_states(obj, y_true, y_pred)
        self.check_results(obj, 0.5714286)

        # obj.reset_states()
        # expected_single_results = [1.0, 0.0, 1.0, 0.0, 0.33333334]
        # for i, res in enumerate(xpected_single_results):
        #     self.update_obj_states(obj, y_true[:, i], y_pred[:, i])
        #     self.check_results(obj, res)
        #     obj.reset_states()


@test_utils.run_all_in_graph_and_eager_modes
class MultiLabelMacroSpecificityTest(tf.test.TestCase):
    def test_config(self):
        _class = MultiLabelMacroSpecificity
        _name = "multi_label_macro_specificity"
        _threshold = 0.5
        _from_logits = True
        _activation = "sigmoid"

        obj_a = _class(
            name=_name,
            threshold=_threshold,
            from_logits=_from_logits,
            activation=_activation,
        )
        self.assertEqual(obj_a._name, _name)
        # self.assertEqual(obj_a._threshold, tf.constant(_threshold, dtype=tf.float32))
        # self.assertEqual(obj_a._from_logits, _from_logits)
        # self.assertEqual(obj_a._activation, _activation)

        # Check save and restore config
        obj_b = _class.from_config(obj_a.get_config())
        self.assertEqual(obj_b._name, _name)
        # self.assertEqual(obj_b._threshold, _threshold)
        # self.assertEqual(obj_b._from_logits, _from_logits)
        # self.assertEqual(obj_b._activation, _activation)

    def initialize_vars(self):
        _class = MultiLabelMacroSpecificity
        _name = "multi_label_macro_specificity"
        _threshold = 0.5
        _from_logits = True
        _activation = "sigmoid"
        obj = _class()
        self.evaluate(tf.compat.v1.variables_initializer(obj.variables))
        return obj

    def update_obj_states(self, obj, y_true, y_pred):
        update_op = obj.update_state(y_true, y_pred)
        self.evaluate(update_op)

    def check_results(self, obj, value):
        self.assertAllClose(value, self.evaluate(obj.result()), atol=1e-5)

    def test_specificity_score(self):
        obj = self.initialize_vars()
        y_true = tf.cast(
            tf.constant(
                [[0, 0, 1, 1, 1], [0, 0, 1, 0, 1], [1, 0, 0, 0, 1],], dtype=tf.float32
            ),
            dtype=tf.float32,
        )

        y_pred = tf.cast(
            tf.constant(
                [[0, 0, 1, 0, 0], [1, 0, 1, 1, 0], [1, 0, 0, 0, 1],], dtype=tf.float32
            ),
            dtype=tf.float32,
        )
        self.update_obj_states(obj, y_true, y_pred)
        self.check_results(obj, 0.75)

        # obj.reset_states()
        # expected_single_results = [0.5, 1.0, 1.0, 0.5, 0.0]
        # for i, res in enumerate(xpected_single_results):
        #     self.update_obj_states(obj, y_true[:, i], y_pred[:, i])
        #     self.check_results(obj, res)
        #     obj.reset_states()


if __name__ == "__main__":
    tf.test.main()
