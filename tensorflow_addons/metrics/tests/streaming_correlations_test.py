# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for streaming correlations metrics."""
import pytest
import numpy as np
import tensorflow as tf
from scipy import stats


from tensorflow_addons.metrics import KendallsTauB
from tensorflow_addons.metrics import KendallsTauC
from tensorflow_addons.metrics import PearsonsCorrelation
from tensorflow_addons.metrics import SpearmansRank
from tensorflow_addons.testing.serialization import check_metric_serialization


class TestStreamingCorrelations:
    scipy_corr = {
        KendallsTauB: lambda x, y: stats.kendalltau(x, y, variant="b"),
        KendallsTauC: lambda x, y: stats.kendalltau(x, y, variant="c"),
        SpearmansRank: stats.spearmanr,
        PearsonsCorrelation: stats.pearsonr,
    }

    testing_types = scipy_corr.keys()

    @pytest.mark.parametrize("correlation_type", testing_types)
    def test_config(self, correlation_type):
        obj = correlation_type(name=correlation_type.__name__)
        assert obj.name == correlation_type.__name__
        assert obj.dtype == tf.float32
        assert obj.actual_min == 0.0
        assert obj.actual_max == 1.0

        # Check save and restore config
        kp_obj2 = correlation_type.from_config(obj.get_config())
        assert kp_obj2.name == correlation_type.__name__
        assert kp_obj2.dtype == tf.float32
        assert kp_obj2.actual_min == 0.0
        assert kp_obj2.actual_max == 1.0

    @pytest.mark.parametrize("correlation_type", testing_types)
    def test_scoring_with_ties(self, correlation_type):
        actuals = [12, 2, 1, 12, 2]
        preds = [1, 4, 7, 1, 0]
        metric = correlation_type(0, 13, 0, 8)
        metric.update_state(actuals, preds)

        scipy_value = self.scipy_corr[correlation_type](actuals, preds)[0]
        np.testing.assert_almost_equal(metric.result(), scipy_value, decimal=2)

    @pytest.mark.parametrize("correlation_type", testing_types)
    def test_perfect(self, correlation_type):
        actuals = [1, 2, 3, 4, 5, 6, 7, 8]
        preds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        metric = correlation_type(0, 10, 0.0, 1.0)
        metric.update_state(actuals, preds)

        scipy_value = self.scipy_corr[correlation_type](actuals, preds)[0]
        np.testing.assert_almost_equal(metric.result(), scipy_value)

    @pytest.mark.parametrize("correlation_type", testing_types)
    def test_reversed(self, correlation_type):
        actuals = [1, 2, 3, 4, 5, 6, 7, 8]
        preds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8][::-1]
        metric = correlation_type(0, 10, 0.0, 1.0)
        metric.update_state(actuals, preds)

        scipy_value = self.scipy_corr[correlation_type](actuals, preds)[0]
        np.testing.assert_almost_equal(metric.result(), scipy_value)

    @pytest.mark.parametrize("correlation_type", testing_types)
    def test_scoring_streaming(self, correlation_type):
        actuals = [12, 2, 1, 12, 2]
        preds = [1, 4, 7, 1, 0]

        metric = correlation_type(0, 13, 0, 8)
        for actual, pred in zip(actuals, preds):
            metric.update_state([[actual]], [[pred]])

        scipy_value = self.scipy_corr[correlation_type](actuals, preds)[0]
        np.testing.assert_almost_equal(metric.result(), scipy_value, decimal=2)

    @pytest.mark.parametrize("correlation_type", testing_types)
    @pytest.mark.usefixtures("maybe_run_functions_eagerly")
    def test_keras_binary_classification_model(self, correlation_type):
        metric = correlation_type()
        inputs = tf.keras.layers.Input(shape=(128,))
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(inputs)
        model = tf.keras.models.Model(inputs, outputs)
        if hasattr(tf.keras.optimizers, "legacy"):
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.1)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[metric],
        )

        x = np.random.rand(1024, 128).astype(np.float32)
        y = np.random.randint(2, size=(1024, 1)).astype(np.float32)

        initial_correlation = self.scipy_corr[correlation_type](
            model(x)[:, 0], y[:, 0]
        )[0]

        history = model.fit(
            x, y, epochs=1, verbose=0, batch_size=32, validation_data=(x, y)
        )

        # the training should increase the correlation metric
        metric_history = history.history["val_" + metric.name]
        assert np.all(metric_history > initial_correlation)

        preds = model(x)
        metric.reset_state()
        # we decorate with tf.function to ensure the metric is also checked against graph mode.
        # keras automatically decorates the metrics compiled within keras.Model.
        tf.function(metric.update_state)(y, preds)
        metric_value = tf.function(metric.result)()
        scipy_value = self.scipy_corr[correlation_type](preds[:, 0], y[:, 0])[0]
        np.testing.assert_almost_equal(metric_value, metric_history[-1], decimal=6)
        np.testing.assert_almost_equal(metric_value, scipy_value, decimal=2)

    @pytest.mark.parametrize("correlation_type", testing_types)
    def test_serialization(self, correlation_type):
        actuals = np.array([4, 4, 3, 3, 2, 2, 1, 1], dtype=np.int32)
        preds = np.array([1, 2, 4, 1, 3, 3, 4, 4], dtype=np.int32)

        kt = correlation_type(0, 5, 0, 5, 10, 10)
        check_metric_serialization(kt, actuals, preds)
