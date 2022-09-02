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
"""Tests for `MutualInformation` metric."""
import pytest
import numpy as np

from tensorflow_addons.metrics import MutualInformation
from tensorflow_addons.testing.serialization import check_metric_serialization


class TestMutualInformation:
    def generate_mixed_data(self, n, correlation=0.9, beta=0.9):
        """Generate distribution from experiment I
        described in https://arxiv.org/pdf/1709.06212v3.pdf
        """
        np.random.seed(0)
        # https://github.com/wgao9/mixed_KSG/blob/master/demo.py
        cov = [[1, correlation], [correlation, 1]]
        p_con, p_dis = 0.5, 0.5
        gt = (
            -p_con * 0.5 * np.log(np.linalg.det(cov))
            + p_dis * (np.log(2) + beta * np.log(beta) + (1 - beta) * np.log(1 - beta))
            - p_con * np.log(p_con)
            - p_dis * np.log(p_dis)
        )

        x_con, y_con = np.random.multivariate_normal([0, 0], cov, int(n * p_con)).T
        x_dis = np.random.binomial(1, 0.5, int(n * p_dis))
        y_dis = (x_dis + np.random.binomial(1, 1 - beta, int(n * p_dis))) % 2
        x_dis, y_dis = 2 * x_dis - np.ones(int(n * p_dis)), 2 * y_dis - np.ones(
            int(n * p_dis)
        )
        x = np.concatenate((x_con, x_dis))
        y = np.concatenate((y_con, y_dis))
        data = np.stack([x, y], axis=-1)
        np.random.shuffle(data)
        return data, gt

    def generate_discrete_continuous_data(self, n=1024, m=5):
        """Generate distribution from experiment II
        described in https://arxiv.org/pdf/1709.06212v3.pdf
        """
        np.random.seed(0)
        x = np.random.randint(m, size=n)
        y = np.random.random(size=n)
        y = y * 2 + x
        data = np.stack([x, y], axis=-1)
        true_mi = np.log(m) - (m - 1) * np.log(2) / m
        return data, true_mi

    def test_uniform(self):
        np.random.seed(0)
        data = np.random.random(size=(4096, 2))
        x, y = data[:, 0], data[:, 1]
        metric = MutualInformation(
            buffer_size=512, compute_batch_size=512, n_neighbors=3
        )
        metric.update_state(x, y)
        np.testing.assert_allclose(metric.result(), 0.0, atol=5e-3)

    @pytest.mark.parametrize("correlation", [0.0, 0.1, 0.5, 0.9])
    def test_gaussian(self, correlation):
        np.random.seed(0)
        cov = np.array([[1.0, correlation], [correlation, 1.0]])
        data = np.random.multivariate_normal([0.0, 0.0], cov=cov, size=4096)
        x, y = data[:, 0], data[:, 1]
        metric = MutualInformation(
            buffer_size=512, compute_batch_size=512, n_neighbors=3
        )
        metric.update_state(x, y)

        true_mi = -0.5 * np.log(1 - correlation**2)
        np.testing.assert_allclose(metric.result(), true_mi, atol=5e-2)

    @pytest.mark.parametrize("correlation, beta", [(0.9, 0.9), (0.5, 0.5)])
    def test_mixed(self, correlation, beta):
        data, true_mi = self.generate_mixed_data(
            n=4096, correlation=correlation, beta=beta
        )
        x, y = data[:, 0], data[:, 1]
        metric = MutualInformation(
            buffer_size=4096, compute_batch_size=512, n_neighbors=1
        )
        metric.update_state(x, y)
        np.testing.assert_allclose(metric.result(), true_mi, atol=6e-2)

    def test_discrete_x_continuous_y(self):
        data, true_mi = self.generate_discrete_continuous_data(n=4096, m=5)
        x, y = data[:, 0], data[:, 1]
        metric = MutualInformation(
            buffer_size=4096, compute_batch_size=512, n_neighbors=1
        )
        metric.update_state(x, y)
        np.testing.assert_allclose(metric.result(), true_mi, atol=5e-3)

    def test_serialization(self):
        labels = np.array([4, 4, 3, 3, 2, 2, 1, 1], dtype=np.int32)
        preds = np.array([1, 2, 4, 1, 3, 3, 4, 4], dtype=np.int32)

        obj = MutualInformation(buffer_size=512, compute_batch_size=512, n_neighbors=3)
        check_metric_serialization(obj, labels, preds)
