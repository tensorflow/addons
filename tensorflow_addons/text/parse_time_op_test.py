# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Parse time op tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_addons import text
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class ParseTimeTest(tf.test.TestCase):
    def test_parse_time(self):
        time_format = "%Y-%m-%dT%H:%M:%E*S%Ez"
        items = [
            ("2019-05-17T23:56:09.05Z", time_format, "NANOSECOND",
             1558137369050000000),
            ("2019-05-17T23:56:09.05Z", time_format, "MICROSECOND",
             1558137369050000),
            ("2019-05-17T23:56:09.05Z", time_format, "MILLISECOND",
             1558137369050),
            ("2019-05-17T23:56:09.05Z", time_format, "SECOND", 1558137369),
            ([
                "2019-05-17T23:56:09.05Z", "2019-05-20T11:22:33.44Z",
                "2019-05-30T22:33:44.55Z"
            ], time_format, "MILLISECOND",
             [1558137369050, 1558351353440, 1559255624550]),
        ]
        for time_string, time_format, output_unit, expected in items:
            result = self.evaluate(
                text.parse_time(
                    time_string=time_string,
                    time_format=time_format,
                    output_unit=output_unit))
            self.assertAllEqual(expected, result)

    def test_invalid_output_unit(self):
        errors = (ValueError, tf.errors.InvalidArgumentError)
        with self.assertRaises(errors):
            text.parse_time(
                time_string="2019-05-17T23:56:09.05Z",
                time_format="%Y-%m-%dT%H:%M:%E*S%Ez",
                output_unit="INVALID")

    def test_invalid_time_format(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.evaluate(
                text.parse_time(
                    time_string="2019-05-17T23:56:09.05Z",
                    time_format="INVALID",
                    output_unit="SECOND"))

    def test_invalid_time_string(self):
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.evaluate(
                text.parse_time(
                    time_string="INVALID",
                    time_format="%Y-%m-%dT%H:%M:%E*S%Ez",
                    output_unit="SECOND"))


if __name__ == "__main__":
    tf.test.main()
