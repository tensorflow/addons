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
import platform

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons import text

IS_WINDOWS = platform.system() == "Windows"

pytestmark = pytest.mark.skipif(
    IS_WINDOWS,
    reason="Doesn't work on Windows, see https://github.com/tensorflow/addons/issues/782",
)


def test_parse_time():
    time_format = "%Y-%m-%dT%H:%M:%E*S%Ez"
    items = [
        ("2019-05-17T23:56:09.05Z", time_format, "NANOSECOND", 1558137369050000000),
        ("2019-05-17T23:56:09.05Z", time_format, "MICROSECOND", 1558137369050000),
        ("2019-05-17T23:56:09.05Z", time_format, "MILLISECOND", 1558137369050),
        ("2019-05-17T23:56:09.05Z", time_format, "SECOND", 1558137369),
        (
            [
                "2019-05-17T23:56:09.05Z",
                "2019-05-20T11:22:33.44Z",
                "2019-05-30T22:33:44.55Z",
            ],
            time_format,
            "MILLISECOND",
            [1558137369050, 1558351353440, 1559255624550],
        ),
    ]
    for time_string, time_format, output_unit, expected in items:
        result = text.parse_time(
            time_string=time_string, time_format=time_format, output_unit=output_unit
        )
        np.testing.assert_equal(expected, result.numpy())


def test_invalid_output_unit():
    errors = (ValueError, tf.errors.InvalidArgumentError)
    with pytest.raises(errors):
        text.parse_time(
            time_string="2019-05-17T23:56:09.05Z",
            time_format="%Y-%m-%dT%H:%M:%E*S%Ez",
            output_unit="INVALID",
        )


def test_invalid_time_format():
    with pytest.raises(tf.errors.InvalidArgumentError):
        text.parse_time(
            time_string="2019-05-17T23:56:09.05Z",
            time_format="INVALID",
            output_unit="SECOND",
        )


def test_invalid_time_string():
    with pytest.raises(tf.errors.InvalidArgumentError):
        text.parse_time(
            time_string="INVALID",
            time_format="%Y-%m-%dT%H:%M:%E*S%Ez",
            output_unit="SECOND",
        )
