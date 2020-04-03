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
"""Tests for util ops."""

import pytest
import numpy as np
import tensorflow as tf

from tensorflow_addons.image import utils as img_utils


def test_to_4D_image_with_unknown_shape():
    fn = tf.function(img_utils.to_4D_image).get_concrete_function(
        tf.TensorSpec(shape=None, dtype=tf.float32)
    )
    for shape in (2, 4), (2, 4, 1), (1, 2, 4, 1):
        exp = tf.ones(shape=(1, 2, 4, 1))
        res = fn(tf.ones(shape=shape))
        np.testing.assert_equal(exp.numpy(), res.numpy())


def test_to_4D_image_with_invalid_shape():
    errors = (ValueError, tf.errors.InvalidArgumentError)
    with pytest.raises(errors, match="`image` must be 2/3/4D tensor"):
        img_utils.to_4D_image(tf.ones(shape=(1,)))

    with pytest.raises(errors, match="`image` must be 2/3/4D tensor"):
        img_utils.to_4D_image(tf.ones(shape=(1, 2, 4, 3, 2)))


def test_from_4D_image():
    for shape in (2, 4), (2, 4, 1), (1, 2, 4, 1):
        exp = tf.ones(shape=shape)
        res = img_utils.from_4D_image(tf.ones(shape=(1, 2, 4, 1)), len(shape))
        # static shape:
        assert exp.get_shape() == res.get_shape()
        np.testing.assert_equal(exp.numpy(), res.numpy())


def test_to_4D_image():
    for shape in (2, 4), (2, 4, 1), (1, 2, 4, 1):
        exp = tf.ones(shape=(1, 2, 4, 1))
        res = img_utils.to_4D_image(tf.ones(shape=shape))
        # static shape:
        assert exp.get_shape() == res.get_shape()
        np.testing.assert_equal(exp.numpy(), res.numpy())


def test_from_4D_image_with_invalid_data():
    with np.testing.assert_raises((ValueError, tf.errors.InvalidArgumentError)):
        img_utils.from_4D_image(tf.ones(shape=(2, 2, 4, 1)), 2)

    with np.testing.assert_raises((ValueError, tf.errors.InvalidArgumentError)):
        img_utils.from_4D_image(tf.ones(shape=(2, 2, 4, 1)), tf.constant(2))


def test_from_4D_image_with_unknown_shape():
    for shape in (2, 4), (2, 4, 1), (1, 2, 4, 1):
        exp = tf.ones(shape=shape)
        fn = tf.function(img_utils.from_4D_image).get_concrete_function(
            tf.TensorSpec(shape=None, dtype=tf.float32), tf.size(shape)
        )
        res = fn(tf.ones(shape=(1, 2, 4, 1)), tf.size(shape))
        np.testing.assert_equal(exp.numpy(), res.numpy())


@pytest.mark.parametrize("rank", [2, tf.constant(2)])
def test_from_4d_image_with_invalid_shape(rank):
    errors = (ValueError, tf.errors.InvalidArgumentError)
    with pytest.raises(errors, match="`image` must be 4D tensor"):
        img_utils.from_4D_image(tf.ones(shape=(2, 4)), rank)

    with pytest.raises(errors, match="`image` must be 4D tensor"):
        img_utils.from_4D_image(tf.ones(shape=(2, 4, 1)), rank)

    with pytest.raises(errors, match="`image` must be 4D tensor"):
        img_utils.from_4D_image(tf.ones(shape=(1, 2, 4, 1, 1)), rank)
