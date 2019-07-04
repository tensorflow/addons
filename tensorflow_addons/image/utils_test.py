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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_addons.image import utils as img_utils
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class UtilsOpsTest(tf.test.TestCase):
    def test_to_4D_image(self):
        for shape in (2, 4), (2, 4, 1), (1, 2, 4, 1):
            exp = tf.ones(shape=(1, 2, 4, 1))
            res = img_utils.to_4D_image(tf.ones(shape=shape))
            # static shape:
            self.assertAllEqual(exp.get_shape(), res.get_shape())
            self.assertAllEqual(self.evaluate(exp), self.evaluate(res))

    def test_to_4D_image_with_unknown_shape(self):
        fn = img_utils.to_4D_image.get_concrete_function(
            tf.TensorSpec(shape=None, dtype=tf.float32))
        for shape in (2, 4), (2, 4, 1), (1, 2, 4, 1):
            exp = tf.ones(shape=(1, 2, 4, 1))
            res = fn(tf.ones(shape=shape))
            self.assertAllEqual(self.evaluate(exp), self.evaluate(res))

    def test_to_4D_image_with_invalid_shape(self):
        with self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
            img_utils.to_4D_image(tf.ones(shape=(1,)))

        with self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
            img_utils.to_4D_image(tf.ones(shape=(1, 2, 4, 3, 2)))

    def test_from_4D_image(self):
        for shape in (2, 4), (2, 4, 1), (1, 2, 4, 1):
            exp = tf.ones(shape=shape)
            res = img_utils.from_4D_image(
                tf.ones(shape=(1, 2, 4, 1)), len(shape))
            # static shape:
            self.assertAllEqual(exp.get_shape(), res.get_shape())
            self.assertAllEqual(self.evaluate(exp), self.evaluate(res))

    def test_from_4D_image_with_unknown_shape(self):
        for shape in (2, 4), (2, 4, 1), (1, 2, 4, 1):
            exp = tf.ones(shape=shape)
            fn = img_utils.from_4D_image.get_concrete_function(
                tf.TensorSpec(shape=None, dtype=tf.float32), tf.size(shape))
            res = fn(tf.ones(shape=(1, 2, 4, 1)), tf.size(shape))
            self.assertAllEqual(self.evaluate(exp), self.evaluate(res))

    def test_from_4D_image_with_invalid_data(self):
        with self.assertRaises(ValueError):
            self.evaluate(
                img_utils.from_4D_image(tf.ones(shape=(2, 2, 4, 1)), 2))

        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.evaluate(
                img_utils.from_4D_image(
                    tf.ones(shape=(2, 2, 4, 1)), tf.constant(2)))

    def test_from_4D_image_with_invalid_shape(self):
        for rank in 2, tf.constant(2):
            with self.subTest(rank=rank):
                with self.assertRaises((ValueError,
                                        tf.errors.InvalidArgumentError)):
                    img_utils.from_4D_image(tf.ones(shape=(2, 4)), rank)

                with self.assertRaises((ValueError,
                                        tf.errors.InvalidArgumentError)):
                    img_utils.from_4D_image(tf.ones(shape=(2, 4, 1)), rank)

                with self.assertRaises((ValueError,
                                        tf.errors.InvalidArgumentError)):
                    img_utils.from_4D_image(
                        tf.ones(shape=(1, 2, 4, 1, 1)), rank)


if __name__ == "__main__":
    tf.test.main()
