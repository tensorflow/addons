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
"""Tests for translate ops."""

import tensorflow as tf

from tensorflow_addons.image import translate_ops
from tensorflow_addons.utils import test_utils

_DTYPES = {
    tf.dtypes.uint8,
    tf.dtypes.int32,
    tf.dtypes.int64,
    tf.dtypes.float16,
    tf.dtypes.float32,
    tf.dtypes.float64,
}


@test_utils.run_all_in_graph_and_eager_modes
class TranslateOpTest(tf.test.TestCase):
    def test_translate(self):
        for dtype in _DTYPES:
            image = tf.constant(
                [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=dtype
            )
            translation = tf.constant([-1, -1], dtype=tf.float32)
            image_translated = translate_ops.translate(image, translation)
            self.assertAllEqual(
                self.evaluate(image_translated),
                [[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]],
            )

    def test_translations_to_projective_transforms(self):
        translation = tf.constant([-1, -1], dtype=tf.float32)
        transform = translate_ops.translations_to_projective_transforms(translation)
        self.assertAllEqual(self.evaluate(transform), [[1, 0, 1, 0, 1, 1, 0, 0]])


if __name__ == "__main__":
    tf.test.main()
