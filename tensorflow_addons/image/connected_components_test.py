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
"""Tests for connected component analysis."""

import sys

import pytest
import logging
import tensorflow as tf
import numpy as np

from tensorflow_addons.image.connected_components import connected_components
from tensorflow_addons.utils import test_utils

# Image for testing connected_components, with a single, winding component.
SNAKE = np.asarray(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)


@test_utils.run_all_in_graph_and_eager_modes
class ConnectedComponentsTest(tf.test.TestCase):
    def testDisconnected(self):
        arr = tf.cast(
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
            ],
            tf.bool,
        )
        expected = [
            [1, 0, 0, 2, 0, 0, 0, 0, 3],
            [0, 4, 0, 0, 0, 5, 0, 6, 0],
            [7, 0, 8, 0, 0, 0, 9, 0, 0],
            [0, 0, 0, 0, 10, 0, 0, 0, 0],
            [0, 0, 11, 0, 0, 0, 0, 0, 0],
        ]
        self.assertAllEqual(self.evaluate(connected_components(arr)), expected)

    def testSimple(self):
        arr = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

        # Single component with id 1.
        self.assertAllEqual(
            self.evaluate(connected_components(tf.cast(arr, tf.bool))), arr
        )

    def testSnake(self):

        # Single component with id 1.
        self.assertAllEqual(
            self.evaluate(connected_components(tf.cast(SNAKE, tf.bool))), SNAKE
        )

    def testSnake_disconnected(self):
        for i in range(SNAKE.shape[0]):
            for j in range(SNAKE.shape[1]):

                # If we disconnect any part of the snake except for the endpoints,
                # there will be 2 components.
                if SNAKE[i, j] and (i, j) not in [(1, 1), (6, 3)]:
                    disconnected_snake = SNAKE.copy()
                    disconnected_snake[i, j] = 0
                    components = self.evaluate(
                        connected_components(tf.cast(disconnected_snake, tf.bool))
                    )
                    self.assertEqual(
                        components.max(), 2, "disconnect (%d, %d)" % (i, j)
                    )
                    bins = np.bincount(components.ravel())
                    # Nonzero number of pixels labeled 0, 1, or 2.
                    self.assertGreater(bins[0], 0)
                    self.assertGreater(bins[1], 0)
                    self.assertGreater(bins[2], 0)

    def testMultipleImages(self):
        images = [
            [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
            [[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]],
            [[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1]],
        ]
        expected = [
            [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]],
            [[2, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 5]],
            [[6, 6, 0, 7], [0, 6, 6, 0], [8, 0, 6, 0], [0, 0, 6, 6]],
        ]

        self.assertAllEqual(
            self.evaluate(connected_components(tf.cast(images, tf.bool))), expected
        )

    def testZeros(self):

        self.assertAllEqual(
            connected_components(self.evaluate(tf.zeros((100, 20, 50), tf.bool))),
            np.zeros((100, 20, 50)),
        )

    def testOnes(self):

        self.assertAllEqual(
            self.evaluate(connected_components(tf.ones((100, 20, 50), tf.bool))),
            np.tile(np.arange(100)[:, None, None] + 1, [1, 20, 50]),
        )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_ones_small():

    np.testing.assert_equal(
        connected_components(tf.ones((3, 5), tf.bool)).numpy(), np.ones((3, 5)),
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_random_scipy():
    np.random.seed(42)
    images = np.random.randint(0, 2, size=(10, 100, 200)).astype(np.bool)
    expected = connected_components_reference_implementation(images)
    if expected is None:
        return

    np.testing.assert_equal(connected_components(images).numpy(), expected)


def connected_components_reference_implementation(images):
    try:
        from scipy.ndimage import measurements
    except ImportError:
        logging.exception("Skipping test method because scipy could not be loaded")
        return
    image_or_images = np.asarray(images)
    if len(image_or_images.shape) == 2:
        images = image_or_images[None, :, :]
    elif len(image_or_images.shape) == 3:
        images = image_or_images
    components = np.asarray([measurements.label(image)[0] for image in images])
    # Get the count of nonzero ids for each image, and offset each image's nonzero
    # ids using the cumulative sum.
    num_ids_per_image = components.reshape(
        [-1, components.shape[1] * components.shape[2]]
    ).max(axis=-1)
    positive_id_start_per_image = np.cumsum(num_ids_per_image)
    for i in range(components.shape[0]):
        new_id_start = positive_id_start_per_image[i - 1] if i > 0 else 0
        components[i, components[i] > 0] += new_id_start
    if len(image_or_images.shape) == 2:
        return components[0, :, :]
    else:
        return components


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
