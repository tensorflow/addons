# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tfa.seq2seq.beam_search_ops."""

import itertools

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.seq2seq import gather_tree


def _transpose_batch_time(x):
    return np.transpose(x, [1, 0, 2]).astype(np.int32)


@pytest.mark.usefixtures("run_custom_and_py_ops")
def test_gather_tree_one():
    # (max_time = 4, batch_size = 1, beams = 3)
    end_token = 10
    step_ids = _transpose_batch_time([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -1, -1]]])
    parent_ids = _transpose_batch_time(
        [[[0, 0, 0], [0, 1, 1], [2, 1, 2], [-1, -1, -1]]]
    )
    max_sequence_lengths = [3]
    expected_result = _transpose_batch_time(
        [[[2, 2, 2], [6, 5, 6], [7, 8, 9], [10, 10, 10]]]
    )
    beams = gather_tree(
        step_ids=step_ids,
        parent_ids=parent_ids,
        max_sequence_lengths=max_sequence_lengths,
        end_token=end_token,
    )
    np.testing.assert_equal(expected_result, beams.numpy())


@pytest.mark.usefixtures("run_custom_and_py_ops")
def test_bad_parent_values_on_cpu():
    # (batch_size = 1, max_time = 4, beams = 3)
    # bad parent in beam 1 time 1
    end_token = 10
    step_ids = _transpose_batch_time([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -1, -1]]])
    parent_ids = _transpose_batch_time(
        [[[0, 0, 0], [0, -1, 1], [2, 1, 2], [-1, -1, -1]]]
    )
    max_sequence_lengths = [3]

    with pytest.raises(tf.errors.InvalidArgumentError, match="parent id"):
        _ = gather_tree(
            step_ids=step_ids,
            parent_ids=parent_ids,
            max_sequence_lengths=max_sequence_lengths,
            end_token=end_token,
        )


@pytest.mark.with_device(["gpu"])
@pytest.mark.usefixtures("run_custom_and_py_ops")
def test_bad_parent_values_on_gpu():
    # (max_time = 4, batch_size = 1, beams = 3)
    # bad parent in beam 1 time 1; appears as a negative index at time 0
    end_token = 10
    step_ids = _transpose_batch_time([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [-1, -1, -1]]])
    parent_ids = _transpose_batch_time(
        [[[0, 0, 0], [0, -1, 1], [2, 1, 2], [-1, -1, -1]]]
    )
    max_sequence_lengths = [3]

    with pytest.raises(tf.errors.InvalidArgumentError, match="parent id"):
        _ = gather_tree(
            step_ids=step_ids,
            parent_ids=parent_ids,
            max_sequence_lengths=max_sequence_lengths,
            end_token=end_token,
        )


@pytest.mark.usefixtures("run_custom_and_py_ops")
def test_gather_tree_batch():
    batch_size = 10
    beam_width = 15
    max_time = 8
    max_sequence_lengths = [0, 1, 2, 4, 7, 8, 9, 10, 11, 0]
    end_token = 5

    step_ids = np.random.randint(
        0, high=end_token + 1, size=(max_time, batch_size, beam_width)
    )
    parent_ids = np.random.randint(
        0, high=beam_width - 1, size=(max_time, batch_size, beam_width)
    )

    beams = gather_tree(
        step_ids=step_ids.astype(np.int32),
        parent_ids=parent_ids.astype(np.int32),
        max_sequence_lengths=max_sequence_lengths,
        end_token=end_token,
    )
    beams = beams.numpy()

    assert (max_time, batch_size, beam_width) == beams.shape
    for b in range(batch_size):
        # Past max_sequence_lengths[b], we emit all end tokens.
        b_value = beams[max_sequence_lengths[b] :, b, :]
        np.testing.assert_allclose(b_value, end_token * np.ones_like(b_value))
    for batch, beam in itertools.product(range(batch_size), range(beam_width)):
        v = np.squeeze(beams[:, batch, beam])
        if end_token in v:
            found_bad = np.where(v == -1)[0]
            assert 0 == len(found_bad)
            found = np.where(v == end_token)[0]
            found = found[0]  # First occurrence of end_token.
            # If an end_token is found, everything before it should be a
            # valid id and everything after it should be end_token.
            if found > 0:
                np.testing.assert_equal(
                    v[: found - 1] >= 0, np.ones_like(v[: found - 1], dtype=bool)
                )
            np.testing.assert_allclose(
                v[found + 1 :], end_token * np.ones_like(v[found + 1 :])
            )
