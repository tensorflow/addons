# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from tensorflow_addons.layers.multihead_attention import MultiHeadAttention


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_output_size():
    batch_size = 10
    num_heads = 8
    head_size = 12
    output_size = 20

    q = tf.random.uniform((batch_size, 5, 9), dtype=np.float32)
    k = tf.random.uniform((batch_size, 7, 11), dtype=np.float32)
    v = tf.random.uniform((batch_size, 7, 13), dtype=np.float32)

    mha = MultiHeadAttention(
        head_size=head_size, num_heads=num_heads, output_size=output_size
    )

    output = mha([q, k, v])

    assert output.shape[0] == batch_size
    assert output.shape[1] == q.shape[1]
    assert output.shape[2] == output_size


def test_output_shape():
    batch_size = 10
    num_heads = 8
    head_size = 12

    q = tf.random.uniform((batch_size, 5, 9), dtype=np.float32)
    k = tf.random.uniform((batch_size, 7, 11), dtype=np.float32)
    v = tf.random.uniform((batch_size, 7, 13), dtype=np.float32)

    mha = MultiHeadAttention(head_size=head_size, num_heads=num_heads)

    output = mha([q, k, v])

    assert output.shape[0] == batch_size
    assert output.shape[1] == q.shape[1]
    assert output.shape[2] == v.shape[2]


def test_no_batch():
    num_heads = 8
    head_size = 12

    q = tf.random.uniform((5, 9), dtype=np.float32)
    k = tf.random.uniform((7, 11), dtype=np.float32)
    v = tf.random.uniform((7, 13), dtype=np.float32)

    mha = MultiHeadAttention(head_size=head_size, num_heads=num_heads)

    output = mha([q, k, v])

    assert output.shape[0] == q.shape[0]
    assert output.shape[1] == v.shape[1]


def test_extra_dims():
    batch_size = 10
    extra_dim = 17
    num_heads = 8
    head_size = 12

    q = tf.random.uniform((batch_size, extra_dim, 5, 9), dtype=np.float32)
    k = tf.random.uniform((batch_size, extra_dim, 7, 11), dtype=np.float32)
    v = tf.random.uniform((batch_size, extra_dim, 7, 13), dtype=np.float32)

    mha = MultiHeadAttention(head_size=head_size, num_heads=num_heads)

    output = mha([q, k, v])

    assert output.shape[0] == batch_size
    assert output.shape[1] == extra_dim
    assert output.shape[2] == q.shape[2]
    assert output.shape[3] == v.shape[3]


def test_extra_dims_atten_coef():
    batch_size = 10
    extra_dim = 17
    num_heads = 8
    head_size = 12

    q = tf.random.uniform((batch_size, extra_dim, 5, 9), dtype=np.float32)
    k = tf.random.uniform((batch_size, extra_dim, 7, 11), dtype=np.float32)
    v = tf.random.uniform((batch_size, extra_dim, 7, 13), dtype=np.float32)

    mha = MultiHeadAttention(
        head_size=head_size, num_heads=num_heads, return_attn_coef=True
    )

    output, attn_coef = mha([q, k, v])

    assert output.shape[0] == batch_size
    assert output.shape[1] == extra_dim
    assert output.shape[2] == q.shape[2]
    assert output.shape[3] == v.shape[3]

    assert attn_coef.shape[0] == batch_size
    assert attn_coef.shape[1] == extra_dim
    assert attn_coef.shape[2] == num_heads
    assert attn_coef.shape[3] == q.shape[2]
    assert attn_coef.shape[4] == k.shape[2]


def test_attention_coefficients_shape():
    batch_size = 10
    num_heads = 8
    head_size = 12

    q = tf.random.uniform((batch_size, 5, 9), dtype=np.float32)
    k = tf.random.uniform((batch_size, 7, 11), dtype=np.float32)
    v = tf.random.uniform((batch_size, 7, 13), dtype=np.float32)

    mha = MultiHeadAttention(
        head_size=head_size, num_heads=num_heads, return_attn_coef=True
    )

    output, attn_coef = mha([q, k, v])

    assert attn_coef.shape[0] == batch_size
    assert attn_coef.shape[1] == num_heads
    assert attn_coef.shape[2] == q.shape[1]
    assert attn_coef.shape[3] == k.shape[1]

    assert output.shape[1] == q.shape[1]
    assert output.shape[2] == v.shape[2]


def test_compute_output_shape():
    batch_size = 10
    num_heads = 8
    head_size = 12

    mha = MultiHeadAttention(head_size=head_size, num_heads=num_heads)

    output_shape = mha.compute_output_shape(
        [(batch_size, 5, 9), (batch_size, 7, 11), (batch_size, 7, 13)]
    )

    assert output_shape[1] == 5
    assert output_shape[2] == 13


def test_compute_output_shape_return_attn():
    batch_size = 10
    num_heads = 8
    head_size = 12

    mha = MultiHeadAttention(
        head_size=head_size, num_heads=num_heads, return_attn_coef=True
    )

    output_shape, attn_coef_shape = mha.compute_output_shape(
        [(batch_size, 5, 9), (batch_size, 7, 11), (batch_size, 7, 13)]
    )

    assert output_shape[1] == 5
    assert output_shape[2] == 13

    assert attn_coef_shape[0] == batch_size
    assert attn_coef_shape[1] == num_heads
    assert attn_coef_shape[2] == 5
    assert attn_coef_shape[3] == 7


def test_no_value():
    batch_size = 10
    num_heads = 8
    head_size = 12

    q = tf.random.uniform((batch_size, 5, 9), dtype=np.float32)
    k = tf.random.uniform((batch_size, 7, 11), dtype=np.float32)

    mha = MultiHeadAttention(
        head_size=head_size, num_heads=num_heads, return_attn_coef=True
    )

    output, attn_coef = mha([q, k])

    assert attn_coef.shape[0] == batch_size
    assert attn_coef.shape[1] == num_heads
    assert attn_coef.shape[2] == q.shape[1]
    assert attn_coef.shape[3] == k.shape[1]

    assert output.shape[1] == q.shape[1]
    assert output.shape[2] == k.shape[2]


def test_mask_no_batch():
    batch_size = 10
    num_heads = 8
    head_size = 12

    q = tf.random.uniform((batch_size, 5, 9), dtype=np.float32)
    k = tf.random.uniform((batch_size, 7, 11), dtype=np.float32)
    v = tf.random.uniform((batch_size, 7, 13), dtype=np.float32)
    mask = tf.random.uniform((5, 7), dtype=np.float32) > 0.1

    mha = MultiHeadAttention(
        head_size=head_size, num_heads=num_heads, return_attn_coef=True
    )

    output, attn_coef = mha([q, k, v], mask=mask)

    assert output.shape[0] == batch_size
    assert output.shape[1] == q.shape[1]
    assert output.shape[2] == v.shape[2]

    if tf.executing_eagerly():
        attn_coef = attn_coef.numpy()
        mask = mask.numpy()

        assert ((attn_coef != 0) == mask).all()


def test_from_to_config():
    num_heads = 8
    head_size = 12

    mha = MultiHeadAttention(head_size=head_size, num_heads=num_heads, dropout=0.5)

    config = mha.get_config()

    new_mha = MultiHeadAttention.from_config(config)

    assert mha.head_size == new_mha.head_size
    assert mha.num_heads == new_mha.num_heads
    assert mha._droput_rate == new_mha._droput_rate


def test_save_load_model():

    num_heads = 8
    head_size = 12

    inputs = tf.keras.layers.Input(shape=[42, 13])

    net, attn_coef = MultiHeadAttention(
        head_size=head_size, num_heads=num_heads, dropout=0.5, return_attn_coef=True
    )([inputs, inputs, inputs])
    net = tf.keras.layers.GlobalAveragePooling1D()(net)
    net = tf.keras.layers.Dense(10, activation="softmax")(net)

    model = tf.keras.Model(inputs=inputs, outputs=[net, attn_coef])

    # initialize model
    model.predict(np.random.uniform(size=(10, 42, 13)))

    with tempfile.TemporaryDirectory() as model_dir:
        model_path = str(Path(model_dir) / "saved_model")
        model.save(model_path)
        new_model = tf.keras.models.load_model(model_path)

    assert model.layers[1].get_config() == new_model.layers[1].get_config()


@pytest.mark.usefixtures("run_with_mixed_precision_policy")
def test_fit_predict_eval():

    num_heads = 8
    head_size = 12

    inputs = tf.keras.layers.Input(shape=[42, 13])

    net = MultiHeadAttention(head_size=head_size, num_heads=num_heads, dropout=0.5)(
        [inputs, inputs, inputs]
    )
    net = tf.keras.layers.GlobalAveragePooling1D()(net)
    net = tf.keras.layers.Dense(10, activation="softmax")(net)

    model = tf.keras.Model(inputs=inputs, outputs=net)

    model.compile(
        loss=tf.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001),
    )

    model.fit(
        x=np.random.uniform(size=(50, 42, 13)),
        y=np.random.randint(10, size=(50,)),
        batch_size=10,
        epochs=2,
    )

    model.predict(np.random.uniform(size=(10, 42, 13)))

    model.evaluate(
        x=np.random.uniform(size=(20, 42, 13)),
        y=np.random.randint(0, 10, size=(20,)),
        batch_size=10,
    )


@pytest.mark.usefixtures("maybe_run_functions_eagerly")
def test_mask():
    batch_size = 10
    num_heads = 8
    head_size = 12

    q = tf.random.uniform((batch_size, 5, 9), dtype=np.float32)
    k = tf.random.uniform((batch_size, 7, 11), dtype=np.float32)
    v = tf.random.uniform((batch_size, 7, 13), dtype=np.float32)
    mask = tf.random.uniform((batch_size, num_heads, 5, 7), dtype=np.float32) > 0.1

    mha = MultiHeadAttention(
        head_size=head_size, num_heads=num_heads, return_attn_coef=True
    )

    output, attn_coef = mha([q, k, v], mask=mask)

    assert attn_coef.shape[0] == batch_size
    assert attn_coef.shape[1] == num_heads
    assert attn_coef.shape[2] == q.shape[1]
    assert attn_coef.shape[3] == k.shape[1]

    assert output.shape[0] == batch_size
    assert output.shape[1] == q.shape[1]
    assert output.shape[2] == v.shape[2]

    np.testing.assert_array_equal((attn_coef != 0), mask)
