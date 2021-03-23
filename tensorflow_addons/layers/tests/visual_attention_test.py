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
# ==============================================================================
"""Tests for """
from tensorflow_addons.layers.visual_attention import PixelAttention2D, ChannelAttention2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D
import tensorflow as tf
import pytest
from tensorflow_addons.utils import test_utils


def pixel_attention_test():
    inp = Input(shape=[None,None,1])
    conv1 = Conv2D(16,3,padding="same")(inp)
    pa = PixelAttention2D(16)(conv1)
    model = Model(inputs=inp,outputs=pa)
    return model

def channel_attention_test():
    inp = Input(shape=[None,None,1])
    conv1 = Conv2D(16,3,padding="same")(inp)
    ca = ChannelAttention2D(16)(conv1)
    model = Model(inputs=inp,outputs=ca)
    return model
    
def tests():
    image = tf.constant([[  0,   0,   0,   0,   0,   0,   4,   0],
                      [  0,   0,  31, 147, 179,  82,   2,   0],
                      [  1,   0,  48, 174, 111, 206,  58,   0],
                      [  1,   0,   0,   0,   0, 179,  85,   0],
                      [  1,  58, 128, 174, 186, 234,  64,   0],
                      [  5, 192, 232, 206, 164, 112, 151,   2],
                      [  3,  30,  61,  30,   0,   0,  11,   1],
                      [  0,   0,   0,   0,   0,   0,   0,   0]])
    #Batch and Channel`
    image = tf.expand_dims(image,-1)
    image = tf.expand_dims(image,0)
    print("input shape = ",image.shape)
    model_pa = pixel_attention_test()
    output = model_pa.predict(image)
    print("output shape pixel attention = ",output.shape)
    del model_pa
    model_ca = channel_attention_test()
    output = model_ca.predict(image)
    print("output shape channel attention = ",output.shape)
    del model_ca