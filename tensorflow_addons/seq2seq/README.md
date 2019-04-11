# Addons - Seq2seq

## Maintainers
| Submodule  |  Maintainers  | Contact Info   |
|:---------- |:----------- |:--------------|
| ALL | Google | @qlzh727 | 


## Contents
| Module | Description                             |
|:----------------------- |:-----------------------------|
| attention_wrapper | Attention related functions and RNN cell wrapper |
| basic_decoder | Basic decoder that does not use beam search |
| beam_search_decoder | Decoder that uses beam search |
| decoder | Base decoders object and functions for user to create customized decoder |
| loss | Sequence loss which can sum/average over batch or timesteps dimention |
| sampler | Objects that work with basic_decoder to provide input for each timestep |

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all objects must:
 * Inherit from proper base class within each module, eg `BaseDecoder` in decoder.py for customized
   decoder or `_BaseAttentionMechanism` for new attentions.
 * [Register as a keras global object](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/utils/python/keras_utils.py)
  so it can be serialized properly.
 * Add the addon to the `py_library` in this sub-package's BUILD file.

#### Testing Requirements
 * Simple unittests that demonstrate the class is behaving as expected on
   some set of known inputs and outputs.
 * When applicable, run all tests with TensorFlow's
   `@run_in_graph_and_eager_modes` (for test method)
   or `run_all_in_graph_and_eager_modes` (for TestCase subclass) decorator.
 * Add a `py_test` to this sub-package's BUILD file.

## Sample code and Migration guide from TF 1.X
The code was originally written in tensorflow.contrib.seq2seq, and has been updated to work with
TF 2.0 API. The API has been reworked to get rid of deprecated TF APIs (eg, using variable
scope to create variable, etc), and also meet the 2.0 API sytle (more object-oriented and use keras
layers). With that, the user side code need to be slightly updated to use the new API. Please see
examples below:

### Basic Decoder
``` python
# TF 1.x, old style

# Build RNN cell
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

# Run Dynamic RNN
#   encoder_outputs: [max_time, batch_size, num_units]
#   encoder_state: [batch_size, num_units]
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_emb_inp,
    sequence_length=source_sequence_length)

# Build RNN cell
decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

# Helper
helper = tf.contrib.seq2seq.TrainingHelper(
    decoder_emb_inp, decoder_lengths)
# Decoder
projection_layer = tf.keras.layers.Dense(num_outputs)
decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, encoder_state,
    output_layer=projection_layer)
# Dynamic decoding
outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
logits = outputs.rnn_output
```

``` python
import tensorflow_addons as tfa

# TF 2.0, new style

# Build RNN
#   encoder_outputs: [max_time, batch_size, num_units]
#   encoder_state: [batch_size, num_units]
encoder = tf.keras.layers.LSTM(num_units, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_emb_inp)
encoder_state = [state_h, state_c]

# Sampler
sampler = tfa.seq2seq.sampler.TrainingSampler()

# Decoder
decoder_cell = tf.keras.layers.LSTMCell(num_units)
projection_layer = tf.keras.layers.Dense(num_outputs)
decoder = tfa.seq2seq.basic_decoder.BasicDecoder(
    decoder_cell, sampler, output_layer=projection_layer)

outputs, _, _ = decoder(
    decoder_emb_inp,
    initial_state=encoder_state,
    sequence_length=decoder_lengths)
logits = outputs.rnn_output
```

Note that the major difference here are:

1. Both encoder and decoder are objects now, and all the metadata can be accessed, eg,
   `encoder.weights`, etc.
1. All the tensor inputs are fed to encoder/decoder by calling it, instead of feeding when constructing
   the instance. This allows the same instance to be reused if needed, just call it with other input
   tensors.
1. Helper has been renamed to Sampler since this better describes its behavior/usage. There is a
   one-to-one mapping between existing Helper and new Sampler. Sampler is also a Keras layer, which
   takes input tensors at `call()` instead of `__init__()`.


### Attention
``` python
# TF 1.x, old style
attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    num_units,
    encoder_state,
    memory_sequence_length=encoder_sequence_length)

decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    decoder_cell, attention_mechanism,
    attention_layer_size=num_units)
```

``` python
import tensorflow_addons as tfa
# TF 2.0, new style
attention_mechanism = tfa.seq2seq.attention_wrapper.LuongAttention(
    num_units,
    encoder_state,
    memory_sequence_length=encoder_sequence_length)

decoder_cell = tfa.seq2seq.attention_wrapper.AttentionWrapper(
    decoder_cell, attention_mechanism,
    attention_layer_size=num_units)
```

1. The `attention_mechanism` here is also a Keras `layer`, we customized it so that it will take
   the memory (encoder_state) during `__init__()`, since the memory of the attention shouldn't be
   changed.

### Beam Search
``` python
# TF 1.x, old style
# Replicate encoder infos beam_width times
decoder_initial_state = tf.contrib.seq2seq.tile_batch(
    encoder_state, multiplier=hparams.beam_width)

# Define a beam-search decoder
decoder = tf.contrib.seq2seq.BeamSearchDecoder(
    cell=decoder_cell,
    embedding=embedding_decoder,
    start_tokens=start_tokens,
    end_token=end_token,
    initial_state=decoder_initial_state,
    beam_width=beam_width,
    output_layer=projection_layer,
    length_penalty_weight=0.0,
    coverage_penalty_weight=0.0)

# Dynamic decoding
outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
```

``` python
# TF 2.0, new style
import tensorflow_addons as tfa

# Replicate encoder infos beam_width times
decoder_initial_state = tfa.seq2seq.beam_search_decoder.tile_batch(
    encoder_state, multiplier=hparams.beam_width)

# Define a beam-search decoder
decoder = tfa.seq2seq.beam_search_decoder.BeamSearchDecoder(
    cell=decoder_cell,
    beam_width=beam_width,
    output_layer=projection_layer,
    length_penalty_weight=0.0,
    coverage_penalty_weight=0.0)

# decoding
outputs, _ = decoder(
    embedding_decoder,
    start_tokens=start_tokens,
    end_token=end_token,
    initial_state=decoder_initial_state)
```
