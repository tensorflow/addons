# Addons - Seq2seq

## Contents
https://www.tensorflow.org/addons/api_docs/python/tfa/seq2seq

## Contribution Guidelines
#### Standard API
In order to conform with the current API standard, all objects must:
 * Inherit from proper base class within each module, eg `BaseDecoder` in decoder.py for customized
   decoder or `_BaseAttentionMechanism` for new attentions.
 * Register as a keras global object so it can be serialized properly: `@tf.keras.utils.register_keras_serializable(package='Addons')`

#### Testing Requirements
 * Simple unittests that demonstrate the class is behaving as expected on
   some set of known inputs and outputs.
 * To run your `tf.functions` in eager mode and graph mode in the tests, 
   you can use the `@pytest.mark.usefixtures("maybe_run_functions_eagerly")` 
   decorator. This will run the tests twice, once normally, and once
   with `tf.config.run_functions_eagerly(True)`.

## Sample code and Migration guide from TF 1.X
The code was originally written in tensorflow.contrib.seq2seq, and has been updated to work with
TF 2.0 API. The API has been reworked to get rid of deprecated TF APIs (eg, using variable
scope to create variable, etc), and also meet the 2.0 API sytle (more object-oriented and use keras
layers). With that, the user side code need to be slightly updated to use the new API. Please see
examples below:

### Decoder with attention

``` python
# TF 1.x, old style

# Encoder
encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell,
    encoder_inputs,
    sequence_length=encoder_lengths,
    dtype=encoder_inputs.dtype,
)

# Decoder RNN cell with attention
attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    num_units, encoder_outputs, memory_sequence_length=encoder_lengths
)
decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units)
decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    decoder_cell,
    attention_mechanism,
    attention_layer_size=num_units,
    initial_cell_state=encoder_state,
)

# Helper
helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, decoder_lengths)

# Decoder
projection_layer = tf.layers.Dense(num_outputs)
decoder_initial_state = decoder_cell.get_initial_state(inputs=decoder_inputs)
decoder = tf.contrib.seq2seq.BasicDecoder(
    decoder_cell, helper, decoder_initial_state, output_layer=projection_layer
)

# Dynamic decoding
outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
logits = outputs.rnn_output
```

``` python
import tensorflow_addons as tfa

# TF 2.0, new style

# Encoder
encoder = tf.keras.layers.LSTM(num_units, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder(
    encoder_inputs, mask=tf.sequence_mask(encoder_lengths), training=True
)
encoder_state = (state_h, state_c)

# Decoder RNN cell with attention
attention_mechanism = tfa.seq2seq.LuongAttention(num_units, encoder_outputs)
decoder_cell = tf.keras.layers.LSTMCell(num_units)
decoder_cell = tfa.seq2seq.AttentionWrapper(
    decoder_cell,
    attention_mechanism,
    attention_layer_size=num_units,
    initial_cell_state=encoder_state,
)

# Sampler
sampler = tfa.seq2seq.sampler.TrainingSampler()

# Decoder
projection_layer = tf.keras.layers.Dense(num_outputs)
decoder = tfa.seq2seq.BasicDecoder(decoder_cell, sampler, output_layer=projection_layer)

# Dynamic decoding
decoder_initial_state = decoder_cell.get_initial_state(inputs=decoder_inputs)
outputs, _, _ = decoder(
    decoder_inputs,
    initial_state=decoder_initial_state,
    sequence_length=decoder_lengths,
    training=True,
)
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
1. The `attention_mechanism` here is also a Keras layer, we customized it so that it will take
   the memory (encoder_outputs) during `__init__()`, since the memory of the attention shouldn't be
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
decoder_initial_state = tfa.seq2seq.tile_batch(
    encoder_state, multiplier=hparams.beam_width)

# Define a beam-search decoder
decoder = tfa.seq2seq.BeamSearchDecoder(
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
