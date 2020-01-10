<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.beam_search_decoder.BeamSearchDecoderMixin" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="output_size"/>
<meta itemprop="property" content="tracks_own_finished"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="finalize"/>
<meta itemprop="property" content="step"/>
</div>

# tfa.seq2seq.beam_search_decoder.BeamSearchDecoderMixin

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/beam_search_decoder.py#L222-L572">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
## Class `BeamSearchDecoderMixin`

BeamSearchDecoderMixin contains the common methods for



<!-- Placeholder for "Used in" -->
BeamSearchDecoder.

It is expected to be used a base class for concrete
BeamSearchDecoder. Since this is a mixin class, it is expected to be
used together with other class as base.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/beam_search_decoder.py#L231-L278">View source</a>

``` python
__init__(
    cell,
    beam_width,
    output_layer=None,
    length_penalty_weight=0.0,
    coverage_penalty_weight=0.0,
    reorder_tensor_arrays=True,
    **kwargs
)
```

Initialize the BeamSearchDecoderMixin.


#### Args:


* <b>`cell`</b>: An `RNNCell` instance.
* <b>`beam_width`</b>:  Python integer, the number of beams.
* <b>`output_layer`</b>: (Optional) An instance of `tf.keras.layers.Layer`,
  i.e., `tf.keras.layers.Dense`.  Optional layer to apply to the RNN
  output prior to storing the result or sampling.
* <b>`length_penalty_weight`</b>: Float weight to penalize length. Disabled with
   0.0.
* <b>`coverage_penalty_weight`</b>: Float weight to penalize the coverage of
  source sentence. Disabled with 0.0.
* <b>`reorder_tensor_arrays`</b>: If `True`, `TensorArray`s' elements within the
  cell state will be reordered according to the beam search path. If
  the `TensorArray` can be reordered, the stacked form will be
  returned. Otherwise, the `TensorArray` will be returned as is. Set
  this flag to `False` if the cell state contains `TensorArray`s that
  are not amenable to reordering.
* <b>`**kwargs`</b>: Dict, other keyword arguments for parent class.


#### Raises:


* <b>`TypeError`</b>: if `cell` is not an instance of `RNNCell`,
  or `output_layer` is not an instance of `tf.keras.layers.Layer`.



## Properties

<h3 id="batch_size"><code>batch_size</code></h3>




<h3 id="output_size"><code>output_size</code></h3>




<h3 id="tracks_own_finished"><code>tracks_own_finished</code></h3>

The BeamSearchDecoder shuffles its beams and their finished state.

For this reason, it conflicts with the `dynamic_decode` function's
tracking of finished states.  Setting this property to true avoids
early stopping of decoding due to mismanagement of the finished state
in `dynamic_decode`.

#### Returns:

`True`.




## Methods

<h3 id="finalize"><code>finalize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/beam_search_decoder.py#L324-L358">View source</a>

``` python
finalize(
    outputs,
    final_state,
    sequence_lengths
)
```

Finalize and return the predicted_ids.


#### Args:


* <b>`outputs`</b>: An instance of BeamSearchDecoderOutput.
* <b>`final_state`</b>: An instance of BeamSearchDecoderState. Passed through to
  the output.
* <b>`sequence_lengths`</b>: An `int64` tensor shaped
  `[batch_size, beam_width]`. The sequence lengths determined for
  each beam during decode. **NOTE** These are ignored; the updated
  sequence lengths are stored in `final_state.lengths`.


#### Returns:


* <b>`outputs`</b>: An instance of `FinalBeamSearchDecoderOutput` where the
  predicted_ids are the result of calling _gather_tree.
* <b>`final_state`</b>: The same input instance of `BeamSearchDecoderState`.

<h3 id="step"><code>step</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/beam_search_decoder.py#L514-L572">View source</a>

``` python
step(
    time,
    inputs,
    state,
    training=None,
    name=None
)
```

Perform a decoding step.


#### Args:


* <b>`time`</b>: scalar `int32` tensor.
* <b>`inputs`</b>: A (structure of) input tensors.
* <b>`state`</b>: A (structure of) state tensors and TensorArrays.
* <b>`training`</b>: Python boolean. Indicates whether the layer should
    behave in training mode or in inference mode. Only relevant
    when `dropout` or `recurrent_dropout` is used.
* <b>`name`</b>: Name scope for any created operations.


#### Returns:

`(outputs, next_state, next_inputs, finished)`.






