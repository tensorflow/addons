<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.dynamic_decode" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.seq2seq.dynamic_decode


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/seq2seq/decoder.py#L272-L492">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Perform dynamic decoding with `decoder`.

### Aliases:

* `tfa.seq2seq.decoder.dynamic_decode`


``` python
tfa.seq2seq.dynamic_decode(
    decoder,
    output_time_major=False,
    impute_finished=False,
    maximum_iterations=None,
    parallel_iterations=32,
    swap_memory=False,
    training=None,
    scope=None,
    **kwargs
)
```



<!-- Placeholder for "Used in" -->

Calls initialize() once and step() repeatedly on the Decoder object.

#### Args:


* <b>`decoder`</b>: A `Decoder` instance.
* <b>`output_time_major`</b>: Python boolean.  Default: `False` (batch major). If
  `True`, outputs are returned as time major tensors (this mode is
  faster). Otherwise, outputs are returned as batch major tensors (this
  adds extra time to the computation).
* <b>`impute_finished`</b>: Python boolean.  If `True`, then states for batch
  entries which are marked as finished get copied through and the
  corresponding outputs get zeroed out.  This causes some slowdown at
  each time step, but ensures that the final state and outputs have
  the correct values and that backprop ignores time steps that were
  marked as finished.
* <b>`maximum_iterations`</b>: `int32` scalar, maximum allowed number of decoding
   steps.  Default is `None` (decode until the decoder is fully done).
* <b>`parallel_iterations`</b>: Argument passed to `tf.while_loop`.
* <b>`swap_memory`</b>: Argument passed to `tf.while_loop`.
* <b>`training`</b>: Python boolean. Indicates whether the layer should behave
    in training  mode or in inference mode. Only relevant
    when `dropout` or `recurrent_dropout` is used.
* <b>`scope`</b>: Optional variable scope to use.
* <b>`**kwargs`</b>: dict, other keyword arguments for dynamic_decode. It might
  contain arguments for `BaseDecoder` to initialize, which takes all
  tensor inputs during call().


#### Returns:

`(final_outputs, final_state, final_sequence_lengths)`.



#### Raises:


* <b>`TypeError`</b>: if `decoder` is not an instance of `Decoder`.
* <b>`ValueError`</b>: if `maximum_iterations` is provided but is not a scalar.