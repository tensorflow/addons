<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.gather_tree_from_array" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.seq2seq.gather_tree_from_array


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/seq2seq/beam_search_decoder.py#L113-L167">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Calculates the full beams for `TensorArray`s.

### Aliases:

* `tfa.seq2seq.beam_search_decoder.gather_tree_from_array`


``` python
tfa.seq2seq.gather_tree_from_array(
    t,
    parent_ids,
    sequence_length
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`t`</b>: A stacked `TensorArray` of size `max_time` that contains `Tensor`s of
  shape `[batch_size, beam_width, s]` or `[batch_size * beam_width, s]`
  where `s` is the depth shape.
* <b>`parent_ids`</b>: The parent ids of shape `[max_time, batch_size, beam_width]`.
* <b>`sequence_length`</b>: The sequence length of shape `[batch_size, beam_width]`.


#### Returns:

A `Tensor` which is a stacked `TensorArray` of the same size and type as
`t` and where beams are sorted in each `Tensor` according to
`parent_ids`.
