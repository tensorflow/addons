<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.beam_search_decoder.get_attention_probs" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.seq2seq.beam_search_decoder.get_attention_probs

Get attention probabilities from the cell state.

``` python
tfa.seq2seq.beam_search_decoder.get_attention_probs(
    next_cell_state,
    coverage_penalty_weight
)
```



Defined in [`seq2seq/beam_search_decoder.py`](https://github.com/tensorflow/addons/tree/0.4-release/tensorflow_addons/seq2seq/beam_search_decoder.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`next_cell_state`</b>: The next state from the cell, e.g. an instance of
  AttentionWrapperState if the cell is attentional.
* <b>`coverage_penalty_weight`</b>: Float weight to penalize the coverage of source
  sentence. Disabled with 0.0.


#### Returns:

The attention probabilities with shape
  `[batch_size, beam_width, max_time]` if coverage penalty is enabled.
  Otherwise, returns None.



#### Raises:


* <b>`ValueError`</b>: If no cell is attentional but coverage penalty is enabled.