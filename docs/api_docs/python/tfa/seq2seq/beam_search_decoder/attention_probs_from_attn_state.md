<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.beam_search_decoder.attention_probs_from_attn_state" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.seq2seq.beam_search_decoder.attention_probs_from_attn_state

Calculates the average attention probabilities.

``` python
tfa.seq2seq.beam_search_decoder.attention_probs_from_attn_state(attention_state)
```



Defined in [`seq2seq/beam_search_decoder.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/seq2seq/beam_search_decoder.py).

<!-- Placeholder for "Used in" -->

#### Args:

* <b>`attention_state`</b>: An instance of `AttentionWrapperState`.


#### Returns:

The attention probabilities in the given AttentionWrapperState.
If there're multiple attention mechanisms, return the average value from
all attention mechanisms.