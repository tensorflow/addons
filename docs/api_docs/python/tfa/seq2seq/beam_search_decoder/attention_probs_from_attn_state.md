<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.beam_search_decoder.attention_probs_from_attn_state" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.seq2seq.beam_search_decoder.attention_probs_from_attn_state

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/beam_search_decoder.py#L1056-L1076">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Calculates the average attention probabilities.

``` python
tfa.seq2seq.beam_search_decoder.attention_probs_from_attn_state(attention_state)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`attention_state`</b>: An instance of `AttentionWrapperState`.


#### Returns:

The attention probabilities in the given AttentionWrapperState.
If there're multiple attention mechanisms, return the average value from
all attention mechanisms.


