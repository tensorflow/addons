<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.text.crf_decode_forward" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.text.crf_decode_forward

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/text/crf.py#L399-L418">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Computes forward decoding in a linear-chain CRF.

**Aliases**: `tfa.text.crf.crf_decode_forward`

``` python
tfa.text.crf_decode_forward(
    inputs,
    state,
    transition_params,
    sequence_lengths
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`inputs`</b>: A [batch_size, num_tags] matrix of unary potentials.
* <b>`state`</b>: A [batch_size, num_tags] matrix containing the previous step's
      score values.
* <b>`transition_params`</b>: A [num_tags, num_tags] matrix of binary potentials.
* <b>`sequence_lengths`</b>: A [batch_size] vector of true sequence lengths.


#### Returns:


* <b>`backpointers`</b>: A [batch_size, num_tags] matrix of backpointers.
* <b>`new_state`</b>: A [batch_size, num_tags] matrix of new score values.

