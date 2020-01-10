<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.text.crf_unary_score" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.text.crf_unary_score

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/text/crf.py#L207-L241">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Computes the unary scores of tag sequences.

**Aliases**: `tfa.text.crf.crf_unary_score`

``` python
tfa.text.crf_unary_score(
    tag_indices,
    sequence_lengths,
    inputs
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`tag_indices`</b>: A [batch_size, max_seq_len] matrix of tag indices.
* <b>`sequence_lengths`</b>: A [batch_size] vector of true sequence lengths.
* <b>`inputs`</b>: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.

#### Returns:


* <b>`unary_scores`</b>: A [batch_size] vector of unary scores.

