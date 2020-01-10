<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.text.crf_log_likelihood" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.text.crf_log_likelihood

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/text/crf.py#L167-L204">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Computes the log-likelihood of tag sequences in a CRF.

**Aliases**: `tfa.text.crf.crf_log_likelihood`

``` python
tfa.text.crf_log_likelihood(
    inputs,
    tag_indices,
    sequence_lengths,
    transition_params=None
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`inputs`</b>: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
    to use as input to the CRF layer.
* <b>`tag_indices`</b>: A [batch_size, max_seq_len] matrix of tag indices for which
    we compute the log-likelihood.
* <b>`sequence_lengths`</b>: A [batch_size] vector of true sequence lengths.
* <b>`transition_params`</b>: A [num_tags, num_tags] transition matrix,
    if available.

#### Returns:


* <b>`log_likelihood`</b>: A [batch_size] `Tensor` containing the log-likelihood of
  each example, given the sequence of tag indices.
* <b>`transition_params`</b>: A [num_tags, num_tags] transition matrix. This is
    either provided by the caller or created in this function.

