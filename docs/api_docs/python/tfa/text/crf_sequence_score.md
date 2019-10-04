<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.text.crf_sequence_score" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.text.crf_sequence_score


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/text/crf.py#L27-L68">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes the unnormalized score for a tag sequence.

### Aliases:

* `tfa.text.crf.crf_sequence_score`


``` python
tfa.text.crf_sequence_score(
    inputs,
    tag_indices,
    sequence_lengths,
    transition_params
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`inputs`</b>: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
    to use as input to the CRF layer.
* <b>`tag_indices`</b>: A [batch_size, max_seq_len] matrix of tag indices for which
    we compute the unnormalized score.
* <b>`sequence_lengths`</b>: A [batch_size] vector of true sequence lengths.
* <b>`transition_params`</b>: A [num_tags, num_tags] transition matrix.

#### Returns:


* <b>`sequence_scores`</b>: A [batch_size] vector of unnormalized sequence scores.