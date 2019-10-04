<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.text.crf_binary_score" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.text.crf_binary_score


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/text/crf.py#L238-L270">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes the binary scores of tag sequences.

### Aliases:

* `tfa.text.crf.crf_binary_score`


``` python
tfa.text.crf_binary_score(
    tag_indices,
    sequence_lengths,
    transition_params
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`tag_indices`</b>: A [batch_size, max_seq_len] matrix of tag indices.
* <b>`sequence_lengths`</b>: A [batch_size] vector of true sequence lengths.
* <b>`transition_params`</b>: A [num_tags, num_tags] matrix of binary potentials.

#### Returns:


* <b>`binary_scores`</b>: A [batch_size] vector of binary scores.