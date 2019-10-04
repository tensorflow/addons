<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.text.viterbi_decode" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.text.viterbi_decode


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/text/crf.py#L309-L338">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Decode the highest scoring sequence of tags outside of TensorFlow.

### Aliases:

* `tfa.text.crf.viterbi_decode`


``` python
tfa.text.viterbi_decode(
    score,
    transition_params
)
```



<!-- Placeholder for "Used in" -->

This should only be used at test time.

#### Args:


* <b>`score`</b>: A [seq_len, num_tags] matrix of unary potentials.
* <b>`transition_params`</b>: A [num_tags, num_tags] matrix of binary potentials.


#### Returns:


* <b>`viterbi`</b>: A [seq_len] list of integers containing the highest scoring tag
    indices.
* <b>`viterbi_score`</b>: A float containing the score for the Viterbi sequence.