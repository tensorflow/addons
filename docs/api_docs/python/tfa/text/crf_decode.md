<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.text.crf_decode" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.text.crf_decode


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/text/crf.py#L432-L489">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Decode the highest scoring sequence of tags in TensorFlow.

### Aliases:

* `tfa.text.crf.crf_decode`


``` python
tfa.text.crf_decode(
    potentials,
    transition_params,
    sequence_length
)
```



<!-- Placeholder for "Used in" -->

This is a function for tensor.

#### Args:


* <b>`potentials`</b>: A [batch_size, max_seq_len, num_tags] tensor of
          unary potentials.
* <b>`transition_params`</b>: A [num_tags, num_tags] matrix of
          binary potentials.
* <b>`sequence_length`</b>: A [batch_size] vector of true sequence lengths.


#### Returns:


* <b>`decode_tags`</b>: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
            Contains the highest scoring tag indices.
* <b>`best_score`</b>: A [batch_size] vector, containing the score of `decode_tags`.