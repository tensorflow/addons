<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.text.crf_forward" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.text.crf_forward


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/text/crf.py#L273-L306">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes the alpha values in a linear-chain CRF.

### Aliases:

* `tfa.text.crf.crf_forward`


``` python
tfa.text.crf_forward(
    inputs,
    state,
    transition_params,
    sequence_lengths
)
```



<!-- Placeholder for "Used in" -->

See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.

#### Args:


* <b>`inputs`</b>: A [batch_size, num_tags] matrix of unary potentials.
* <b>`state`</b>: A [batch_size, num_tags] matrix containing the previous alpha
   values.
* <b>`transition_params`</b>: A [num_tags, num_tags] matrix of binary potentials.
    This matrix is expanded into a [1, num_tags, num_tags] in preparation
    for the broadcast summation occurring within the cell.
* <b>`sequence_lengths`</b>: A [batch_size] vector of true sequence lengths.


#### Returns:


* <b>`new_alphas`</b>: A [batch_size, num_tags] matrix containing the
    new alpha values.