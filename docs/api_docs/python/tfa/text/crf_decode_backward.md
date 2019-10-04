<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.text.crf_decode_backward" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.text.crf_decode_backward


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/text/crf.py#L409-L429">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes backward decoding in a linear-chain CRF.

### Aliases:

* `tfa.text.crf.crf_decode_backward`


``` python
tfa.text.crf_decode_backward(
    inputs,
    state
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`inputs`</b>: A [batch_size, num_tags] matrix of
      backpointer of next step (in time order).
* <b>`state`</b>: A [batch_size, 1] matrix of tag index of next step.


#### Returns:


* <b>`new_tags`</b>: A [batch_size, num_tags]
  tensor containing the new tag indices.