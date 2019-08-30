<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.hardmax" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.seq2seq.hardmax


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/seq2seq/attention_wrapper.py#L1473-L1490">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Returns batched one-hot vectors.

### Aliases:

* `tfa.seq2seq.attention_wrapper.hardmax`


``` python
tfa.seq2seq.hardmax(
    logits,
    name=None
)
```



<!-- Placeholder for "Used in" -->

The depth index containing the `1` is that of the maximum logit value.

#### Args:


* <b>`logits`</b>: A batch tensor of logit values.
* <b>`name`</b>: Name to use when creating ops.

#### Returns:

A batched one-hot tensor.
