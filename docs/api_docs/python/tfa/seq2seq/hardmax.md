<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.hardmax" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.seq2seq.hardmax

Returns batched one-hot vectors.

### Aliases:

* `tfa.seq2seq.attention_wrapper.hardmax`
* `tfa.seq2seq.hardmax`

``` python
tfa.seq2seq.hardmax(
    logits,
    name=None
)
```



Defined in [`seq2seq/attention_wrapper.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/seq2seq/attention_wrapper.py).

<!-- Placeholder for "Used in" -->

The depth index containing the `1` is that of the maximum logit value.

#### Args:

* <b>`logits`</b>: A batch tensor of logit values.
* <b>`name`</b>: Name to use when creating ops.

#### Returns:

A batched one-hot tensor.