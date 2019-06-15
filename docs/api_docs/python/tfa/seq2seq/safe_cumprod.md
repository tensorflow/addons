<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.safe_cumprod" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.seq2seq.safe_cumprod

Computes cumprod of x in logspace using cumsum to avoid underflow.

### Aliases:

* `tfa.seq2seq.attention_wrapper.safe_cumprod`
* `tfa.seq2seq.safe_cumprod`

``` python
tfa.seq2seq.safe_cumprod(
    x,
    *args,
    **kwargs
)
```



Defined in [`seq2seq/attention_wrapper.py`](https://github.com/tensorflow/addons/tree/0.4-release/tensorflow_addons/seq2seq/attention_wrapper.py).

<!-- Placeholder for "Used in" -->

The cumprod function and its gradient can result in numerical instabilities
when its argument has very small and/or zero values.  As long as the
argument is all positive, we can instead compute the cumulative product as
exp(cumsum(log(x))).  This function can be called identically to
tf.cumprod.

#### Args:


* <b>`x`</b>: Tensor to take the cumulative product of.
* <b>`*args`</b>: Passed on to cumsum; these are identical to those in cumprod.
* <b>`**kwargs`</b>: Passed on to cumsum; these are identical to those in cumprod.

#### Returns:

Cumulative product of x.
