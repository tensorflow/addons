<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.safe_cumprod" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.seq2seq.safe_cumprod

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/attention_wrapper.py#L810-L831">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Computes cumprod of x in logspace using cumsum to avoid underflow.

**Aliases**: `tfa.seq2seq.attention_wrapper.safe_cumprod`

``` python
tfa.seq2seq.safe_cumprod(
    x,
    *args,
    **kwargs
)
```



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


