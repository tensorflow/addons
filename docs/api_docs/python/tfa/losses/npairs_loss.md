<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.npairs_loss" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.losses.npairs_loss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/losses/npairs.py#L23-L63">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Computes the npairs loss between `y_true` and `y_pred`.

**Aliases**: `tfa.losses.npairs.npairs_loss`

``` python
tfa.losses.npairs_loss(
    y_true,
    y_pred
)
```



<!-- Placeholder for "Used in" -->

Npairs loss expects paired data where a pair is composed of samples from
the same labels and each pairs in the minibatch have different labels.
The loss takes each row of the pair-wise similarity matrix, `y_pred`,
as logits and the remapped multi-class labels, `y_true`, as labels.

The similarity matrix `y_pred` between two embedding matrices `a` and `b`
with shape `[batch_size, hidden_size]` can be computed as follows:

```python
# y_pred = a * b^T
y_pred = tf.matmul(a, b, transpose_a=False, transpose_b=True)
```

See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

#### Args:


* <b>`y_true`</b>: 1-D integer `Tensor` with shape `[batch_size]` of
  multi-class labels.
* <b>`y_pred`</b>: 2-D float `Tensor` with shape `[batch_size, batch_size]` of
  similarity matrix between embedding matrices.


#### Returns:


* <b>`npairs_loss`</b>: float scalar.

