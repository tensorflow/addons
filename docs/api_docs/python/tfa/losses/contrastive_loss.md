<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.contrastive_loss" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.losses.contrastive_loss


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/losses/contrastive.py#L24-L58">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes the contrastive loss between `y_true` and `y_pred`.

### Aliases:

* `tfa.losses.contrastive.contrastive_loss`


``` python
tfa.losses.contrastive_loss(
    y_true,
    y_pred,
    margin=1.0
)
```



<!-- Placeholder for "Used in" -->

This loss encourages the embedding to be close to each other for
the samples of the same label and the embedding to be far apart at least
by the margin constant for the samples of different labels.

The euclidean distances `y_pred` between two embedding matrices
`a` and `b` with shape [batch_size, hidden_size] can be computed
as follows:

```python
# y_pred = \sqrt (\sum_i (a[:, i] - b[:, i])^2)
y_pred = tf.linalg.norm(a - b, axis=1)
```

See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

#### Args:


* <b>`y_true`</b>: 1-D integer `Tensor` with shape [batch_size] of
  binary labels indicating positive vs negative pair.
* <b>`y_pred`</b>: 1-D float `Tensor` with shape [batch_size] of
  distances between two embedding matrices.
* <b>`margin`</b>: margin term in the loss definition.


#### Returns:


* <b>`contrastive_loss`</b>: 1-D float `Tensor` with shape [batch_size].