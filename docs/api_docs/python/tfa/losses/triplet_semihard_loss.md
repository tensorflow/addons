<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.triplet_semihard_loss" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.losses.triplet_semihard_loss

Computes the triplet loss with semi-hard negative mining.

### Aliases:

* `tfa.losses.triplet.triplet_semihard_loss`
* `tfa.losses.triplet_semihard_loss`

``` python
tfa.losses.triplet_semihard_loss(
    y_true,
    y_pred,
    margin=1.0
)
```



Defined in [`losses/triplet.py`](https://github.com/tensorflow/addons/tree/0.4-release/tensorflow_addons/losses/triplet.py).

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`y_true`</b>: 1-D integer `Tensor` with shape [batch_size] of
  multiclass integer labels.
* <b>`y_pred`</b>: 2-D float `Tensor` of embedding vectors. Embeddings should
  be l2 normalized.
* <b>`margin`</b>: Float, margin term in the loss definition.