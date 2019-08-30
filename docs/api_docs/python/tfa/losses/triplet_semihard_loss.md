<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.triplet_semihard_loss" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.losses.triplet_semihard_loss


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/losses/triplet.py#L63-L131">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes the triplet loss with semi-hard negative mining.

### Aliases:

* `tfa.losses.triplet.triplet_semihard_loss`


``` python
tfa.losses.triplet_semihard_loss(
    y_true,
    y_pred,
    margin=1.0
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`y_true`</b>: 1-D integer `Tensor` with shape [batch_size] of
  multiclass integer labels.
* <b>`y_pred`</b>: 2-D float `Tensor` of embedding vectors. Embeddings should
  be l2 normalized.
* <b>`margin`</b>: Float, margin term in the loss definition.