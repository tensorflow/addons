<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.lifted_struct_loss" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.losses.lifted_struct_loss


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/losses/lifted.py#L26-L103">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes the lifted structured loss.

### Aliases:

* `tfa.losses.lifted.lifted_struct_loss`


``` python
tfa.losses.lifted_struct_loss(
    labels,
    embeddings,
    margin=1.0
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`labels`</b>: 1-D tf.int32 `Tensor` with shape [batch_size] of
  multiclass integer labels.
* <b>`embeddings`</b>: 2-D float `Tensor` of embedding vectors. Embeddings should
  not be l2 normalized.
* <b>`margin`</b>: Float, margin term in the loss definition.


#### Returns:


* <b>`lifted_loss`</b>: tf.float32 scalar.