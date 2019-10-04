<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.npairs_multilabel_loss" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.losses.npairs_multilabel_loss


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/losses/npairs.py#L67-L129">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Computes the npairs loss between multilabel data `y_true` and `y_pred`.

### Aliases:

* `tfa.losses.npairs.npairs_multilabel_loss`


``` python
tfa.losses.npairs_multilabel_loss(
    y_true,
    y_pred
)
```



<!-- Placeholder for "Used in" -->

Npairs loss expects paired data where a pair is composed of samples from
the same labels and each pairs in the minibatch have different labels.
The loss takes each row of the pair-wise similarity matrix, `y_pred`,
as logits and the remapped multi-class labels, `y_true`, as labels.

To deal with multilabel inputs, the count of label intersection
is computed as follows:

```
L_{i,j} = | set_of_labels_for(i) \cap set_of_labels_for(j) |
```

Each row of the count based label matrix is further normalized so that
each row sums to one.

`y_true` should be a binary indicator for classes.
That is, if `y_true[i, j] = 1`, then `i`th sample is in `j`th class;
if `y_true[i, j] = 0`, then `i`th sample is not in `j`th class.

The similarity matrix `y_pred` between two embedding matrices `a` and `b`
with shape `[batch_size, hidden_size]` can be computed as follows:

```python
# y_pred = a * b^T
y_pred = tf.matmul(a, b, transpose_a=False, transpose_b=True)
```

See: http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf

#### Args:


* <b>`y_true`</b>: Either 2-D integer `Tensor` with shape
  `[batch_size, num_classes]`, or `SparseTensor` with dense shape
  `[batch_size, num_classes]`. If `y_true` is a `SparseTensor`, then
  it will be converted to `Tensor` via `tf.sparse.to_dense` first.

* <b>`y_pred`</b>: 2-D float `Tensor` with shape `[batch_size, batch_size]` of
  similarity matrix between embedding matrices.


#### Returns:


* <b>`npairs_multilabel_loss`</b>: float scalar.