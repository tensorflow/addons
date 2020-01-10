<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.sigmoid_focal_crossentropy" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.losses.sigmoid_focal_crossentropy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/losses/focal_loss.py#L100-L151">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Args

**Aliases**: `tfa.losses.focal_loss.sigmoid_focal_crossentropy`

``` python
tfa.losses.sigmoid_focal_crossentropy(
    y_true,
    y_pred,
    alpha=0.25,
    gamma=2.0,
    from_logits=False
)
```



<!-- Placeholder for "Used in" -->
    y_true: true targets tensor.
    y_pred: predictions tensor.
    alpha: balancing factor.
    gamma: modulating factor.

#### Returns:

Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
same shape as `y_true`; otherwise, it is scalar.


