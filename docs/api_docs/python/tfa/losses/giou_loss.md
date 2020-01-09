<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.giou_loss" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.losses.giou_loss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/losses/giou_loss.py#L72-L94">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Args:

``` python
tfa.losses.giou_loss(
    y_true,
    y_pred,
    mode='giou'
)
```



<!-- Placeholder for "Used in" -->
    y_true: true targets tensor. The coordinates of the each bounding
        box in boxes are encoded as [y_min, x_min, y_max, x_max].
    y_pred: predictions tensor. The coordinates of the each bounding
        box in boxes are encoded as [y_min, x_min, y_max, x_max].
    mode: one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.

#### Returns:

GIoU loss float `Tensor`.


