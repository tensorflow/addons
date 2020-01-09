<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.dense_image_warp" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.dense_image_warp

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/image/dense_image_warp.py#L184-L237">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Image warping using per-pixel flow vectors.

``` python
tfa.image.dense_image_warp(
    image,
    flow,
    name=None
)
```



<!-- Placeholder for "Used in" -->

Apply a non-linear warp to the image, where the warp is specified by a
dense flow field of offset vectors that define the correspondences of
pixel values in the output image back to locations in the source image.
Specifically, the pixel value at output[b, j, i, c] is
images[b, j - flow[b, j, i, 0], i - flow[b, j, i, 1], c].

The locations specified by this formula do not necessarily map to an int
index. Therefore, the pixel value is obtained by bilinear
interpolation of the 4 nearest pixels around
(b, j - flow[b, j, i, 0], i - flow[b, j, i, 1]). For locations outside
of the image, we use the nearest pixel values at the image boundary.

#### Args:


* <b>`image`</b>: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
* <b>`flow`</b>: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
* <b>`name`</b>: A name for the operation (optional).

Note that image and flow can be of type tf.half, tf.float32, or
tf.float64, and do not necessarily have to be the same type.


#### Returns:

A 4-D float `Tensor` with shape`[batch, height, width, channels]`
  and same type as input image.



#### Raises:


* <b>`ValueError`</b>: if height < 2 or width < 2 or the inputs have the wrong
  number of dimensions.

