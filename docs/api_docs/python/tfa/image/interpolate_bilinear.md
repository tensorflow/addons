<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.interpolate_bilinear" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.interpolate_bilinear


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.6/tensorflow_addons/image/dense_image_warp.py#L24-L174">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Similar to Matlab's interp2 function.

``` python
tfa.image.interpolate_bilinear(
    grid,
    query_points,
    indexing='ij',
    name=None
)
```



<!-- Placeholder for "Used in" -->

Finds values for query points on a grid using bilinear interpolation.

#### Args:


* <b>`grid`</b>: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
* <b>`query_points`</b>: a 3-D float `Tensor` of N points with shape
  `[batch, N, 2]`.
* <b>`indexing`</b>: whether the query points are specified as row and column (ij),
  or Cartesian coordinates (xy).
* <b>`name`</b>: a name for the operation (optional).


#### Returns:


* <b>`values`</b>: a 3-D `Tensor` with shape `[batch, N, channels]`


#### Raises:


* <b>`ValueError`</b>: if the indexing mode is invalid, or if the shape of the
  inputs invalid.