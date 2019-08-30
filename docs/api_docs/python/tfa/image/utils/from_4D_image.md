<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.image.utils.from_4D_image" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.image.utils.from_4D_image


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/image/utils.py#L72-L96">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Convert back to an image with `ndims` rank.

``` python
tfa.image.utils.from_4D_image(
    image,
    ndims
)
```



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`image`</b>: 4D tensor.
* <b>`ndims`</b>: The original rank of the image.


#### Returns:

`ndims`-D tensor with the same type.
