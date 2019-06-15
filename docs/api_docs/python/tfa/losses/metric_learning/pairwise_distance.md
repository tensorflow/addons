<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.losses.metric_learning.pairwise_distance" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.losses.metric_learning.pairwise_distance

Computes the pairwise distance matrix with numerical stability.

``` python
tfa.losses.metric_learning.pairwise_distance(
    feature,
    squared=False
)
```



Defined in [`losses/metric_learning.py`](https://github.com/tensorflow/addons/tree/0.4-release/tensorflow_addons/losses/metric_learning.py).

<!-- Placeholder for "Used in" -->

output[i, j] = || feature[i, :] - feature[j, :] ||_2

#### Args:


* <b>`feature`</b>: 2-D Tensor of size [number of data, feature dimension].
* <b>`squared`</b>: Boolean, whether or not to square the pairwise distances.


#### Returns:


* <b>`pairwise_distances`</b>: 2-D Tensor of size [number of data, number of data].