<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.InferenceSampler" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="sample_ids_dtype"/>
<meta itemprop="property" content="sample_ids_shape"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initialize"/>
<meta itemprop="property" content="next_inputs"/>
<meta itemprop="property" content="sample"/>
</div>

# tfa.seq2seq.InferenceSampler


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/seq2seq/sampler.py#L626-L691">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



## Class `InferenceSampler`

A helper to use during inference with a custom sampling function.

Inherits From: [`Sampler`](../../tfa/seq2seq/Sampler.md)

### Aliases:

* Class `tfa.seq2seq.sampler.InferenceSampler`


<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/seq2seq/sampler.py#L629-L656">View source</a>

``` python
__init__(
    sample_fn,
    sample_shape,
    sample_dtype,
    end_fn,
    next_inputs_fn=None
)
```

Initializer.


#### Args:


* <b>`sample_fn`</b>: A callable that takes `outputs` and emits tensor
  `sample_ids`.
* <b>`sample_shape`</b>: Either a list of integers, or a 1-D Tensor of type
  `int32`, the shape of the each sample in the batch returned by
  `sample_fn`.
* <b>`sample_dtype`</b>: the dtype of the sample returned by `sample_fn`.
* <b>`end_fn`</b>: A callable that takes `sample_ids` and emits a `bool` vector
  shaped `[batch_size]` indicating whether each sample is an end
  token.
* <b>`next_inputs_fn`</b>: (Optional) A callable that takes `sample_ids` and
  returns the next batch of inputs. If not provided, `sample_ids` is
  used as the next batch of inputs.



## Properties

<h3 id="batch_size"><code>batch_size</code></h3>




<h3 id="sample_ids_dtype"><code>sample_ids_dtype</code></h3>




<h3 id="sample_ids_shape"><code>sample_ids_shape</code></h3>






## Methods

<h3 id="initialize"><code>initialize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/seq2seq/sampler.py#L673-L678">View source</a>

``` python
initialize(start_inputs)
```




<h3 id="next_inputs"><code>next_inputs</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/seq2seq/sampler.py#L684-L691">View source</a>

``` python
next_inputs(
    time,
    outputs,
    state,
    sample_ids
)
```




<h3 id="sample"><code>sample</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/seq2seq/sampler.py#L680-L682">View source</a>

``` python
sample(
    time,
    outputs,
    state
)
```






