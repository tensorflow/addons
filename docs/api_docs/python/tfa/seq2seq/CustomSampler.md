<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.CustomSampler" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="sample_ids_dtype"/>
<meta itemprop="property" content="sample_ids_shape"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initialize"/>
<meta itemprop="property" content="next_inputs"/>
<meta itemprop="property" content="sample"/>
</div>

# tfa.seq2seq.CustomSampler

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L107-L165">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
## Class `CustomSampler`

Base abstract class that allows the user to customize sampling.

Inherits From: [`Sampler`](../../tfa/seq2seq/Sampler.md)

**Aliases**: `tfa.seq2seq.sampler.CustomSampler`

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L110-L137">View source</a>

``` python
__init__(
    initialize_fn,
    sample_fn,
    next_inputs_fn,
    sample_ids_shape=None,
    sample_ids_dtype=None
)
```

Initializer.


#### Args:


* <b>`initialize_fn`</b>: callable that returns `(finished, next_inputs)` for
  the first iteration.
* <b>`sample_fn`</b>: callable that takes `(time, outputs, state)` and emits
  tensor `sample_ids`.
* <b>`next_inputs_fn`</b>: callable that takes
  `(time, outputs, state, sample_ids)` and emits
  `(finished, next_inputs, next_state)`.
* <b>`sample_ids_shape`</b>: Either a list of integers, or a 1-D Tensor of type
  `int32`, the shape of each value in the `sample_ids` batch.
  Defaults to a scalar.
* <b>`sample_ids_dtype`</b>: The dtype of the `sample_ids` tensor. Defaults to
  int32.



## Properties

<h3 id="batch_size"><code>batch_size</code></h3>

Batch size of tensor returned by `sample`.

Returns a scalar int32 tensor. The return value might not
available before the invocation of initialize(), in this case,
ValueError is raised.

<h3 id="sample_ids_dtype"><code>sample_ids_dtype</code></h3>

DType of tensor returned by `sample`.

Returns a DType. The return value might not available before the
invocation of initialize().

<h3 id="sample_ids_shape"><code>sample_ids_shape</code></h3>

Shape of tensor returned by `sample`, excluding the batch dimension.

Returns a `TensorShape`. The return value might not available
before the invocation of initialize().



## Methods

<h3 id="initialize"><code>initialize</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L154-L158">View source</a>

``` python
initialize(
    inputs,
    **kwargs
)
```

initialize the sampler with the input tensors.

This method suppose to be only invoke once before the calling other
methods of the Sampler.

#### Args:


* <b>`inputs`</b>: A (structure of) input tensors, it could be a nested tuple or
  a single tensor.
* <b>`**kwargs`</b>: Other kwargs for initialization. It could contain tensors
  like mask for inputs, or non tensor parameter.


#### Returns:

`(initial_finished, initial_inputs)`.


<h3 id="next_inputs"><code>next_inputs</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L163-L165">View source</a>

``` python
next_inputs(
    time,
    outputs,
    state,
    sample_ids
)
```

Returns `(finished, next_inputs, next_state)`.


<h3 id="sample"><code>sample</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/seq2seq/sampler.py#L160-L161">View source</a>

``` python
sample(
    time,
    outputs,
    state
)
```

Returns `sample_ids`.






