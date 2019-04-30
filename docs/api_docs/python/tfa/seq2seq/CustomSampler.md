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

## Class `CustomSampler`

Base abstract class that allows the user to customize sampling.

Inherits From: [`Sampler`](../../tfa/seq2seq/Sampler.md)

### Aliases:

* Class `tfa.seq2seq.CustomSampler`
* Class `tfa.seq2seq.sampler.CustomSampler`



Defined in [`seq2seq/sampler.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/seq2seq/sampler.py).

<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

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



<h3 id="sample_ids_dtype"><code>sample_ids_dtype</code></h3>



<h3 id="sample_ids_shape"><code>sample_ids_shape</code></h3>





## Methods

<h3 id="initialize"><code>initialize</code></h3>

``` python
initialize(
    inputs,
    **kwargs
)
```



<h3 id="next_inputs"><code>next_inputs</code></h3>

``` python
next_inputs(
    time,
    outputs,
    state,
    sample_ids
)
```



<h3 id="sample"><code>sample</code></h3>

``` python
sample(
    time,
    outputs,
    state
)
```





