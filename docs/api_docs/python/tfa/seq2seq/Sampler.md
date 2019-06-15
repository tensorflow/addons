<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.Sampler" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="sample_ids_dtype"/>
<meta itemprop="property" content="sample_ids_shape"/>
<meta itemprop="property" content="initialize"/>
<meta itemprop="property" content="next_inputs"/>
<meta itemprop="property" content="sample"/>
</div>

# tfa.seq2seq.Sampler

## Class `Sampler`

Interface for implementing sampling in seq2seq decoders.



### Aliases:

* Class `tfa.seq2seq.Sampler`
* Class `tfa.seq2seq.sampler.Sampler`



Defined in [`seq2seq/sampler.py`](https://github.com/tensorflow/addons/tree/0.4-release/tensorflow_addons/seq2seq/sampler.py).

<!-- Placeholder for "Used in" -->

Sampler instances are used by `BasicDecoder`. The normal usage of a sampler
is like below:
sampler = Sampler(init_args)
(initial_finished, initial_inputs) = sampler.initialize(input_tensors)
for time_step in range(time):
  cell_output, cell_state = cell.call(cell_input, previous_state)
  sample_ids = sampler.sample(time_step, cell_output, cell_state)
  (finished, next_inputs, next_state) = sampler.next_inputs(
      time_step,cell_output, cell_state)

Note that all the tensor input should not be feed to Sampler as __init__()
parameters, instead, they should be feed by decoders via initialize().

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

``` python
sample(
    time,
    outputs,
    state
)
```

Returns `sample_ids`.




