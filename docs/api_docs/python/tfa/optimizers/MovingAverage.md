<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.optimizers.MovingAverage" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="iterations"/>
<meta itemprop="property" content="learning_rate"/>
<meta itemprop="property" content="lr"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_slot"/>
<meta itemprop="property" content="add_weight"/>
<meta itemprop="property" content="apply_gradients"/>
<meta itemprop="property" content="assign_average_vars"/>
<meta itemprop="property" content="average_op"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_gradients"/>
<meta itemprop="property" content="get_slot"/>
<meta itemprop="property" content="get_slot_names"/>
<meta itemprop="property" content="get_updates"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="minimize"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="variables"/>
</div>

# tfa.optimizers.MovingAverage

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/moving_average.py#L27-L100">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
## Class `MovingAverage`

Optimizer that computes a moving average of the variables.

Inherits From: [`AveragedOptimizerWrapper`](../../tfa/optimizers/AveragedOptimizerWrapper.md)

**Aliases**: `tfa.optimizers.moving_average.MovingAverage`

<!-- Placeholder for "Used in" -->

Empirically it has been found that using the moving average of the trained
parameters of a deep network is better than using its trained parameters
directly. This optimizer allows you to compute this moving average and swap
the variables at save time so that any code outside of the training loop
will use by default the average values instead of the original ones.

#### Example of usage:



```python
opt = tf.keras.optimizers.SGD(learning_rate)
opt = tfa.optimizers.MovingAverage(opt)

```

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/moving_average.py#L45-L83">View source</a>

``` python
__init__(
    optimizer,
    sequential_update=True,
    average_decay=0.99,
    num_updates=None,
    name='MovingAverage',
    **kwargs
)
```

Construct a new MovingAverage optimizer.


#### Args:


* <b>`optimizer`</b>: str or `tf.keras.optimizers.Optimizer` that will be
    used to compute and apply gradients.
* <b>`sequential_update`</b>: Bool. If False, will compute the moving average
    at the same time as the model is updated, potentially doing
    benign data races. If True, will update the moving average
    after gradient updates.
* <b>`average_decay`</b>: float. Decay to use to maintain the moving averages
    of trained variables. 
* <b>`num_updates`</b>: Optional count of the number of updates applied to
    variables. 
* <b>`name`</b>: Optional name for the operations created when applying
    gradients. Defaults to "MovingAverage".
* <b>`**kwargs`</b>: keyword arguments. Allowed to be {`clipnorm`,
    `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
    norm; `clipvalue` is clip gradients by value, `decay` is
    included for backward compatibility to allow time inverse
    decay of learning rate. `lr` is included for backward
    compatibility, recommended to use `learning_rate` instead.



## Properties

<h3 id="iterations"><code>iterations</code></h3>

Variable. The number of training steps this Optimizer has run.


<h3 id="learning_rate"><code>learning_rate</code></h3>




<h3 id="lr"><code>lr</code></h3>




<h3 id="weights"><code>weights</code></h3>

Returns variables of this Optimizer based on the order created.




## Methods

<h3 id="add_slot"><code>add_slot</code></h3>

``` python
add_slot(
    var,
    slot_name,
    initializer='zeros'
)
```

Add a new slot variable for `var`.


<h3 id="add_weight"><code>add_weight</code></h3>

``` python
add_weight(
    name,
    shape,
    dtype=None,
    initializer='zeros',
    trainable=None,
    synchronization=tf_variables.VariableSynchronization.AUTO,
    aggregation=tf_variables.VariableAggregation.NONE
)
```




<h3 id="apply_gradients"><code>apply_gradients</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/average_wrapper.py#L59-L62">View source</a>

``` python
apply_gradients(
    grads_and_vars,
    name=None
)
```

Apply gradients to variables.

This is the second part of `minimize()`. It returns an `Operation` that
applies gradients.

#### Args:


* <b>`grads_and_vars`</b>: List of (gradient, variable) pairs.
* <b>`name`</b>: Optional name for the returned operation.  Default to the name
  passed to the `Optimizer` constructor.


#### Returns:

An `Operation` that applies the specified gradients. The `iterations`
will be automatically increased by 1.



#### Raises:


* <b>`TypeError`</b>: If `grads_and_vars` is malformed.
* <b>`ValueError`</b>: If none of the variables have gradients.

<h3 id="assign_average_vars"><code>assign_average_vars</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/average_wrapper.py#L95-L123">View source</a>

``` python
assign_average_vars(var_list)
```

Assign variables in var_list with their respective averages.


#### Args:


* <b>`var_list`</b>: List of model variables to be assigned to their average.


#### Returns:


* <b>`assign_op`</b>: The op corresponding to the assignment operation of
variables to their average.


#### Example:


```python
model = tf.Sequential([...])
opt = tfa.optimizers.SWA(
        tf.keras.optimizers.SGD(lr=2.0), 100, 10)
model.compile(opt, ...)
model.fit(x, y, ...)

# Update the weights to their mean before saving
opt.assign_average_vars(model.variables)

model.save('model.h5')
```

<h3 id="average_op"><code>average_op</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/moving_average.py#L85-L87">View source</a>

``` python
average_op(
    var,
    average_var
)
```




<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/average_wrapper.py#L133-L139">View source</a>

``` python
from_config(
    cls,
    config,
    custom_objects=None
)
```

Creates an optimizer from its config.

This method is the reverse of `get_config`,
capable of instantiating the same optimizer from the config
dictionary.

#### Arguments:


* <b>`config`</b>: A Python dictionary, typically the output of get_config.
* <b>`custom_objects`</b>: A Python dictionary mapping names to additional Python
  objects used to create this optimizer, such as a function used for a
  hyperparameter.


#### Returns:

An optimizer instance.


<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/moving_average.py#L89-L95">View source</a>

``` python
get_config()
```

Returns the config of the optimimizer.

An optimizer config is a Python dictionary (serializable)
containing the configuration of an optimizer.
The same optimizer can be reinstantiated later
(without any saved state) from this configuration.

#### Returns:

Python dictionary.


<h3 id="get_gradients"><code>get_gradients</code></h3>

``` python
get_gradients(
    loss,
    params
)
```

Returns gradients of `loss` with respect to `params`.


#### Arguments:


* <b>`loss`</b>: Loss tensor.
* <b>`params`</b>: List of variables.


#### Returns:

List of gradient tensors.



#### Raises:


* <b>`ValueError`</b>: In case any gradient cannot be computed (e.g. if gradient
  function not implemented).

<h3 id="get_slot"><code>get_slot</code></h3>

``` python
get_slot(
    var,
    slot_name
)
```




<h3 id="get_slot_names"><code>get_slot_names</code></h3>

``` python
get_slot_names()
```

A list of names for this optimizer's slots.


<h3 id="get_updates"><code>get_updates</code></h3>

``` python
get_updates(
    loss,
    params
)
```




<h3 id="get_weights"><code>get_weights</code></h3>

``` python
get_weights()
```




<h3 id="minimize"><code>minimize</code></h3>

``` python
minimize(
    loss,
    var_list,
    grad_loss=None,
    name=None
)
```

Minimize `loss` by updating `var_list`.

This method simply computes gradient using `tf.GradientTape` and calls
`apply_gradients()`. If you want to process the gradient before applying
then call `tf.GradientTape` and `apply_gradients()` explicitly instead
of using this function.

#### Args:


* <b>`loss`</b>: A callable taking no arguments which returns the value to minimize.
* <b>`var_list`</b>: list or tuple of `Variable` objects to update to minimize
  `loss`, or a callable returning the list or tuple of `Variable` objects.
  Use callable when the variable list would otherwise be incomplete before
  `minimize` since the variables are created at the first time `loss` is
  called.
* <b>`grad_loss`</b>: Optional. A `Tensor` holding the gradient computed for `loss`.
* <b>`name`</b>: Optional name for the returned operation.


#### Returns:

An `Operation` that updates the variables in `var_list`. The `iterations`
will be automatically increased by 1.



#### Raises:


* <b>`ValueError`</b>: If some of the variables are not `Variable` objects.

<h3 id="set_weights"><code>set_weights</code></h3>

``` python
set_weights(weights)
```




<h3 id="variables"><code>variables</code></h3>

``` python
variables()
```

Returns variables of this Optimizer based on the order created.






