<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.optimizers.AveragedOptimizerWrapper" />
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

# tfa.optimizers.AveragedOptimizerWrapper

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/average_wrapper.py#L27-L159">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
## Class `AveragedOptimizerWrapper`

Updated base class for optimizers.



**Aliases**: `tfa.optimizers.average_wrapper.AveragedOptimizerWrapper`

<!-- Placeholder for "Used in" -->

This class defines the API to add Ops to train a model.  You never use this
class directly, but instead instantiate one of its subclasses such as
`tf.keras.optimizers.SGD`, `tf.keras.optimizers.Adam`.

### Usage

```python
# Create an optimizer with the desired parameters.
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
# `loss` is a callable that takes no argument and returns the value
# to minimize.
loss = lambda: 3 * var1 * var1 + 2 * var2 * var2
# In graph mode, returns op that minimizes the loss by updating the listed
# variables.
opt_op = opt.minimize(loss, var_list=[var1, var2])
opt_op.run()
# In eager mode, simply call minimize to update the list of variables.
opt.minimize(loss, var_list=[var1, var2])
```

### Custom training loop with Keras models

In Keras models, sometimes variables are created when the model is first
called, instead of construction time. Examples include 1) sequential models
without input shape pre-defined, or 2) subclassed models. Pass var_list as
callable in these cases.

#### Example:


```python
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(num_hidden, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))
loss_fn = lambda: tf.keras.losses.mse(model(input), output)
var_list_fn = lambda: model.trainable_weights
for input, output in data:
  opt.minimize(loss_fn, var_list_fn)
```

### Processing gradients before applying them.

Calling `minimize()` takes care of both computing the gradients and
applying them to the variables.  If you want to process the gradients
before applying them you can instead use the optimizer in three steps:

1.  Compute the gradients with `tf.GradientTape`.
2.  Process the gradients as you wish.
3.  Apply the processed gradients with `apply_gradients()`.

#### Example:



```python
# Create an optimizer.
opt = tf.keras.optimizers.SGD(learning_rate=0.1)

# Compute the gradients for a list of variables.
with tf.GradientTape() as tape:
  loss = <call_loss_function>
vars = <list_of_variables>
grads = tape.gradient(loss, vars)

# Process the gradients, for example cap them, etc.
# capped_grads = [MyCapper(g) for g in grads]
processed_grads = [process_gradient(g) for g in grads]

# Ask the optimizer to apply the processed gradients.
opt.apply_gradients(zip(processed_grads, var_list))
```

### Use with `tf.distribute.Strategy`.

This optimizer class is `tf.distribute.Strategy` aware, which means it
automatically sums gradients across all replicas. To average gradients,
you divide your loss by the global batch size, which is done
automatically if you use `tf.keras` built-in training or evaluation loops.
See the `reduction` argument of your loss which should be set to
`tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` for averaging or
`tf.keras.losses.Reduction.SUM` for not.

If you are not using these and you want to average gradients, you should use
`tf.math.reduce_sum` to add up your per-example losses and then divide by the
global batch size. Note that when using `tf.distribute.Strategy`, the first
component of a tensor's shape is the *replica-local* batch size, which is off
by a factor equal to the number of replicas being used to compute a single
step. As a result, using `tf.math.reduce_mean` will give the wrong answer,
resulting in gradients that can be many times too big.

### Variable Constraint

All Keras optimizers respect variable constraints. If constraint function is
passed to any variable, the constraint will be applied to the variable after
the gradient has been applied to the variable.
Important: If gradient is sparse tensor, variable constraint is not supported.

### Thread Compatibility

The entire optimizer is currently thread compatible, not thread-safe. The user
needs to perform synchronization if necessary.

### Slots

Many optimizer subclasses, such as `Adam` and `Adagrad` allocate and manage
additional variables associated with the variables to train.  These are called
<i>Slots</i>.  Slots have names and you can ask the optimizer for the names of
the slots that it uses.  Once you have a slot name you can ask the optimizer
for the variable it created to hold the slot value.

This can be useful if you want to log debug a training algorithm, report stats
about the slots, etc.

### Hyper parameters

These are arguments passed to the optimizer subclass constructor
(the `__init__` method), and then passed to `self._set_hyper()`.
They can be either regular Python values (like 1.0), tensors, or
callables. If they are callable, the callable will be called during
`apply_gradients()` to get the value for the hyper parameter.

Hyper parameters can be overwritten through user code:

#### Example:



```python
# Create an optimizer with the desired parameters.
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
# `loss` is a callable that takes no argument and returns the value
# to minimize.
loss = lambda: 3 * var1 + 2 * var2
# In eager mode, simply call minimize to update the list of variables.
opt.minimize(loss, var_list=[var1, var2])
# update learning rate
opt.learning_rate = 0.05
opt.minimize(loss, var_list=[var1, var2])
```

### Write a customized optimizer.
If you intend to create your own optimization algorithm, simply inherit from
this class and override the following methods:

  - resource_apply_dense (update variable given gradient tensor is dense)
  - resource_apply_sparse (update variable given gradient tensor is sparse)
  - create_slots (if your optimizer algorithm requires additional variables)
  - get_config (serialization of the optimizer, include all hyper parameters)

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/average_wrapper.py#L28-L46">View source</a>

``` python
__init__(
    optimizer,
    sequential_update=True,
    name='AverageOptimizer',
    **kwargs
)
```

Create a new Optimizer.

This must be called by the constructors of subclasses.
Note that Optimizer instances should not bind to a single graph,
and so shouldn't keep Tensors as member variables. Generally
you should be able to use the _set_hyper()/state.get_hyper()
facility instead.

This class in stateful and thread-compatible.

#### Args:


* <b>`name`</b>: A non-empty string.  The name to use for accumulators created
  for the optimizer.
* <b>`**kwargs`</b>: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
  `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
  gradients by value, `decay` is included for backward compatibility to
  allow time inverse decay of learning rate. `lr` is included for backward
  compatibility, recommended to use `learning_rate` instead.


#### Raises:


* <b>`ValueError`</b>: If name is malformed.
* <b>`RuntimeError`</b>: If _create_slots has been overridden instead of
    _create_vars.



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

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/average_wrapper.py#L64-L66">View source</a>

``` python
average_op(
    var,
    average_var
)
```




<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/average_wrapper.py#L133-L139">View source</a>

``` python
@classmethod
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

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/average_wrapper.py#L125-L131">View source</a>

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






