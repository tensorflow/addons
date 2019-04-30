<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.optimizers.SGDW" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="iterations"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__getattribute__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__setattr__"/>
<meta itemprop="property" content="add_slot"/>
<meta itemprop="property" content="add_weight"/>
<meta itemprop="property" content="apply_gradients"/>
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

# tfa.optimizers.SGDW

## Class `SGDW`

Optimizer that implements the Momentum algorithm with weight_decay.

Inherits From: [`DecoupledWeightDecayExtension`](../../tfa/optimizers/weight_decay_optimizers/DecoupledWeightDecayExtension.md)

### Aliases:

* Class `tfa.optimizers.SGDW`
* Class `tfa.optimizers.weight_decay_optimizers.SGDW`



Defined in [`optimizers/weight_decay_optimizers.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/optimizers/weight_decay_optimizers.py).

<!-- Placeholder for "Used in" -->

This is an implementation of the SGDW optimizer described in "Decoupled
Weight Decay Regularization" by Loshchilov & Hutter
(https://arxiv.org/abs/1711.05101)
([pdf])(https://arxiv.org/pdf/1711.05101.pdf).
It computes the update step of `tf.keras.optimizers.SGD` and additionally
decays the variable. Note that this is different from adding
L2 regularization on the variables to the loss. Decoupling the weight decay
from other hyperparameters (in particular the learning rate) simplifies
hyperparameter search.

For further information see the documentation of the SGD Optimizer.

This optimizer can also be instantiated as
```python
extend_with_decoupled_weight_decay(tf.keras.optimizers.SGD,
                                   weight_decay=weight_decay)
```

Note: when applying a decay to the learning rate, be sure to manually apply
the decay to the `weight_decay` as well. For example:

```python
step = tf.Variable(0, trainable=False)
schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
    [10000, 15000], [1e-0, 1e-1, 1e-2])
# lr and wd can be a function or a tensor
lr = 1e-1 * schedule(step)
wd = lambda: 1e-4 * schedule(step)

# ...

optimizer = tfa.optimizers.SGDW(
    learning_rate=lr, weight_decay=wd, momentum=0.9)
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    weight_decay,
    learning_rate=0.001,
    momentum=0.0,
    nesterov=False,
    name='SGDW',
    **kwargs
)
```

Construct a new SGDW optimizer.

For further information see the documentation of the SGD Optimizer.

#### Args:

* <b>`learning_rate`</b>: float hyperparameter >= 0. Learning rate.
* <b>`momentum`</b>: float hyperparameter >= 0 that accelerates SGD in the
        relevant direction and dampens oscillations.
* <b>`nesterov`</b>: boolean. Whether to apply Nesterov momentum.
* <b>`name`</b>: Optional name prefix for the operations created when applying
        gradients.  Defaults to 'SGD'.
* <b>`**kwargs`</b>: keyword arguments. Allowed to be {`clipnorm`,
        `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
        norm; `clipvalue` is clip gradients by value, `decay` is
        included for backward compatibility to allow time inverse decay
        of learning rate. `lr` is included for backward compatibility,
        recommended to use `learning_rate` instead.



## Properties

<h3 id="iterations"><code>iterations</code></h3>

Variable. The number of training steps this Optimizer has run.

<h3 id="weights"><code>weights</code></h3>

Returns variables of this Optimizer based on the order created.



## Methods

<h3 id="__getattribute__"><code>__getattribute__</code></h3>

``` python
__getattribute__(name)
```

Overridden to support hyperparameter access.

<h3 id="__setattr__"><code>__setattr__</code></h3>

``` python
__setattr__(
    name,
    value
)
```

Override setattr to support dynamic hyperparameter setting.

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

``` python
apply_gradients(
    grads_and_vars,
    name=None,
    decay_var_list=None
)
```

Apply gradients to variables.

This is the second part of `minimize()`. It returns an `Operation` that
applies gradients.

#### Args:

* <b>`grads_and_vars`</b>: List of (gradient, variable) pairs.
* <b>`name`</b>: Optional name for the returned operation.  Default to the
        name passed to the `Optimizer` constructor.
* <b>`decay_var_list`</b>: Optional list of variables to be decayed. Defaults
        to all variables in var_list.

#### Returns:

An `Operation` that applies the specified gradients. If
`global_step` was not None, that operation also increments
`global_step`.

#### Raises:

* <b>`TypeError`</b>: If `grads_and_vars` is malformed.
* <b>`ValueError`</b>: If none of the variables have gradients.

<h3 id="from_config"><code>from_config</code></h3>

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

``` python
get_config()
```



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
    name=None,
    decay_var_list=None
)
```

Minimize `loss` by updating `var_list`.

This method simply computes gradient using `tf.GradientTape` and calls
`apply_gradients()`. If you want to process the gradient before
applying then call `tf.GradientTape` and `apply_gradients()` explicitly
instead of using this function.

#### Args:

* <b>`loss`</b>: A callable taking no arguments which returns the value to
        minimize.
* <b>`var_list`</b>: list or tuple of `Variable` objects to update to
        minimize `loss`, or a callable returning the list or tuple of
        `Variable` objects. Use callable when the variable list would
        otherwise be incomplete before `minimize` since the variables
        are created at the first time `loss` is called.
* <b>`grad_loss`</b>: Optional. A `Tensor` holding the gradient computed for
        `loss`.
* <b>`decay_var_list`</b>: Optional list of variables to be decayed. Defaults
        to all variables in var_list.
* <b>`name`</b>: Optional name for the returned operation.

#### Returns:

An Operation that updates the variables in `var_list`.  If
`global_step` was not `None`, that operation also increments
`global_step`.

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



