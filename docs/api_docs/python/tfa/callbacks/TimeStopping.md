<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.callbacks.TimeStopping" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="set_model"/>
<meta itemprop="property" content="set_params"/>
</div>

# tfa.callbacks.TimeStopping

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/callbacks/time_stopping.py#L27-L64">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
## Class `TimeStopping`

Stop training when a specified amount of time has passed.



**Aliases**: `tfa.callbacks.time_stopping.TimeStopping`

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`seconds`</b>: maximum amount of time before stopping.
    Defaults to 86400 (1 day).
* <b>`verbose`</b>: verbosity mode. Defaults to 0.

<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/callbacks/time_stopping.py#L36-L40">View source</a>

``` python
__init__(
    seconds=86400,
    verbose=0
)
```

Initialize self.  See help(type(self)) for accurate signature.




## Methods

<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/callbacks/time_stopping.py#L57-L64">View source</a>

``` python
get_config()
```




<h3 id="set_model"><code>set_model</code></h3>

``` python
set_model(model)
```




<h3 id="set_params"><code>set_params</code></h3>

``` python
set_params(params)
```








