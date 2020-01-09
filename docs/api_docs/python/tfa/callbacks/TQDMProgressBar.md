<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.callbacks.TQDMProgressBar" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="format_metrics"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="set_model"/>
<meta itemprop="property" content="set_params"/>
</div>

# tfa.callbacks.TQDMProgressBar

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/callbacks/tqdm_progress_bar.py#L27-L218">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
## Class `TQDMProgressBar`

TQDM Progress Bar for Tensorflow Keras.



**Aliases**: `tfa.callbacks.tqdm_progress_bar.TQDMProgressBar`

<!-- Placeholder for "Used in" -->


#### Arguments:

metrics_separator (string): Custom separator between metrics.
    Defaults to ' - '
overall_bar_format (string format): Custom bar format for overall
    (outer) progress bar, see https://github.com/tqdm/tqdm#parameters
    for more detail.
epoch_bar_format (string format): Custom bar format for epoch
    (inner) progress bar, see https://github.com/tqdm/tqdm#parameters
    for more detail.
update_per_second (int): Maximum number of updates in the epochs bar
    per second, this is to prevent small batches from slowing down
    training. Defaults to 10.
leave_epoch_progress (bool): True to leave epoch progress bars
leave_overall_progress (bool): True to leave overall progress bar
show_epoch_progress (bool): False to hide epoch progress bars
show_overall_progress (bool): False to hide overall progress bar


<h2 id="__init__"><code>__init__</code></h2>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/callbacks/tqdm_progress_bar.py#L48-L89">View source</a>

``` python
__init__(
    metrics_separator=' - ',
    overall_bar_format='{l_bar}{bar} {n_fmt}/{total_fmt} ETA: {remaining}s,  {rate_fmt}{postfix}',
    epoch_bar_format='{n_fmt}/{total_fmt}{bar} ETA: {remaining}s - {desc}',
    update_per_second=10,
    leave_epoch_progress=True,
    leave_overall_progress=True,
    show_epoch_progress=True,
    show_overall_progress=True
)
```

Initialize self.  See help(type(self)) for accurate signature.




## Methods

<h3 id="format_metrics"><code>format_metrics</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/callbacks/tqdm_progress_bar.py#L182-L204">View source</a>

``` python
format_metrics(
    logs={},
    factor=1
)
```

Format metrics in logs into a string.


#### Arguments:


* <b>`logs`</b>: dictionary of metrics and their values. Defaults to
    empty dictionary.
factor (int): The factor we want to divide the metrics in logs
    by, useful when we are computing the logs after each batch.
    Defaults to 1.


#### Returns:


* <b>`metrics_string`</b>: a string displaying metrics using the given
formators passed in through the constructor.

<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/callbacks/tqdm_progress_bar.py#L206-L218">View source</a>

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








