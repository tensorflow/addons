<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.optimizers.lazy_adam" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfa.optimizers.lazy_adam


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/optimizers/lazy_adam.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



Variant of the Adam optimizer that handles sparse updates more efficiently.


Compared with the original Adam optimizer, the one in this file can
provide a large improvement in model training throughput for some
applications. However, it provides slightly different semantics than the
original Adam algorithm, and may lead to different empirical results.

## Classes

[`class LazyAdam`](../../tfa/optimizers/LazyAdam.md): Variant of the Adam optimizer that handles sparse updates more



