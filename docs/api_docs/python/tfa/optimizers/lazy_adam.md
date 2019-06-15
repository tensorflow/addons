<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.optimizers.lazy_adam" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfa.optimizers.lazy_adam

Variant of the Adam optimizer that handles sparse updates more efficiently.



Defined in [`optimizers/lazy_adam.py`](https://github.com/tensorflow/addons/tree/0.4-release/tensorflow_addons/optimizers/lazy_adam.py).

<!-- Placeholder for "Used in" -->

Compared with the original Adam optimizer, the one in this file can
provide a large improvement in model training throughput for some
applications. However, it provides slightly different semantics than the
original Adam algorithm, and may lead to different empirical results.

## Classes

[`class LazyAdam`](../../tfa/optimizers/LazyAdam.md): Variant of the Adam optimizer that handles sparse updates more

