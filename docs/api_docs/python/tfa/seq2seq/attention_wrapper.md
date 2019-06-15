<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.seq2seq.attention_wrapper" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfa.seq2seq.attention_wrapper

A powerful dynamic attention wrapper object.



Defined in [`seq2seq/attention_wrapper.py`](https://github.com/tensorflow/addons/tree/0.4-release/tensorflow_addons/seq2seq/attention_wrapper.py).

<!-- Placeholder for "Used in" -->


## Classes

[`class AttentionMechanism`](../../tfa/seq2seq/AttentionMechanism.md)

[`class AttentionWrapper`](../../tfa/seq2seq/AttentionWrapper.md): Wraps another `RNNCell` with attention.

[`class AttentionWrapperState`](../../tfa/seq2seq/AttentionWrapperState.md): `namedtuple` storing the state of a `AttentionWrapper`.

[`class BahdanauAttention`](../../tfa/seq2seq/BahdanauAttention.md): Implements Bahdanau-style (additive) attention.

[`class BahdanauMonotonicAttention`](../../tfa/seq2seq/BahdanauMonotonicAttention.md): Monotonic attention mechanism with Bahadanau-style energy function.

[`class LuongAttention`](../../tfa/seq2seq/LuongAttention.md): Implements Luong-style (multiplicative) attention scoring.

[`class LuongMonotonicAttention`](../../tfa/seq2seq/LuongMonotonicAttention.md): Monotonic attention mechanism with Luong-style energy function.

## Functions

[`hardmax(...)`](../../tfa/seq2seq/hardmax.md): Returns batched one-hot vectors.

[`monotonic_attention(...)`](../../tfa/seq2seq/monotonic_attention.md): Compute monotonic attention distribution from choosing probabilities.

[`safe_cumprod(...)`](../../tfa/seq2seq/safe_cumprod.md): Computes cumprod of x in logspace using cumsum to avoid underflow.

