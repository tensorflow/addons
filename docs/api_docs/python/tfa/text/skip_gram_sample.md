<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.text.skip_gram_sample" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.text.skip_gram_sample

Generates skip-gram token and label paired Tensors from the input

### Aliases:

* `tfa.text.skip_gram_ops.skip_gram_sample`
* `tfa.text.skip_gram_sample`

``` python
tfa.text.skip_gram_sample(
    input_tensor,
    min_skips=1,
    max_skips=5,
    start=0,
    limit=-1,
    emit_self_as_target=False,
    vocab_freq_table=None,
    vocab_min_count=None,
    vocab_subsampling=None,
    corpus_size=None,
    batch_size=None,
    batch_capacity=None,
    seed=None,
    name=None
)
```



Defined in [`text/skip_gram_ops.py`](https://github.com/tensorflow/addons/tree/r0.3/tensorflow_addons/text/skip_gram_ops.py).

<!-- Placeholder for "Used in" -->
tensor.

Generates skip-gram `("token", "label")` pairs using each element in the
rank-1 `input_tensor` as a token. The window size used for each token will
be randomly selected from the range specified by `[min_skips, max_skips]`,
inclusive. See https://arxiv.org/abs/1301.3781 for more details about
skip-gram.

For example, given `input_tensor = ["the", "quick", "brown", "fox",
"jumps"]`, `min_skips = 1`, `max_skips = 2`, `emit_self_as_target = False`,
the output `(tokens, labels)` pairs for the token "quick" will be randomly
selected from either `(tokens=["quick", "quick"], labels=["the", "brown"])`
for 1 skip, or `(tokens=["quick", "quick", "quick"],
labels=["the", "brown", "fox"])` for 2 skips.

If `emit_self_as_target = True`, each token will also be emitted as a label
for itself. From the previous example, the output will be either
`(tokens=["quick", "quick", "quick"], labels=["the", "quick", "brown"])`
for 1 skip, or `(tokens=["quick", "quick", "quick", "quick"],
labels=["the", "quick", "brown", "fox"])` for 2 skips.

The same process is repeated for each element of `input_tensor` and
concatenated together into the two output rank-1 `Tensors` (one for all the
tokens, another for all the labels).

If `vocab_freq_table` is specified, tokens in `input_tensor` that are not
present in the vocabulary are discarded. Tokens whose frequency counts are
below `vocab_min_count` are also discarded. Tokens whose frequency
proportions in the corpus exceed `vocab_subsampling` may be randomly
down-sampled. See Eq. 5 in http://arxiv.org/abs/1310.4546 for more details
about subsampling.

Due to the random window sizes used for each token, the lengths of the
outputs are non-deterministic, unless `batch_size` is specified to batch
the outputs to always return `Tensors` of length `batch_size`.

#### Args:

* <b>`input_tensor`</b>: A rank-1 `Tensor` from which to generate skip-gram
    candidates.
* <b>`min_skips`</b>: `int` or scalar `Tensor` specifying the minimum window size to
    randomly use for each token. Must be >= 0 and <= `max_skips`. If
    `min_skips` and `max_skips` are both 0, the only label outputted will
    be the token itself when `emit_self_as_target = True` -
    or no output otherwise.
* <b>`max_skips`</b>: `int` or scalar `Tensor` specifying the maximum window size to
    randomly use for each token. Must be >= 0.
* <b>`start`</b>: `int` or scalar `Tensor` specifying the position in
    `input_tensor` from which to start generating skip-gram candidates.
* <b>`limit`</b>: `int` or scalar `Tensor` specifying the maximum number of
    elements in `input_tensor` to use in generating skip-gram candidates.
    -1 means to use the rest of the `Tensor` after `start`.
* <b>`emit_self_as_target`</b>: `bool` or scalar `Tensor` specifying whether to emit
    each token as a label for itself.
* <b>`vocab_freq_table`</b>: (Optional) A lookup table (subclass of
    `lookup.InitializableLookupTableBase`) that maps tokens to their raw
    frequency counts. If specified, any token in `input_tensor` that is not
    found in `vocab_freq_table` will be filtered out before generating
    skip-gram candidates. While this will typically map to integer raw
    frequency counts, it could also map to float frequency proportions.
    `vocab_min_count` and `corpus_size` should be in the same units
    as this.
* <b>`vocab_min_count`</b>: (Optional) `int`, `float`, or scalar `Tensor` specifying
    minimum frequency threshold (from `vocab_freq_table`) for a token to be
    kept in `input_tensor`. If this is specified, `vocab_freq_table` must
    also be specified - and they should both be in the same units.
* <b>`vocab_subsampling`</b>: (Optional) `float` specifying frequency proportion
    threshold for tokens from `input_tensor`. Tokens that occur more
    frequently (based on the ratio of the token's `vocab_freq_table` value
    to the `corpus_size`) will be randomly down-sampled. Reasonable
    starting values may be around 1e-3 or 1e-5. If this is specified, both
    `vocab_freq_table` and `corpus_size` must also be specified. See Eq. 5
    in http://arxiv.org/abs/1310.4546 for more details.
* <b>`corpus_size`</b>: (Optional) `int`, `float`, or scalar `Tensor` specifying the
    total number of tokens in the corpus (e.g., sum of all the frequency
    counts of `vocab_freq_table`). Used with `vocab_subsampling` for
    down-sampling frequently occurring tokens. If this is specified,
    `vocab_freq_table` and `vocab_subsampling` must also be specified.
* <b>`batch_size`</b>: (Optional) `int` specifying batch size of returned `Tensors`.
* <b>`batch_capacity`</b>: (Optional) `int` specifying batch capacity for the queue
    used for batching returned `Tensors`. Only has an effect if
    `batch_size` > 0. Defaults to 100 * `batch_size` if not specified.
* <b>`seed`</b>: (Optional) `int` used to create a random seed for window size and
    subsampling. See `set_random_seed` docs for behavior.
* <b>`name`</b>: (Optional) A `string` name or a name scope for the operations.


#### Returns:

A `tuple` containing (token, label) `Tensors`. Each output `Tensor` is of
rank-1 and has the same type as `input_tensor`. The `Tensors` will be of
length `batch_size`; if `batch_size` is not specified, they will be of
random length, though they will be in sync with each other as long as
they are evaluated together.


#### Raises:

* <b>`ValueError`</b>: If `vocab_freq_table` is not provided, but `vocab_min_count`,
    `vocab_subsampling`, or `corpus_size` is specified.
    If `vocab_subsampling` and `corpus_size` are not both present or
    both absent.