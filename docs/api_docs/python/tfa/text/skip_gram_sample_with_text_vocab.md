<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.text.skip_gram_sample_with_text_vocab" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.text.skip_gram_sample_with_text_vocab

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/text/skip_gram_ops.py#L206-L386">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
Skip-gram sampling with a text vocabulary file.

**Aliases**: `tfa.text.skip_gram_ops.skip_gram_sample_with_text_vocab`

``` python
tfa.text.skip_gram_sample_with_text_vocab(
    input_tensor,
    vocab_freq_file,
    vocab_token_index=0,
    vocab_token_dtype=tf.dtypes.string,
    vocab_freq_index=1,
    vocab_freq_dtype=tf.dtypes.float64,
    vocab_delimiter=',',
    vocab_min_count=0,
    vocab_subsampling=None,
    corpus_size=None,
    min_skips=1,
    max_skips=5,
    start=0,
    limit=-1,
    emit_self_as_target=False,
    batch_size=None,
    batch_capacity=None,
    seed=None,
    name=None
)
```



<!-- Placeholder for "Used in" -->

Wrapper around `skip_gram_sample()` for use with a text vocabulary file.
The vocabulary file is expected to be a plain-text file, with lines of
`vocab_delimiter`-separated columns. The `vocab_token_index` column should
contain the vocabulary term, while the `vocab_freq_index` column should
contain the number of times that term occurs in the corpus. For example,
with a text vocabulary file of:

  ```
  bonjour,fr,42
  hello,en,777
  hola,es,99
  ```

You should set `vocab_delimiter=","`, `vocab_token_index=0`, and
`vocab_freq_index=2`.

See `skip_gram_sample()` documentation for more details about the skip-gram
sampling process.

#### Args:


* <b>`input_tensor`</b>:   A rank-1 `Tensor` from which to generate skip-gram candidates.
* <b>`vocab_freq_file`</b>:   `string` specifying full file path to the text vocab file.
* <b>`vocab_token_index`</b>: `int` specifying which column in the text vocab file
  contains the tokens.
* <b>`vocab_token_dtype`</b>:   `DType` specifying the format of the tokens in the text vocab file.
* <b>`vocab_freq_index`</b>: `int` specifying which column in the text vocab file
  contains the frequency counts of the tokens.
* <b>`vocab_freq_dtype`</b>: `DType` specifying the format of the frequency counts
  in the text vocab file.
* <b>`vocab_delimiter`</b>: `string` specifying the delimiter used in the text vocab
  file.
* <b>`vocab_min_count`</b>: `int`, `float`, or scalar `Tensor` specifying
  minimum frequency threshold (from `vocab_freq_file`) for a token to be
  kept in `input_tensor`. This should correspond with `vocab_freq_dtype`.
* <b>`vocab_subsampling`</b>: (Optional) `float` specifying frequency proportion
  threshold for tokens from `input_tensor`. Tokens that occur more
  frequently will be randomly down-sampled. Reasonable starting values
  may be around 1e-3 or 1e-5. See Eq. 5 in http://arxiv.org/abs/1310.4546
  for more details.
* <b>`corpus_size`</b>: (Optional) `int`, `float`, or scalar `Tensor` specifying the
  total number of tokens in the corpus (e.g., sum of all the frequency
  counts of `vocab_freq_file`). Used with `vocab_subsampling` for
  down-sampling frequently occurring tokens. If this is specified,
  `vocab_freq_file` and `vocab_subsampling` must also be specified.
  If `corpus_size` is needed but not supplied, then it will be calculated
  from `vocab_freq_file`. You might want to supply your own value if you
  have already eliminated infrequent tokens from your vocabulary files
  (where frequency < vocab_min_count) to save memory in the internal
  token lookup table. Otherwise, the unused tokens' variables will waste
  memory.  The user-supplied `corpus_size` value must be greater than or
  equal to the sum of all the frequency counts of `vocab_freq_file`.
* <b>`min_skips`</b>: `int` or scalar `Tensor` specifying the minimum window size to
  randomly use for each token. Must be >= 0 and <= `max_skips`. If
  `min_skips` and `max_skips` are both 0, the only label outputted will
  be the token itself.
* <b>`max_skips`</b>: `int` or scalar `Tensor` specifying the maximum window size to
  randomly use for each token. Must be >= 0.
* <b>`start`</b>: `int` or scalar `Tensor` specifying the position in `input_tensor`
  from which to start generating skip-gram candidates.
* <b>`limit`</b>: `int` or scalar `Tensor` specifying the maximum number of elements
  in `input_tensor` to use in generating skip-gram candidates. -1 means
  to use the rest of the `Tensor` after `start`.
* <b>`emit_self_as_target`</b>: `bool` or scalar `Tensor` specifying whether to emit
  each token as a label for itself.
* <b>`batch_size`</b>: (Optional) `int` specifying batch size of returned `Tensors`.
* <b>`batch_capacity`</b>: (Optional) `int` specifying batch capacity for the queue
  used for batching returned `Tensors`. Only has an effect if
  `batch_size` > 0. Defaults to 100 * `batch_size` if not specified.
* <b>`seed`</b>: (Optional) `int` used to create a random seed for window size and
  subsampling. See
  [`set_random_seed`](../../g3doc/python/constant_op.md#set_random_seed)
  for behavior.
* <b>`name`</b>: (Optional) A `string` name or a name scope for the operations.


#### Returns:

A `tuple` containing (token, label) `Tensors`. Each output `Tensor` is of
rank-1 and has the same type as `input_tensor`. The `Tensors` will be of
length `batch_size`; if `batch_size` is not specified, they will be of
random length, though they will be in sync with each other as long as
they are evaluated together.



#### Raises:


* <b>`ValueError`</b>: If `vocab_token_index` or `vocab_freq_index` is less than 0
  or exceeds the number of columns in `vocab_freq_file`.
  If `vocab_token_index` and `vocab_freq_index` are both set to the same
  column. If any token in `vocab_freq_file` has a negative frequency.

