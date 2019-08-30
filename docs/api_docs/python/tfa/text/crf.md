<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.text.crf" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tfa.text.crf


<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.5/tensorflow_addons/text/crf.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>





<!-- Placeholder for "Used in" -->


## Classes

[`class CrfDecodeForwardRnnCell`](../../tfa/text/crf/CrfDecodeForwardRnnCell.md): Computes the forward decoding in a linear-chain CRF.

## Functions

[`crf_binary_score(...)`](../../tfa/text/crf_binary_score.md): Computes the binary scores of tag sequences.

[`crf_decode(...)`](../../tfa/text/crf_decode.md): Decode the highest scoring sequence of tags in TensorFlow.

[`crf_decode_backward(...)`](../../tfa/text/crf_decode_backward.md): Computes backward decoding in a linear-chain CRF.

[`crf_decode_forward(...)`](../../tfa/text/crf_decode_forward.md): Computes forward decoding in a linear-chain CRF.

[`crf_forward(...)`](../../tfa/text/crf_forward.md): Computes the alpha values in a linear-chain CRF.

[`crf_log_likelihood(...)`](../../tfa/text/crf_log_likelihood.md): Computes the log-likelihood of tag sequences in a CRF.

[`crf_log_norm(...)`](../../tfa/text/crf_log_norm.md): Computes the normalization for a CRF.

[`crf_multitag_sequence_score(...)`](../../tfa/text/crf_multitag_sequence_score.md): Computes the unnormalized score of all tag sequences matching

[`crf_sequence_score(...)`](../../tfa/text/crf_sequence_score.md): Computes the unnormalized score for a tag sequence.

[`crf_unary_score(...)`](../../tfa/text/crf_unary_score.md): Computes the unary scores of tag sequences.

[`viterbi_decode(...)`](../../tfa/text/viterbi_decode.md): Decode the highest scoring sequence of tags outside of TensorFlow.

