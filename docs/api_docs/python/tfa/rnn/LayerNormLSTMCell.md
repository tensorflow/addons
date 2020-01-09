<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tfa.rnn.LayerNormLSTMCell" />
<meta itemprop="path" content="Stable" />
</div>

# tfa.rnn.LayerNormLSTMCell

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/tensorflow/addons/tree/r0.7/tensorflow_addons/rnn/cell.py#L213-L368">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td></table>



<!-- Equality marker -->
## Class `LayerNormLSTMCell`

LSTM cell with layer normalization and recurrent dropout.



**Aliases**: `tfa.rnn.cell.LayerNormLSTMCell`

<!-- Placeholder for "Used in" -->

This class adds layer normalization and recurrent dropout to a LSTM unit.
Layer normalization implementation is based on:

  https://arxiv.org/abs/1607.06450.

"Layer Normalization" Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

and is applied before the internal nonlinearities.
Recurrent dropout is based on:

  https://arxiv.org/abs/1603.05118

"Recurrent Dropout without Memory Loss"
Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.



