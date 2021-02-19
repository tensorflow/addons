# [tensorflow.org/addons](https://tensorflow.org/addons)

This directory contains the source for [tensorflow.org/addons](https://tensorflow.org/addons).

It comprises two main components:

## 1. Narrative Docs

Any markdown or notebook files in this directory will be published to tensorflow.org/addons.

`tutorials/_toc.yaml` controls the left-nav on the tutorials tab. Make sure to keep that file up to date.
Notify the tensorflow/docs team if you need to major changes. 

The preferred formatting for TensorFlow notebooks is to use the [tensorflow/docs](https://github.com/tensorflow/docs) [`nbfmt` tool](https://github.com/tensorflow/docs/tree/master/tools/tensorflow_docs/tools). If modifying a tutorial gives you
an unreadable diff use the following commands to re-apply the standard formatting:  

```
pip install git+https://github.com/tensorflow/docs
python -m tensorflow_docs.tools.nbfmt {path to notebook file or directory}
```











