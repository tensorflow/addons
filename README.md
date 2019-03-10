# TensorFlow Addons

[![PyPI Status Badge](https://badge.fury.io/py/tensorflow-addons.svg)](https://pypi.org/project/tensorflow-addons/)
[![Gitter chat](https://img.shields.io/badge/chat-on%20gitter-46bc99.svg)](https://gitter.im/tensorflow/sig-addons)

TensorFlow Addons is a repository of contributions that conform to
well-established API patterns, but implement new functionality
not available in core TensorFlow. TensorFlow natively supports
a large number of operators, layers, metrics, losses, and optimizers.
However, in a fast moving field like ML, there are many interesting new
developments that cannot be integrated into core TensorFlow
(because their broad applicability is not yet clear, or it is mostly
 used by a smaller subset of the community).

## Contents
| Sub-Package    | Addon  | Reference                                  |
|:----------------------- |:----------- |:---------------------------- |
| tfa.activations | Sparsemax | https://arxiv.org/abs/1602.02068    |
| tfa.image | transform |                                           |
| tfa.layers | GroupNormalization | https://arxiv.org/abs/1803.08494 |
| tfa.layers | InstanceNormalization | https://arxiv.org/abs/1607.08022 |
| tfa.layers | LayerNormalization  | https://arxiv.org/abs/1607.06450 |
| tfa.layers | Maxout | https://arxiv.org/abs/1302.4389             |
| tfa.layers | PoinareNormalize | https://arxiv.org/abs/1705.08039  |
| tfa.layers | WeightNormalization | https://arxiv.org/abs/1602.07868 |
| tfa.losses | LiftedStructLoss | https://arxiv.org/abs/1511.06452       |
| tfa.losses | SparsemaxLoss | https://arxiv.org/abs/1602.02068 | 
| tfa.losses | TripletSemiHardLoss | https://arxiv.org/abs/1503.03832       |
| tfa.optimizers | LazyAdamOptimizer | https://arxiv.org/abs/1412.6980 |
| tfa.text | skip_gram_sample | https://arxiv.org/abs/1301.3781 |

## Core Concepts

#### Standardized APIs
User experience and project maintainability are core concepts in
TF-Addons. In order to achieve these we require that our additions
conform to established API patterns seen in core TensorFlow. Below is
the list we adhere to:


1) [Layers](tensorflow_addons/layers/README.md)
1) [Optimizers](tensorflow_addons/optimizers/README.md)
1) [Losses](tensorflow_addons/losses/README.md)
1) [Custom Ops](tensorflow_addons/custom_ops/README.md)

#### Periodic Evaluation
Based on the nature of this repository, there will be contributions that
in time become dated and unused. In order to keep the project
maintainable, SIG-Addons will perform periodic reviews and deprecate
contributions which will be slated for removal. More information will
be available after we submit a formal request for comment.


## Examples
See [`tensorflow_addons/examples/`](tensorflow_addons/examples/)
for end-to-end examples of various addons.

## Installation
#### Stable Builds
To install the latest version, run the following:
```
pip install tensorflow-addons
```

**Note:** You will also need [TensorFlow 2.0 or higher](https://www.tensorflow.org/alpha). 

To use addons:

```python
import tensorflow as tf
import tensorflow_addons as tfa
```

#### Installing from Source
You can also install from source. This requires the [Bazel](
https://bazel.build/) build system.

```
git clone https://github.com/tensorflow/addons.git
cd addons

# This script links project with TensorFlow dependency
./configure.sh

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_addons-*.whl
```

## Contributing
TF-Addons is a community led open source project. As such, the project
depends on public contributions, bug-fixes, and documentation. Please
see [contribution guidelines](CONTRIBUTING.md) for a guide on how to
contribute. This project adheres to [TensorFlow's code of conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## Community
* [Public Mailing List](https://groups.google.com/a/tensorflow.org/forum/#!forum/addons)
* [SIG Monthly Meeting Notes](https://docs.google.com/document/d/1kxg5xIHWLY7EMdOJCdSGgaPu27a9YKpupUz2VTXqTJg)
    * Join our mailing list and receive calendar invites to the meeting

## License
[Apache License 2.0](LICENSE)
