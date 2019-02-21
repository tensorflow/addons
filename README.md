# TensorFlow Addons

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
| addons.image | Transform |                                           |
| addons.layers | Maxout | https://arxiv.org/abs/1302.4389             |
| addons.layers | PoinareNormalize | https://arxiv.org/abs/1705.08039  |
| addons.layers | WeightNormalization | https://arxiv.org/abs/1602.07868 |
| addons.losses | TripletLoss | https://arxiv.org/abs/1503.03832       |
| addons.optimizers | LazyAdamOptimizer | https://arxiv.org/abs/1412.6980 |
| addons.text | SkipGrams | https://arxiv.org/abs/1301.3781 |

## Core Concepts

#### Standardized APIs
User experience and project maintainability are core concepts in
TF-Addons. In order to achieve these we require that our additions
conform to established API patterns seen in core TensorFlow. Below is
the list we adhere to:


1) [Layers](tensorflow_addons/layers/README.md)
1) [Optimizers](tensorflow_addons/optimizers/README.md)
1) [Losses](tensorflow_addons/losses/README.md)
1) Custom Ops

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
`tensorflow-addons` will soon be available in PyPi.

#### Installing from Source
You can also install from source. This requires the [Bazel](
https://bazel.build/) build system.

```
git clone https://github.com/tensorflow/addons.git
cd addons

# This script tells bazel where the tensorflow dependency can be found
./configure.sh  # Links project with TensorFlow dependency

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifact

pip install artifacts/tensorflow_addons-*.whl
```

## Contributing
TF-Addons is a community led open source project. As such, the project
depends on public contributions, bug-fixes, and documentation. Please
see [CONTRIBUTING.md](CONTRIBUTING.md) for a guide on how to contribute.
This project adheres to [TensorFlow's code of conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## Community
* [Public Mailing List](https://groups.google.com/a/tensorflow.org/forum/#!forum/addons)
* [SIG Monthly Meeting Notes](https://docs.google.com/document/d/1kxg5xIHWLY7EMdOJCdSGgaPu27a9YKpupUz2VTXqTJg)

## License
[Apache License 2.0](LICENSE)
