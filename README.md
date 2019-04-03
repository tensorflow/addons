<div align="center">
  <img src="static/SIGAddons.png" width="60%"><br><br>
</div>

-----------------

[![PyPI Status Badge](https://badge.fury.io/py/tensorflow-addons.svg)](https://pypi.org/project/tensorflow-addons/)
[![Gitter chat](https://img.shields.io/badge/chat-on%20gitter-46bc99.svg)](https://gitter.im/tensorflow/sig-addons)

### Official Builds

| Build Type      | Status |
| ---             | ---    |
| **Linux Py2 CPU**   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/addons/ubuntu-py2.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/addons/ubuntu-py2.html) |
| **Linux Py3 CPU**   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/addons/ubuntu-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/addons/ubuntu-py3.html) |
| **Linux Py2 GPU**   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/addons/ubuntu-gpu-py2.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/addons/ubuntu-gpu-py2.html) |
| **Linux Py3 GPU**   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/addons/ubuntu-gpu-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/addons/ubuntu-gpu-py3.html) |
| **Linux Sanity Check**   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/addons/ubuntu-sanity.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/addons/ubuntu-sanity.html) |

**TensorFlow Addons** is a repository of contributions that conform to
well-established API patterns, but implement new functionality
not available in core TensorFlow. TensorFlow natively supports
a large number of operators, layers, metrics, losses, and optimizers.
However, in a fast moving field like ML, there are many interesting new
developments that cannot be integrated into core TensorFlow
(because their broad applicability is not yet clear, or it is mostly
 used by a smaller subset of the community).

## Maintainers
| Subpackage    | Maintainers  | Contact Info                        |
|:----------------------- |:----------- |:----------------------------|
| [tfa.activations](tensorflow_addons/activations/README.md) | SIG-Addons | addons@tensorflow.org    |
| [tfa.image](tensorflow_addons/image/README.md) |  |                                   |
| [tfa.layers](tensorflow_addons/layers/README.md) | SIG-Addons |     addons@tensorflow.org |
| [tfa.losses](tensorflow_addons/losses/README.md) | SIG-Addons |     addons@tensorflow.org |
| [tfa.optimizers](tensorflow_addons/optimizers/README.md) | SIG-Addons | addons@tensorflow.org |
| [tfa.seq2seq](tensorflow_addons/seq2seq/README.md) | Google | @qlzh727 | 
| [tfa.text](tensorflow_addons/text/README.md) |  |  |

## Core Concepts

#### Standardized API within Subpackages
User experience and project maintainability are core concepts in
TF-Addons. In order to achieve these we require that our additions
conform to established API patterns seen in core TensorFlow.

#### Periodic Evaluation of Subpackages
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

**Notice**: If you are using Mac OS X, you need install addition software by `brew install coreutils` (this requires the [Homebrew](https://brew.sh/)) or `port install coreutils` (this requires the [MacPorts](https://www.macports.org/))

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
