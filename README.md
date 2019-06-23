<div align="center">
  <img src="https://github.com/tensorflow/community/blob/master/sigs/logos/SIGAddons.png" width="60%"><br><br>
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
| [tfa.activations](tensorflow_addons/activations/README.md) | SIG-Addons | @facaiy @seanpmorgan | 
| [tfa.callbacks](tensorflow_addons/callbacks/README.md) | SIG-Addons | @squadrick |
| [tfa.image](tensorflow_addons/image/README.md) | SIG-Addons | @windqaq @facaiy |
| [tfa.layers](tensorflow_addons/layers/README.md) | SIG-Addons | @seanpmorgan @facaiy |
| [tfa.losses](tensorflow_addons/losses/README.md) | SIG-Addons | @facaiy @windqaq   |
| [tfa.metrics](tensorflow_addons/metrics/README.md) | SIG-Addons | @squadrick | 
| [tfa.optimizers](tensorflow_addons/optimizers/README.md) | SIG-Addons | @facaiy @windqaq @squadrick |
| [tfa.rnn](tensorflow_addons/rnn/README.md) | Google | @qlzh727 |
| [tfa.seq2seq](tensorflow_addons/seq2seq/README.md) | Google | @qlzh727 |
| [tfa.text](tensorflow_addons/text/README.md) |  SIG-Addons |  @seanpmorgan @facaiy |

## Installation
#### Stable Builds
To install the latest version, run the following:
```
pip install tensorflow-addons
```
 
**Note:** You will also need [`tensorflow==2.0.0-beta1`](https://www.tensorflow.org/beta) installed.

To use addons:

```python
import tensorflow as tf
import tensorflow_addons as tfa
```

#### Nightly Builds
There are also nightly builds of TensorFlow Addons under the pip package 
`tfa-nightly`, which is built against `tf-nightly-2.0-preview`. Nightly builds 
include newer features, but may be less stable than the versioned releases.

```
pip install tfa-nightly
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

## Examples
See [`examples/`](examples/)
for end-to-end examples of various addons.

## Core Concepts

#### Standardized API within Subpackages
User experience and project maintainability are core concepts in
TF-Addons. In order to achieve these we require that our additions
conform to established API patterns seen in core TensorFlow.

#### Proxy Maintainership
Addons has been designed to compartmentalize subpackages and submodules so 
that they can be maintained by users who have expertise and a vested interest 
in that component. 

Subpackage maintainership will only be granted after substantial contribution 
has been made in order to limit the number of users with write permission. 
Contributions can come in the form of issue closings, bug fixes, documentation, 
new code, or optimizing existing code. Submodule maintainership can be granted 
with a lower barrier for entry as this will not include write permissions to 
the repo.

For more information see [the RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190308-addons-proxy-maintainership.md) 
on this topic.

#### Periodic Evaluation of Subpackages
Given the nature of this repository, subpackages and submodules may become less 
and less useful to the community as time goes on. In order to keep the 
repository sustainable, we'll be performing bi-annual reviews of our code to 
ensure everything still belongs within the repo. Contributing factors to this 
review will be:

1. Number of active maintainers
2. Amount of OSS use
3. Amount of issues or bugs attributed to the code
4. If a better solution is now available

Functionality within TensorFlow Addons can be categorized into three groups:

* **Suggested**: well-maintained API; use is encouraged.
* **Discouraged**: a better alternative is available; the API is kept for 
historic reasons; or the API requires maintenance and is the waiting period 
to be deprecated.
* **Deprecated**: use at your own risk; subject to be deleted.

The status change between these three groups is: 
Suggested <-> Discouraged -> Deprecated.

The period between an API being marked as deprecated and being deleted will be 
90 days. The rationale being:

1. In the event that TensorFlow Addons releases monthly, there will be 2-3 
releases before an API is deleted. The release notes could give user enough 
warning.

2. 90 days gives maintainers ample time to fix their code.


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
