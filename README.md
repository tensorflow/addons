<div align="center">
  <img src="https://github.com/tensorflow/community/blob/master/sigs/logos/SIGAddons.png" width="60%"><br><br>
</div>

-----------------

[![PyPI Status Badge](https://badge.fury.io/py/tensorflow-addons.svg)](https://pypi.org/project/tensorflow-addons/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tensorflow-addons)](https://pypi.org/project/tensorflow-addons/)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://www.tensorflow.org/addons/api_docs/python/tfa)
[![Gitter chat](https://img.shields.io/badge/chat-on%20gitter-46bc99.svg)](https://gitter.im/tensorflow/sig-addons)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### Continuous Build Status

| Build      | Status |
| ---             | ---    |
| **Ubuntu/macOS/Windows**   | [![Status](https://github.com/tensorflow/addons/workflows/addons-release/badge.svg)](https://github.com/tensorflow/addons/actions?query=workflow%3Aaddons-release) |
| **Ubuntu GPU custom ops**   | [![Status](https://storage.googleapis.com/tensorflow-kokoro-build-badges/addons/ubuntu-gpu-py3.svg)](https://storage.googleapis.com/tensorflow-kokoro-build-badges/addons/ubuntu-gpu-py3.html) |

**TensorFlow Addons** is a repository of contributions that conform to
well-established API patterns, but implement new functionality
not available in core TensorFlow. TensorFlow natively supports
a large number of operators, layers, metrics, losses, and optimizers.
However, in a fast moving field like ML, there are many interesting new
developments that cannot be integrated into core TensorFlow
(because their broad applicability is not yet clear, or it is mostly
 used by a smaller subset of the community).

## Addons Subpackages

* [tfa.activations](https://www.tensorflow.org/addons/api_docs/python/tfa/activations) 
* [tfa.callbacks](https://www.tensorflow.org/addons/api_docs/python/tfa/callbacks) 
* [tfa.image](https://www.tensorflow.org/addons/api_docs/python/tfa/image) 
* [tfa.layers](https://www.tensorflow.org/addons/api_docs/python/tfa/layers)
* [tfa.losses](https://www.tensorflow.org/addons/api_docs/python/tfa/losses)
* [tfa.metrics](https://www.tensorflow.org/addons/api_docs/python/tfa/metrics) 
* [tfa.optimizers](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers) 
* [tfa.rnn](https://www.tensorflow.org/addons/api_docs/python/tfa/rnn) 
* [tfa.seq2seq](https://www.tensorflow.org/addons/api_docs/python/tfa/seq2seq) 
* [tfa.text](https://www.tensorflow.org/addons/api_docs/python/tfa/text) 

## Maintainership
The maintainers of TensorFlow Addons can be found in the [CODEOWNERS](.github/CODEOWNERS) file of the repo. This file 
is parsed and pull requests will automatically tag the owners using a bot. If you would
like to maintain something, please feel free to submit a PR. We encourage multiple 
owners for all submodules.

## Installation
#### Stable Builds
TensorFlow Addons is available on PyPI for Linux, macOS, and Windows. To install the latest version, 
run the following:
```
pip install tensorflow-addons
```

To ensure you have a version of TensorFlow that is compatible with TensorFlow Addons, 
you can specify the `tensorflow` extra requirement during install:

```
pip install tensorflow-addons[tensorflow]
```

Similar extras exist for the `tensorflow-gpu` and `tensorflow-cpu` packages.
 

To use TensorFlow Addons:

```python
import tensorflow as tf
import tensorflow_addons as tfa
```

### Python Op Compatility
TensorFlow Addons is actively working towards forward compatibility with TensorFlow 2.x. 
However, there are still a few private API uses within the repository so at the moment 
we can only guarantee compatibility with the TensorFlow versions which it was tested against. 
Warnings will be emitted when importing `tensorflow_addons` if your TensorFlow version does not match 
what it was tested against.

#### Python Op Compatibility Matrix
| TensorFlow Addons | TensorFlow | Python  |
|:----------------------- |:---|:---------- |
| tfa-nightly | 2.7, 2.8, 2.9 | 3.7, 3.8, 3.9, 3.10 | 
| tensorflow-addons-0.17.1 | 2.7, 2.8, 2.9 |3.7, 3.8, 3.9, 3.10 |
| tensorflow-addons-0.16.1 | 2.6, 2.7, 2.8 |3.7, 3.8, 3.9, 3.10 |
| tensorflow-addons-0.15.0 | 2.5, 2.6, 2.7 |3.7, 3.8, 3.9 |
| tensorflow-addons-0.14.0 | 2.4, 2.5, 2.6 |3.6, 3.7, 3.8, 3.9 |
| tensorflow-addons-0.13.0 | 2.3, 2.4, 2.5 |3.6, 3.7, 3.8, 3.9 |
| tensorflow-addons-0.12.1 | 2.3, 2.4 |3.6, 3.7, 3.8 |
| tensorflow-addons-0.11.2 | 2.2, 2.3 |3.5, 3.6, 3.7, 3.8 |
| tensorflow-addons-0.10.0 | 2.2 |3.5, 3.6, 3.7, 3.8 |
| tensorflow-addons-0.9.1 | 2.1, 2.2 |3.5, 3.6, 3.7 |
| tensorflow-addons-0.8.3 | 2.1 |3.5, 3.6, 3.7 |
| tensorflow-addons-0.7.1 | 2.1 | 2.7, 3.5, 3.6, 3.7 | 
| tensorflow-addons-0.6.0 | 2.0 | 2.7, 3.5, 3.6, 3.7 |

### C++ Custom Op Compatibility
TensorFlow C++ APIs are not stable and thus we can only guarantee compatibility with the 
version TensorFlow Addons was built against. It is possible custom ops will work with multiple 
versions of TensorFlow, but there is also a chance for segmentation faults or other problematic crashes.
Warnings will be emitted when loading a custom op if your TensorFlow version does not match 
what it was built against.

Additionally, custom ops registration does not have a stable ABI interface so it is 
required that users have a compatible installation of TensorFlow even if the versions 
match what we had built against. A simplification of this is that **TensorFlow Addons 
custom ops will work with `pip`-installed TensorFlow** but will have issues when TensorFlow 
is compiled differently. A typical example of this would be `conda`-installed TensorFlow.
[RFC #133](https://github.com/tensorflow/community/pull/133) aims to fix this.


#### C++ Custom Op Compatibility Matrix
| TensorFlow Addons | TensorFlow | Compiler  | cuDNN | CUDA | 
|:----------------------- |:---- |:---------|:---------|:---------|
| tfa-nightly | 2.9 | GCC 9.3.1 | 8.1 | 11.2 |
| tensorflow-addons-0.17.1 | 2.9  | GCC 9.3.1 | 8.1 | 11.2 |
| tensorflow-addons-0.16.1 | 2.8  | GCC 7.3.1 | 8.1 | 11.2 |
| tensorflow-addons-0.15.0 | 2.7  | GCC 7.3.1 | 8.1 | 11.2 |
| tensorflow-addons-0.14.0 | 2.6  | GCC 7.3.1 | 8.1 | 11.2 |
| tensorflow-addons-0.13.0 | 2.5  | GCC 7.3.1 | 8.1 | 11.2 |
| tensorflow-addons-0.12.1 | 2.4  | GCC 7.3.1 | 8.0 | 11.0 |
| tensorflow-addons-0.11.2 | 2.3  | GCC 7.3.1 | 7.6 | 10.1 |
| tensorflow-addons-0.10.0 | 2.2  | GCC 7.3.1 | 7.6 | 10.1 |
| tensorflow-addons-0.9.1 | 2.1  | GCC 7.3.1 | 7.6 | 10.1 |
| tensorflow-addons-0.8.3 | 2.1  | GCC 7.3.1 | 7.6 | 10.1 |
| tensorflow-addons-0.7.1 | 2.1  | GCC 7.3.1 | 7.6 | 10.1 |
| tensorflow-addons-0.6.0 | 2.0  | GCC 7.3.1 | 7.4 | 10.0 |


#### Nightly Builds
There are also nightly builds of TensorFlow Addons under the pip package
`tfa-nightly`, which is built against **the latest stable version of TensorFlow**. Nightly builds
include newer features, but may be less stable than the versioned releases. Contrary to 
what the name implies, nightly builds are not released every night, but at every commit 
of the master branch. `0.9.0.dev20200306094440` means that the commit time was 
2020/03/06 at 09:44:40 Coordinated Universal Time.

```
pip install tfa-nightly
```

#### Installing from Source
You can also install from source. This requires the [Bazel](
https://bazel.build/) build system (version >= 1.0.0).

##### CPU Custom Ops
```
git clone https://github.com/tensorflow/addons.git
cd addons

# This script links project with TensorFlow dependency
python3 ./configure.py

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_addons-*.whl
```

##### GPU and CPU Custom Ops
```
git clone https://github.com/tensorflow/addons.git
cd addons

export TF_NEED_CUDA="1"

# Set these if the below defaults are different on your system
export TF_CUDA_VERSION="11"
export TF_CUDNN_VERSION="8"
export CUDA_TOOLKIT_PATH="/usr/local/cuda"
export CUDNN_INSTALL_PATH="/usr/lib/x86_64-linux-gnu"

# This script links project with TensorFlow dependency
python3 ./configure.py

bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts

pip install artifacts/tensorflow_addons-*.whl
```

## Tutorials
See [`docs/tutorials/`](docs/tutorials/)
for end-to-end examples of various addons.

## Core Concepts

#### Standardized API within Subpackages
User experience and project maintainability are core concepts in
TensorFlow Addons. In order to achieve these we require that our additions
conform to established API patterns seen in core TensorFlow.

#### GPU and CPU Custom Ops
TensorFlow Addons supports precompiled custom ops for CPU and GPU. However, 
GPU custom ops currently only work on Linux distributions. For this reason Windows and macOS 
will fallback to pure TensorFlow Python implementations whenever possible.

The order of priority on macOS/Windows is:
1) Pure TensorFlow + Python implementation (works on CPU and GPU)
2) C++ implementation for CPU

The order of priority on Linux is:
1) CUDA implementation
2) C++ implementation
3) Pure TensorFlow + Python implementation (works on CPU and GPU)

If you want to change the default priority, "C++ and CUDA" VS "pure TensorFlow Python",
you can set the environment variable `TF_ADDONS_PY_OPS=1` from the command line or
run `tfa.options.disable_custom_kernel()` in your code.

For example, if you are on Linux and you have compatibility problems with the compiled ops,
you can give priority to the Python implementations:

From the command line:
```bash
export TF_ADDONS_PY_OPS=1
```

or in your code:

```python
import tensorflow_addons as tfa
tfa.options.disable_custom_kernel()
```

This variable defaults to `True` on Windows and macOS, and `False` on Linux.

#### Proxy Maintainership
TensorFlow Addons has been designed to compartmentalize submodules so 
that they can be maintained by community users who have expertise, and a vested 
interest in that component. We heavily encourage users to submit sign up to maintain a 
submodule by submitting your username to the [CODEOWNERS](.github/CODEOWNERS) file.

Full write access will only be granted after substantial contribution 
has been made in order to limit the number of users with write permission. 
Contributions can come in the form of issue closings, bug fixes, documentation, 
new code, or optimizing existing code. Submodule maintainership can be granted 
with a lower barrier for entry as this will not include write permissions to 
the repo.

For more information see [the RFC](https://github.com/tensorflow/community/blob/master/rfcs/20190308-addons-proxy-maintainership.md) 
on this topic.

#### Periodic Evaluation of Subpackages
Given the nature of this repository, submodules may become less 
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
TensorFlow Addons is a community-led open source project (only a few maintainers work for Google!). 
As such, the project depends on public contributions, bug fixes, and documentation. 
This project adheres to [TensorFlow's code of conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

Do you want to contribute but are not sure of what? Here are a few suggestions:
1. Add a new tutorial. Located in [`docs/tutorials/`](docs/tutorials),
  these are a great way to familiarize yourself and others with TensorFlow Addons. See
  [the guidelines](docs/tutorials/README.md) for more information on how to add
  examples.
2. Improve the docstrings. The docstrings are fetched and then displayed in the documentation.
  Do a change and hundreds of developers will see it and benefit from it. Maintainers are often focused 
  on making APIs, fixing bugs and other code related changes. The documentation will never 
  be loved enough!
3. Solve an [existing issue](https://github.com/tensorflow/addons/issues).
  These range from low-level software bugs to higher-level design problems.
  Check out the label [help wanted](https://github.com/tensorflow/addons/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22). If you're a new contributor, the label [good first issue](https://github.com/tensorflow/addons/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) can be a good place to start.
4. Review a pull request. So you're not a software engineer but you know a lot
  about a certain field a research? That's awesome and we need your help! Many people 
  are submitting pull requests to add layers/optimizers/functions taken from recent
  papers. Since TensorFlow Addons maintainers are not specialized in everything,
  you can imagine how hard it is to review. It takes very long to read the paper,
  understand it and check the math in the pull request. If you're specialized, look at 
  the [list of pull requests](https://github.com/tensorflow/addons/pulls). 
  If there is something from a paper you know, please comment on the pull request to
  check the math is ok. If you see that everything is good, say it! It will help 
  the maintainers to sleep better at night knowing that he/she wasn't the only
  person to approve the pull request.
5. You have an opinion and want to share it? The docs are not very helpful for 
  a function or a class? You tried to open a pull request but you didn't manage to 
  install or test anything and you think it's too complicated? You made a pull request
  but you didn't find the process good enough and it made no sense to you? Please 
  say it! We want feedback. Maintainers are too much the head into the code 
  to understand what it's like for someone new to open source to come to this project. 
  If you don't understand something, be aware there are no people who are 
  bad at understanding, there are just bad tutorials and bad guides.

Please see [contribution guidelines](CONTRIBUTING.md) to get started (and remember,
if you don't understand something, open an issue, or even make a pull request to 
improve the guide!).

## Community
* [Public Mailing List](https://groups.google.com/a/tensorflow.org/forum/#!forum/addons)
* [SIG Monthly Meeting Notes](https://docs.google.com/document/d/1kxg5xIHWLY7EMdOJCdSGgaPu27a9YKpupUz2VTXqTJg)
    * Join our mailing list and receive calendar invites to the meeting

## License
[Apache License 2.0](LICENSE)

