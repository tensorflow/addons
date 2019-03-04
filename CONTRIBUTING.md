# Contributing

Interested in contributing to TensorFlow Addons? We appreciate all kinds
of help and are working to make this guide as comprehensive as possible.
Please [let us know](https://github.com/tensorflow/addons/issues) if
you think of something we could do to help lower the barrier to
contributing.

## Pull Requests

We gladly welcome [pull requests](
https://help.github.com/articles/about-pull-requests/).

Before making any changes, we recommend opening an issue (if it
doesn't already exist) and discussing your proposed changes. This will
let us give you advice on the proposed changes. If the changes are
minor, then feel free to make them without discussion.

Want to contribute but not sure of what? Here are a few suggestions:
1. Add a new example or tutorial. Located in [`tensorflow_addons/examples/`](tensorflow_addons/examples),
  these are a great way to familiarize yourself and others with TF-Addons.
2. Solve an [existing issue](https://github.com/tensorflow/addons/issues).
  These range from low-level software bugs to higher-level design problems.
  Check out the label [help wanted](https://github.com/tensorflow/addons/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22). If you're a new contributor, the label [good first issue](https://github.com/tensorflow/addons/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) can be a good place to start.

All submissions, including submissions by project members, require
review.

## Requirements for New Contributions to the Repository
The tensorflow/addons repository contains additional functionality
fitting the following criteria:

* The functionality is not otherwise available in TensorFlow
* Addons have to be compatible with TensorFlow 2.x.
* The addon conforms to the code and documentation standards
* The addon is impactful to the community (e.g. an implementation used
 in widely cited paper)
 * Lastly, the functionality conforms to the contribution guidelines of
 its API pattern:
    * [Layers](tensorflow_addons/layers/README.md) |
      [Optimizers](tensorflow_addons/optimizers/README.md) |
      [Losses](tensorflow_addons/losses/README.md) |
      [Custom Ops](tensorflow_addons/custom_ops/README.md)

**Note: New contributions often require team-members to read a research
paper and understand how it fits into the TensorFlow community. This
process can take longer than typical commit reviews so please bare with
us**


## Development Environment
It is recommended that development is done in the latest
`nightly-custom-op` docker image.

```bash
docker run --rm -it -v ${PWD}:/addons -w /addons tensorflow/tensorflow:nightly-custom-op /bin/bash
```

Try those commands below:

0. Format codes automatically: `make code-format`
1. Sanity check: `make sanity-check`
2. Run unit test: `make unit-test`
3. All of the above: `make`

## Coding style

Addons provides `make code-format` command to format your changes
automatically, don't forget to use it before pushing your codes.

#### C++
C++ code should conform to [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

Addons uses [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
to check your C/C++ changes. Sometimes you have some manually formatted
code that you don’t want clang-format to touch.
You can disable formatting like this:

```cpp
int formatted_code;
// clang-format off
    void    unformatted_code  ;
// clang-format on
void formatted_code_again;
```

#### Python
Python code should conform to [PEP8](https://www.python.org/dev/peps/pep-0008/).

Addons uses [yapf](https://github.com/google/yapf) to format code,
and [pylint](https://www.pylint.org/) for code analysis.
You can disable them locally like this:

```python
# yapf: disable
FOO = {
    # ... some very large, complex data literal.
}

BAR = [
    # ... another large data literal.
]
# yapf: enable
```

```python
# pylint: disable=protected-access
foo._protected_member
# pylint: enable=protected-access
```

## Code Testing
#### CI Testing
We're in the process of setting up our nightly CI testing. Because this
project will contain CUDA kernels, we need to make sure that the
hardware will be available from our CI provider.

#### Locally Testing

```bash
docker run --rm -it -v ${PWD}:/addons -w /addons tensorflow/tensorflow:nightly-custom-op make unit-test
```

or run manually:

```bash
docker run --rm -it -v ${PWD}:/addons -w /addons tensorflow/tensorflow:nightly-custom-op /bin/bash

./configure.sh  # Links project with TensorFlow dependency

bazel test -c opt -k \
--test_timeout 300,450,1200,3600 \
--test_output=errors \
//tensorflow_addons/...
```

## Code Reviews

All submissions, including submissions by project members, require review. We
use Github pull requests for this purpose.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to https://cla.developers.google.com/ to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.
