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
  Check out the label [help wanted](https://github.com/tensorflow/addons/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

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
 it's API pattern:
    * [Layers](tensorflow_addons/layers/README.md) |
      [Optimizers](tensorflow_addons/optimizers/README.md) |
      [Losses](tensorflow_addons/losses/README.md) |
      Custom Ops

**Note: New contributions often require team-members to read a research
paper and understand how it fits into the TensorFlow community. This
process can take longer than typical commit reviews so please bare with
us**


## Development Environment
It is recommended that development is done in the latest
`nightly-custom-op` docker image.
```
docker run --rm -it -v ${PWD}:/working_dir -w /working_dir tensorflow/tensorflow:nightly-custom-op /bin/bash
```

## Code Testing
#### CI Testing
We're in the process of setting up our nightly CI testing. Because this
project will contain CUDA kernels, we need to make sure that the
hardware will be available from our CI provider.

#### Locally Testing
```
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