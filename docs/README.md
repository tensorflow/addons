# [tensorflow.org/addons](https://tensorflow.org/addons)

This directory contains the source for [tensorflow.org/addons](https://tensorflow.org/addons).

This is made from two main components:

## 1. Narrative Docs

Any markdown or notebook files in this directory will be published to tensorflow.org/addons.

`tutorials/_toc.yaml` controls the left-nav on the tutorials tab. Make sure to keep that file up to date.
Notify the tensorflow/docs team if you need to major changes. 


## 2. Generated API docs

[tensorflow.org/addons/api_docs/python/tfa](https://tensorflow.org/addons/api_docs/python/tfa)

`build_docs.py` controls executed this docs generation. To test-run it:

```bash
# Install dependencies:
pip install -r tools/install_deps/doc_requirements.txt

# Build tool:
bazel build docs:build_docs

# Generate API doc:
# Use current branch
bazel-bin/docs/build_docs --git_branch=$(git rev-parse --abbrev-ref HEAD)
# or specified explicitly
bazel-bin/docs/build_docs --git_branch=master --output_dir=/tmp/tfa_api
```
