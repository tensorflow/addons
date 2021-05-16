## 1. Generated API docs

[tensorflow.org/addons/api_docs/python/tfa](https://tensorflow.org/addons/api_docs/python/tfa)

`build_docs.py` controls executed this docs generation. To test-run it:

```bash
# Install dependencies:
pip install -r tools/install_deps/doc_requirements.txt

# Build tool:
bazel build //tools/docs:build_docs

# Generate API doc:
# Use current branch
bazel-bin/tools/docs/build_docs --git_branch=$(git rev-parse --abbrev-ref HEAD)
# or specified explicitly
bazel-bin/tools/docs/build_docs --git_branch=master --output_dir=/tmp/tfa_api
```
