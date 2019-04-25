Generate API documents:

```bash
pip install -r doc_requirements.txt

bazel build tools/docs:build_docs
bazel-bin/tools/docs/build_docs --git_branch=master --output_dir=docs/api_docs/python/
```
