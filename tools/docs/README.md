Generate API documents

```bash
# Install dependencies:
pip install -r doc_requirements.txt

# Build tool:
bazel build tools/docs:build_docs

# Generate API doc:
# Use current branch
bazel-bin/tools/docs/build_docs --git_branch=$(git rev-parse --abbrev-ref HEAD)
# or specified explicitly
bazel-bin/tools/docs/build_docs --git_branch=master --output_dir=docs/api_docs/python/

# Release API doc:
git add -f doc/
git commit -m "DOC: xxxxx"
git push
```
