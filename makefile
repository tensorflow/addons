.PHONY: all

all: auto-format sanity-check unit-test

# TODO: install those dependencies in docker image (dockerfile).
install-ci-dependency:
	bash tools/ci_build/install/install_ci_dependency.sh --quiet

code-format: install-ci-dependency
	bash tools/ci_build/code_format.sh --incremental --in-place

sanity-check: install-ci-dependency
	bash tools/ci_build/ci_sanity.sh --incremental

unit-test:
	# Use default configuration here.
	yes 'y' | ./configure.sh
	bazel test //tensorflow_addons/...
