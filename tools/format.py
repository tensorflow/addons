#!/usr/bin/env python
from subprocess import check_call, CalledProcessError


def check_bash_call(string):
    check_call(["bash", "-c", string])


files_changed = False

try:
    check_bash_call("python -m black --check ./")
except CalledProcessError:
    check_bash_call("python -m black ./")
    files_changed = True

try:
    check_bash_call("buildifier --check -r .")
except CalledProcessError:
    check_bash_call("buildifier -r .")
    files_changed = True

# todo: find a way to check if files changed
# see https://github.com/DoozyX/clang-format-lint-action for inspiration
check_bash_call(
    "shopt -s globstar && " "clang-format-9 -i --style=google **/*.cc **/*.h"
)


if files_changed:
    print("Some files have changed.")
    print("Please do git add and git commit again")
    exit(1)
