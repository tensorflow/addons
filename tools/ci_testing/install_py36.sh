#!/usr/bin/env bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
# Needed until docker image defaults to at least py35.

set -e -x

curl -sSOL https://bootstrap.pypa.io/get-pip.py
add-apt-repository -y ppa:deadsnakes/ppa

apt-get -y -qq update && apt-get -y -qq install python3.6

python3.6 get-pip.py -q
python3.6 -m pip --version

ln -sfn /usr/bin/python3.6 /usr/bin/python3
pip3 install scipy  # Pre-installed in custom-op