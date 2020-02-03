#!/bin/bash

docker run --name tf_addons -it -v ${PWD}:/addons -w /addons gcr.io/tensorflow-testing/nosla-ubuntu16.04-manylinux2010 /bin/bash
