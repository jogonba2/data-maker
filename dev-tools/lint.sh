#!/usr/bin/env bash

set -e
set -x

mypy "datamaker"
flake8 "datamaker" --ignore=E501,W503,E203,E402,E704
black "datamaker" --check -l 80
