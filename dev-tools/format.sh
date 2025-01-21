#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place "datamaker" --exclude=__init__.py
isort "datamaker"
black "datamaker" -l 80
