#/bin/bash

set -e

echo "\n---- CODE STYLE CHECK ----\n"

echo "Running black\n"
poetry run black --check --diff .

echo "Running pylint on package\n"
poetry run pylint src/sparkml_base_classes

echo "---- UNIT TESTS ----\n"
poetry run pytest .
