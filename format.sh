#!/bin/sh

set -e

# formatting flags
black_flags=""
isort_flags=""
poetry_flags="--remove-all-unused-imports --verbose --recursive --in-place --exclude=__init__.py"

echo "Formatting..."
set -x
poetry run isort . $isort_flags
poetry run black . $black_flags
poetry run autoflake $poetry_flags .
