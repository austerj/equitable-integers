import json
from pathlib import Path

import pytest

from eqint.solver import Bounds


@pytest.fixture(scope="session")
def cases() -> tuple[Bounds, ...]:
    with open(Path("tests") / "cases.json") as f:
        return tuple(tuple(tuple(bounds) for bounds in bounds_case) for bounds_case in json.load(f))
