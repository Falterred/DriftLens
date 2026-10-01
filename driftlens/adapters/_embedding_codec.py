"""Shared embedding decode helpers (Step 14)."""

from __future__ import annotations

import base64
import binascii
import math

import numpy as np

from driftlens.errors import InvalidInputError

ITEMSIZE = {"float16": 2, "float32": 4, "float64": 8, "int8": 1}
NUMPY_DTYPE = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
}
NUMPY_LE_CODE = {"float16": "f2", "float32": "f4", "float64": "f8", "int8": "i1"}


def _check_dtype(declared_dtype: str, location: str) -> None:
    if declared_dtype not in NUMPY_DTYPE:
        raise InvalidInputError(
            message=f"unknown embedding dtype {declared_dtype!r}", location=location
        )


def decode_embedding_array(
    value: list, *, declared_dtype: str | None, location: str
) -> tuple[np.ndarray, str]:
    """Decode a JSON array of numbers into a typed vector."""

    if not isinstance(value, list):
        raise InvalidInputError(
            message="embedding value is not an array", location=location
        )
    numbers: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise InvalidInputError(
                message=f"embedding element {index} is not numeric", location=location
            )
        numbers.append(float(item))

    if declared_dtype is None:
        return np.asarray(numbers, dtype=np.float32), "float32"

    _check_dtype(declared_dtype, location)
    if declared_dtype == "int8":
        for index, number in enumerate(numbers):
            if (
                not math.isfinite(number)
                or number != int(number)
                or number < -128
                or number > 127
            ):
                raise InvalidInputError(
                    message=f"embedding element {index} is not a valid int8 value",
                    location=location,
                )
        return np.asarray([int(number) for number in numbers], dtype=np.int8), "int8"

    return np.asarray(numbers, dtype=NUMPY_DTYPE[declared_dtype]), declared_dtype


def decode_embedding_b64(
    b64_text: str, *, declared_dtype: str | None, location: str
) -> tuple[np.ndarray, str]:
    """Decode a base64 little-endian embedding blob requiring a declared dtype."""

    if not isinstance(b64_text, str):
        raise InvalidInputError(
            message="embedding_b64 value is not a string", location=location
        )
    if declared_dtype is None:
        raise InvalidInputError(
            message="embedding_b64 requires a declared dtype", location=location
        )
    _check_dtype(declared_dtype, location)
    try:
        raw = base64.b64decode(b64_text, validate=True)
    except (binascii.Error, ValueError):
        raise InvalidInputError(
            message="invalid base64 embedding", location=location
        ) from None
    try:
        vector = np.frombuffer(raw, dtype=f"<{NUMPY_LE_CODE[declared_dtype]}")
    except ValueError:
        raise InvalidInputError(
            message="base64 embedding byte length does not match its dtype",
            location=location,
        ) from None
    return vector, declared_dtype
