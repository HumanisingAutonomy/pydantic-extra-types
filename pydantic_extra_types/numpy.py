from __future__ import annotations

from typing import Annotated, Any, ClassVar, Union, List

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

try:
    import numpy as np
    from numpy import typing as npt
except ModuleNotFoundError:  # pragma: no cover
    raise RuntimeError(
        "'NumPy' requires 'numpy' to be installed. You can install it with 'pip install numpy'"
    )


class _NumPy:
    """
    Base class for numpy scalar types
    """

    np_type: ClassVar[npt.DTypeLike]

    @classmethod
    def __get_pydantic_core_schema__(
            cls, source: type[Any],
            handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        # max_value = np.iinfo(cls.np_type).max
        # min_value = np.iinfo(cls.np_type).min

        def _validate_from_list(value: List[Any]) -> npt.NDArray[Any]:
            return np.array(value, dtype=cls.np_type)

        from_list_schema = core_schema.chain_schema([
            core_schema.list_schema(),
            core_schema.no_info_plain_validator_function(_validate_from_list),
        ])

        json_schema = from_list_schema

        serializer = core_schema.plain_serializer_function_ser_schema(
            lambda instance: instance.tolist(), when_used="json")

        return core_schema.json_or_python_schema(
            python_schema=from_list_schema,
            json_schema=json_schema,
            serialization=serializer)

    @classmethod
    def __get_pydantic_json_schema__(
            cls, schema: core_schema.CoreSchema,
            handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        json_schema = handler(schema)
        return json_schema


class _NumPyInt8(_NumPy):
    np_type = np.int8


class _NumPyInt16(_NumPy):
    np_type = np.int16


class _NumPyInt32(_NumPy):
    np_type = np.int32


class _NumPyInt64(_NumPy):
    np_type = np.int64


class _NumPyUInt8(_NumPy):
    np_type = np.uint8


class _NumPyUInt16(_NumPy):
    np_type = np.uint16


class _NumPyUInt32(_NumPy):
    np_type = np.uint32


class _NumPyUInt64(_NumPy):
    np_type = np.uint64


class _NumPyFloat32(_NumPy):
    np_type = np.float32


class _NumPyFloat64(_NumPy):
    np_type = np.float64


NumPyInt8 = Annotated[Union[npt.NDArray, npt.ArrayLike], _NumPyInt8]
NumPyInt16 = Annotated[Union[npt.NDArray, npt.ArrayLike], _NumPyInt16]
NumPyInt32 = Annotated[Union[npt.NDArray, npt.ArrayLike], _NumPyInt32]
NumPyInt64 = Annotated[Union[npt.NDArray, npt.ArrayLike], _NumPyInt64]
NumPyUInt8 = Annotated[Union[npt.NDArray, npt.ArrayLike], _NumPyUInt8]
NumPyUInt16 = Annotated[Union[npt.NDArray, npt.ArrayLike], _NumPyUInt16]
NumPyUInt32 = Annotated[Union[npt.NDArray, npt.ArrayLike], _NumPyUInt32]
NumPyUInt64 = Annotated[Union[npt.NDArray, npt.ArrayLike], _NumPyUInt64]

NumPyFloat32 = Annotated[Union[npt.NDArray, List[float]], _NumPyFloat32]
NumPyFloat64 = Annotated[Union[npt.NDArray, List[float]], _NumPyFloat64]

__all__ = [
    "NumPyInt8",
    "NumPyInt16",
    "NumPyInt32",
    "NumPyInt64",
    "NumPyUInt8",
    "NumPyUInt16",
    "NumPyUInt32",
    "NumPyUInt64",
    "NumPyFloat32",
    "NumPyFloat64",
]