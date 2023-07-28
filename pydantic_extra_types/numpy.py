from __future__ import annotations

from typing import Annotated, Any, ClassVar, Union, List, TypeVar, Generic, TypeVarTuple, get_args
import collections.abc

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema, PydanticCustomError

import warnings

try:
    import numpy as np
    from numpy import typing as npt
except ModuleNotFoundError:  # pragma: no cover
    raise RuntimeError(
        "'NumPy' requires 'numpy' to be installed. You can install it with 'pip install numpy'"
    )

DType = TypeVar('DType', bound=np.generic)


class _NumPy(Generic[DType]):
    """
    Base class for numpy scalar types
    """

    should_warn = True

    @classmethod
    def __get_pydantic_core_schema__(
            cls, source: type[Any],
            handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        
        dtype: npt.DTypeLike = get_args(
            cls.__orig_bases__[0])[0]  # type: ignore

        def _validate_from_list(value: List[Any]) -> npt.NDArray[Any]:
            if not isinstance(value, (collections.abc.Sequence, np.ndarray)):
                raise PydanticCustomError("numpy_array", f"Expected sequence")
            arr = np.array(value)
            int_to_float = (np.issubdtype(arr.dtype, np.integer)
                            and np.issubdtype(dtype, np.floating))
            float_to_int = (np.issubdtype(arr.dtype, np.floating)
                            and np.issubdtype(dtype, np.integer))
            int_to_uint = (np.any(arr < 0)
                           and np.issubdtype(dtype, np.unsignedinteger))
            complex_to_not = (np.issubdtype(arr.dtype, np.complexfloating)
                              and not np.issubdtype(dtype, np.complexfloating))
            should_warn = (int_to_float or float_to_int or int_to_uint
                           or complex_to_not)
            if should_warn and cls.should_warn:
                print(cls)
                cls.should_warn = False
                warnings.warn(
                    f'Implicit conversion from {arr.dtype} to {np.dtype(dtype).name}'
                )
            return arr.astype(dtype)

        from_list_schema = core_schema.no_info_plain_validator_function(_validate_from_list)

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


####################################
# Integers
####################################
class _NumPyInt8(_NumPy[np.int8]):
    ...


class _NumPyInt16(_NumPy[np.int16]):
    ...


class _NumPyInt32(_NumPy[np.int32]):
    ...


class _NumPyInt64(_NumPy[np.int64]):
    ...


NumPyInt8 = Annotated[npt.NDArray[np.int8], _NumPyInt8]
NumPyInt16 = Annotated[npt.NDArray[np.int16], _NumPyInt16]
NumPyInt32 = Annotated[npt.NDArray[np.int32], _NumPyInt32]
NumPyInt64 = Annotated[npt.NDArray[np.int64], _NumPyInt64]


####################################
# Unsigned Integers
####################################
class _NumPyUInt8(_NumPy[np.uint8]):
    ...


class _NumPyUInt16(_NumPy[np.uint16]):
    ...


class _NumPyUInt32(_NumPy[np.uint32]):
    ...


class _NumPyUInt64(_NumPy[np.uint64]):
    ...


NumPyUInt8 = Annotated[npt.NDArray[np.uint8], _NumPyUInt8]
NumPyUInt16 = Annotated[npt.NDArray[np.uint16], _NumPyUInt16]
NumPyUInt32 = Annotated[npt.NDArray[np.uint32], _NumPyUInt32]
NumPyUInt64 = Annotated[npt.NDArray[np.uint64], _NumPyUInt64]


####################################
# Floats
####################################
class _NumPyFloat16(_NumPy[np.float16]):
    ...


class _NumPyFloat32(_NumPy[np.float32]):
    ...


class _NumPyFloat64(_NumPy[np.float64]):
    ...


NumPyFloat16 = Annotated[npt.NDArray[np.float16], _NumPyFloat16]
NumPyFloat32 = Annotated[npt.NDArray[np.float32], _NumPyFloat32]
NumPyFloat64 = Annotated[npt.NDArray[np.float64], _NumPyFloat64]

####################################
# Complex
####################################


class _NumPyComplex64(_NumPy[np.complex64]):
    ...


class _NumPyComplex128(_NumPy[np.complex128]):
    ...


NumPyComplex64 = Annotated[npt.NDArray[np.complex64], _NumPyComplex64]
NumPyComplex128 = Annotated[npt.NDArray[np.float128], _NumPyComplex128]

__all__ = [
    "NumPyInt8", "NumPyInt16", "NumPyInt32", "NumPyInt64", "NumPyUInt8",
    "NumPyUInt16", "NumPyUInt32", "NumPyUInt64", "NumPyFloat16",
    "NumPyFloat32", "NumPyFloat64", "NumPyComplex64", "NumPyComplex128"
]