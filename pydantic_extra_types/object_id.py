from __future__ import annotations

from typing import Annotated, Any, ClassVar, Union, List, TypeVar, Generic, TypeVarTuple, get_args

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

import warnings

try:
    from bson import ObjectId
except ModuleNotFoundError:  # pragma: no cover
    raise RuntimeError(
        "'ObjectId' requires 'bson' to be installed. You can install it with 'pip install bson'"
    )


class _ObjectId:
    """
    Base class for numpy scalar types
    """

    @classmethod
    def __get_pydantic_core_schema__(
            cls, source: type[Any],
            handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:

        from_string_schema = core_schema.chain_schema([
            core_schema.union_schema([
                core_schema.is_instance_schema(ObjectId),
                core_schema.str_schema()
            ]),
            core_schema.no_info_plain_validator_function(ObjectId),
        ])

        json_schema = from_string_schema

        serializer = core_schema.plain_serializer_function_ser_schema(
            lambda instance: str(instance), when_used="json")

        return core_schema.json_or_python_schema(
            python_schema=from_string_schema,
            json_schema=json_schema,
            serialization=serializer)


ObjectIdType = Annotated[Union[str, ObjectId], _ObjectId]

__all__ = ["ObjectIdType"]
