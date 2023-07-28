from typing import Any, List

import pytest
from pydantic import BaseModel, ValidationError

import pydantic_extra_types.numpy as pnp
import numpy as np
import json


class IntModel(BaseModel):
    array: pnp.NumPyInt32


class FloatModel(BaseModel):
    array: pnp.NumPyFloat32


INT_COMBOS = [[1, 2, 3, 4], [[1, 2], [3, 4]], [[[1, 2], [3, 4]]]]
FLOAT_COMBOS = [[0.1, 0, 2, 0.3, 0.4], [[0.1, 0.2], [0.3, 0.4]],
                [[[0.1, 0.2], [0.3, 0.4]]]]


@pytest.mark.parametrize('arr', INT_COMBOS)
def test_int_array(arr: List[Any]) -> None:
    m = IntModel(array=arr)

    assert (isinstance(m.array, np.ndarray) and m.array.dtype == np.int32)


@pytest.mark.parametrize('arr', FLOAT_COMBOS)
def test_float_array(arr: List[Any]) -> None:
    m = FloatModel(array=arr)
    assert (isinstance(m.array, np.ndarray) and m.array.dtype == np.float32)


@pytest.mark.parametrize('arr', INT_COMBOS)
def test_parse_dict(arr: List[Any]) -> None:
    dict_value = {"array": arr}

    m = IntModel.model_validate(dict_value)

    assert (isinstance(m.array, np.ndarray) and m.array.dtype == np.int32)


def test_parse_json() -> None:
    json_value = "{\"array\": [1, 2, 3, 4]}"

    m = IntModel.model_validate_json(json_value)

    assert (isinstance(m.array, np.ndarray) and m.array.dtype == np.int32)


@pytest.mark.parametrize('arr', INT_COMBOS)
def test_dump_json(arr: List[Any]) -> None:
    m = IntModel(array=arr)
    val = m.model_dump(mode="json")
    assert val == {"array": arr}


@pytest.mark.parametrize('arr', INT_COMBOS)
def test_dump_dict(arr: List[Any]) -> None:
    m = IntModel(array=arr)
    val = m.model_dump()

    assert ("array" in val and isinstance(val["array"], np.ndarray)
            and np.all(val["array"] == np.array(arr)))
