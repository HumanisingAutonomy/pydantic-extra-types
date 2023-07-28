from typing import Any, List, Literal as L

import pytest
import pydantic
from pydantic import BaseModel, ValidationError

import pydantic_extra_types.numpy as pnp
import numpy as np
import json


class IntModel(BaseModel):
    array: pnp.NumPyInt32


class FloatModel(BaseModel):
    array: pnp.NumPyFloat32

class UintModel(BaseModel):
    array: pnp.NumPyUInt8

INT_COMBOS = [[1, 2, 3, 4], [[1, 2], [3, 4]], [[[1, 2], [3, 4]]]]
FLOAT_COMBOS = [[0.1, 0, 2, 0.3, 0.4], [[0.1, 0.2], [0.3, 0.4]],
                [[[0.1, 0.2], [0.3, 0.4]]]]


@pytest.mark.parametrize('arr', INT_COMBOS)
def test_int_array(arr: List[Any]) -> None:
    m = IntModel(array=arr)

    assert isinstance(m.array, np.ndarray)
    assert m.array.dtype == np.int32


@pytest.mark.parametrize('arr', FLOAT_COMBOS)
def test_float_array(arr: List[Any]) -> None:
    m = FloatModel(array=arr)
    assert isinstance(m.array, np.ndarray)
    assert m.array.dtype == np.float32


@pytest.mark.parametrize('arr', INT_COMBOS)
def test_parse_dict(arr: List[Any]) -> None:
    dict_value = {"array": arr}

    m = IntModel.model_validate(dict_value)

    assert isinstance(m.array, np.ndarray)
    assert m.array.dtype == np.int32


def test_parse_json() -> None:
    json_value = "{\"array\": [1, 2, 3, 4]}"

    m = IntModel.model_validate_json(json_value)

    assert isinstance(m.array, np.ndarray)
    assert m.array.dtype == np.int32


@pytest.mark.parametrize('arr', INT_COMBOS)
def test_dump_json(arr: List[Any]) -> None:
    m = IntModel(array=arr)
    val = m.model_dump(mode="json")
    assert val == {"array": arr}


@pytest.mark.parametrize('arr', INT_COMBOS)
def test_dump_dict(arr: List[Any]) -> None:
    m = IntModel(array=arr)
    val = m.model_dump()

    assert "array" in val
    assert isinstance(val["array"], np.ndarray)
    assert np.all(val["array"] == np.array(arr))


def test_int_to_float():
    # only warn the first time
    with pytest.warns(UserWarning, match="Implicit conversion from int64 to"):
        m = FloatModel(array=INT_COMBOS[0])
    assert isinstance(m.array, np.ndarray)
    assert m.array.dtype == np.float32
    
    for comb in INT_COMBOS:
        m = FloatModel(array=comb)
        assert isinstance(m.array, np.ndarray)
        assert m.array.dtype == np.float32


def test_float_to_int():
    
    # only warn the first time
    with pytest.warns(UserWarning, match="Implicit conversion from float64 to"):
        m = IntModel(array=FLOAT_COMBOS[0])
    assert isinstance(m.array, np.ndarray) and m.array.dtype == np.int32
    
    for comb in FLOAT_COMBOS:
        m = IntModel(array=comb)
        assert isinstance(m.array, np.ndarray)
        assert m.array.dtype == np.int32

def test_int_to_uint():

    with pytest.warns(UserWarning, match="Implicit conversion from int64 to"):
        m = UintModel(array=[-1, 0, 1, 2])
    
    assert isinstance(m.array, np.ndarray)
    assert m.array.dtype == np.uint8

def test_all():
    int_vals = [-1, -2, -3, -4]
    uint_vals = [1, 2, 3, 4]
    float_vals = [0.1, 0.2, 0.3, 0.4]
    complex_vals = list(map(complex, [1, 2, 3, 4]))

    validation = {
        pnp.NumPyInt8: int_vals,
        pnp.NumPyInt16: int_vals,
        pnp.NumPyInt32: int_vals,
        pnp.NumPyInt64: int_vals,
        pnp.NumPyUInt8: uint_vals,
        pnp.NumPyUInt16: uint_vals,
        pnp.NumPyUInt32: uint_vals,
        pnp.NumPyUInt64: uint_vals,
        pnp.NumPyFloat16: float_vals,
        pnp.NumPyFloat32: float_vals,
        pnp.NumPyFloat64: float_vals,
        pnp.NumPyComplex64: complex_vals,
        pnp.NumPyComplex128: complex_vals,
    }

    for t, v in validation.items():
        pydantic.TypeAdapter(t).validate_python(v)