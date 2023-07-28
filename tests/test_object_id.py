from typing import Any

import pytest
import pydantic
from pydantic import BaseModel, ValidationError

from pydantic_extra_types.object_id import ObjectIdType
from bson import ObjectId


class Model(BaseModel):
    id: ObjectIdType

TEST_OID = "64c3d72734e15b09e36dd9a4"

def test_from_str():
    m = Model(id=TEST_OID)

    assert isinstance(m.id, ObjectId)
    assert str(m.id) == TEST_OID

def test_from_object_id():
    in_id = ObjectId(TEST_OID)
    m = Model(id=in_id)

    assert isinstance(m.id, ObjectId)
    assert str(m.id) == TEST_OID

def test_to_dict():
    m = Model(id=TEST_OID)
    
    model_dict = m.model_dump()

    assert isinstance(model_dict["id"], ObjectId)
    assert str(model_dict["id"]) == TEST_OID

def test_to_json():
    m = Model(id=TEST_OID)
    
    model_dict = m.model_dump(mode="json")

    assert isinstance(model_dict["id"], str)
    assert model_dict["id"] == TEST_OID