from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_extra_types.phone_numbers import PhoneNumber


class Something(BaseModel):
    phone_number: PhoneNumber


# Note: the 555 area code will result in an invalid phone number
def test_valid_phone_number() -> None:
    Something(phone_number='+1 901 555 1212')


def test_when_extension_provided() -> None:
    Something(phone_number='+1 901 555 1212 ext 12533')


@pytest.mark.parametrize('invalid_number', ['', '123', 12, None, object(), '55 121'])
def test_invalid_phone_number(invalid_number: Any) -> None:
    with pytest.raises(ValidationError):
        Something(phone_number=invalid_number)


def test_formats_phone_number() -> None:
    result = Something(phone_number='+1 901 555 1212 ext 12533')
    assert result.phone_number == 'tel:+1-901-555-1212;ext=12533'


def test_supported_regions() -> None:
    assert 'US' in PhoneNumber.supported_regions
    assert 'GB' in PhoneNumber.supported_regions


def test_supported_formats() -> None:
    assert 'E164' in PhoneNumber.supported_formats
    assert 'RFC3966' in PhoneNumber.supported_formats
    assert '__dict__' not in PhoneNumber.supported_formats
    assert 'to_string' not in PhoneNumber.supported_formats


def test_parse_error() -> None:
    with pytest.raises(ValidationError, match='value is not a valid phone number'):
        Something(phone_number='555 1212')


def test_parsed_but_not_a_valid_number() -> None:
    with pytest.raises(ValidationError, match='value is not a valid phone number'):
        Something(phone_number='+1 555-1212')
