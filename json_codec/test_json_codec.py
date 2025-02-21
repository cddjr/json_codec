from datetime import date, datetime, time, timezone
from enum import Enum
import json
from dataclasses import dataclass
from decimal import Decimal
from this import d
from typing import Dict, List, NewType, Optional, Union, Tuple, Set

import pytest

from json_codec import (
    LocatedValidationErrorCollection,
    get_class_or_type_name,
    decode,
    encode,
    mapping,
)


class TestJsonDeserializerCodec:
    def test_decode_primitives(self) -> None:
        assert decode(json.loads("true"), bool) is True
        assert decode(json.loads("false"), bool) is False
        assert decode(json.loads("null"), Optional[bool]) is None
        assert decode(json.loads("1"), int) == 1
        assert decode(json.loads("1"), Decimal) == Decimal("1")
        assert decode(json.loads('"1.1"'), Decimal) == Decimal("1.1")
        assert decode(json.loads('"1.1"'), float) == 1.1
        assert decode(json.loads('"1.1"'), str) == "1.1"
        
        assert decode(json.loads('[1,1]'), List[int]) == [1, 1]


    def test_frozen_dataclass(self) -> None:
        @dataclass(frozen=True)
        class User:
            name: str
            age: int

        assert decode({"name": "John", "age": 30}, User) == User(name="John", age=30)

    def test_basic_dataclass(self) -> None:
        @dataclass
        class Dummy:
            text_list: List[str]
            text_dict: Dict[str, Decimal]
            optional_text: Optional[str]

        dummy_json_text = """
        {
            "text_list": ["a", "b", "c"],
            "text_dict": {
                "a": 1.0,
                "b": 2,
                "c": "3.3",
                "d": 2.2
            },
            "optional_text": "hello"
        }
        """

        dummy_json = json.loads(dummy_json_text)

        parsed = decode(dummy_json, Dummy)

        assert parsed.text_list == ["a", "b", "c"]
        assert parsed.text_dict["a"] == Decimal("1.0")
        assert parsed.text_dict["b"] == Decimal("2.0")
        assert parsed.text_dict["c"] == Decimal("3.3")
        assert parsed.text_dict["d"].quantize(Decimal("1.0")) == Decimal("2.2")
        assert parsed.optional_text == "hello"

    def test_nested_dataclass(self) -> None:
        @dataclass
        class NestedDummy:
            text: str
            number: Decimal

            boolean: bool

        @dataclass
        class Dummy:
            text_list: List[str]
            text_dict: Dict[str, Decimal]
            nested: NestedDummy

        dummy_json_text = """
        {

            "text_list": ["a", "b", "c"],
            "text_dict": {
                "a": 1.0,
                "b": 2,
                "c": "3.3",
                "d": 2.2
            },
            "nested": {
                "text": "hello",
                "number": 1.1,
                "boolean": true
            }
        }
        """

        dummy_json = json.loads(dummy_json_text)

        parsed = decode(dummy_json, Dummy)

        assert parsed.text_list == ["a", "b", "c"]
        assert parsed.text_dict["a"] == Decimal("1.0")
        assert parsed.text_dict["b"] == Decimal("2.0")
        assert parsed.text_dict["c"] == Decimal("3.3")
        assert parsed.text_dict["d"].quantize(Decimal("1.0")) == Decimal("2.2")
        assert parsed.nested.text == "hello"
        assert parsed.nested.number.quantize(Decimal("1.0")) == Decimal("1.1")
        assert parsed.nested.boolean is True

    def test_raise_when_type_not_mapped(self) -> None:

        with pytest.raises(ValueError):

            class NonMappedDummy:
                pass

            @dataclass
            class Dummy:
                text: str
                non_mapped: NonMappedDummy

            dummy_json_text = """
            {
                "text": "hello",
                "non_mapped": {}
            }
            """

            dummy_json = json.loads(dummy_json_text)

            decode(dummy_json, Dummy)

    def test_raise_when_missing_field(self) -> None:

        with pytest.raises(LocatedValidationErrorCollection):

            @dataclass
            class Dummy:
                text: int

            dummy_json_text = """
            {
            }
            """

            dummy_json = json.loads(dummy_json_text)

            decode(dummy_json, Dummy)

    def test_get_class_or_type_name(self) -> None:
        @dataclass
        class Dummy:
            text: str

        class NormalClass:
            pass

        assert get_class_or_type_name(Dummy) == "Dummy"
        assert get_class_or_type_name(List) == "typing.List"
        assert get_class_or_type_name(
            NormalClass
        ) == "{cls_name}.{method_name}.<locals>.NormalClass".format(
            cls_name=TestJsonDeserializerCodec.__name__,
            method_name=TestJsonDeserializerCodec.test_get_class_or_type_name.__name__,
        )

    def test_type_not_in_union(self) -> None:

        with pytest.raises(LocatedValidationErrorCollection):

            @dataclass
            class Dummy:
                text: Union[List[str], Dict[str, str]]

            dummy_json_text = """
            {
                "text": 1
            }

            """

            dummy_json = json.loads(dummy_json_text)

            decode(dummy_json, Dummy)

    def test_dict_with_wrong_type(self) -> None:

        with pytest.raises(LocatedValidationErrorCollection) as e:

            @dataclass
            class Dummy:
                text: Dict[int, int]

            dummy_json_text = """
            {
                "text": {
                    "a": "1"
                }
            }

            """

            dummy_json = json.loads(dummy_json_text)

            a = decode(dummy_json, Dummy)

        assert e.value is not None

    def test_enum(self) -> None:
        class MyEnum(Enum):
            A = "A"
            B = "B"

        @dataclass
        class Dummy:
            my_enum: MyEnum

        dummy_json_text = """
        {
            "my_enum": "A"
        }

        """

        dummy_json = json.loads(dummy_json_text)

        a = decode(dummy_json, Dummy)

        assert a.my_enum == MyEnum.A

    def test_enum_with_wrong_value(self) -> None:

        with pytest.raises(LocatedValidationErrorCollection):

            class MyEnum(Enum):
                A = "A"
                B = "B"

            @dataclass
            class Dummy:
                my_enum: MyEnum

            dummy_json_text = """
            {
                "my_enum": "C"
            }

            """

            dummy_json = json.loads(dummy_json_text)

            a = decode(dummy_json, Dummy)

    def test_date(self) -> None:
        @dataclass
        class Dummy:
            date_time: datetime
            date_: date
            time_: time

        dummy_json_text = """
        {
            "date_": "2020-01-01",
            "date_time": "2020-01-01T00:00:00+00:00",
            "time_": "00:00:00"
        }

        """

        dummy_json = json.loads(dummy_json_text)

        a = decode(dummy_json, Dummy)

        assert a.date_ == date(2020, 1, 1)
        assert a.date_time == datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert a.time_ == time(0, 0, 0)

    def test_date_with_wrong_value(self) -> None:
            
            with pytest.raises(LocatedValidationErrorCollection):
    
                @dataclass
                class Dummy:
                    date_time: datetime
    
                dummy_json_text = """
                {
                    "date_time": "2020-01-01T00:00:00"
                }
    
                """
    
                dummy_json = json.loads(dummy_json_text)
    
                a = decode(dummy_json, Dummy)
    def test_primitive_class_inheritance(self) -> None:
        class MyInt(int):
            pass

        @dataclass
        class Dummy:
            my_int: MyInt

        dummy_json_text = """
        {
            "my_int": 1
        }

        """

        dummy_json = json.loads(dummy_json_text)

        a = decode(dummy_json, Dummy)

        assert a.my_int == MyInt(1)

    def test_primitive_class_inheritance_class_match(self) -> None:
        class MyInt(int):
            pass

        @dataclass
        class Dummy:
            my_int: MyInt

        dummy_json_text = """
        {
            "my_int": "1"
        }

        """

        dummy_json = json.loads(dummy_json_text)
        
        parsed = decode(dummy_json, Dummy)

        assert parsed.my_int == MyInt(1)
        assert isinstance(parsed.my_int, MyInt)
        

        
    def test_decode_newtype(self):

        UserId = NewType("UserId", int)

        assert decode(json.loads("1"), UserId) == UserId(1)
        assert isinstance(decode(json.loads("1"), UserId), int)

    def test_tuple(self):
        @dataclass
        class Dummy:
            t: Tuple[int, str, bool]

        dummy_json_text = '{"t": [1, "2", true]}'

        foo = Dummy((1, "2", True))
        assert json.dumps(encode(foo)) == dummy_json_text

        bar = decode(json.loads(dummy_json_text), Dummy)
        assert foo == bar

    def test_set(self):
        @dataclass
        class Dummy:
            s: Set[int]

        dummy_json_text = '{"s": [1, 2, 3]}'

        foo = Dummy({1, 2, 3})
        assert json.dumps(encode(foo)) == dummy_json_text

        bar = decode(json.loads(dummy_json_text), Dummy)
        assert foo == bar

    def test_rename_fields(self):
        @dataclass
        @mapping(x="a", y="b")
        class C:
            x: int
            y: int

        @dataclass()
        @mapping(value="val", __op="op", id="guid", c="sub")
        class Dummy:
            id: str
            __op: int
            value: int
            c:C
            succ: bool = True

        dummy_json_text = '{"guid": "12", "op": 3, "val": 4, "sub": {"a": 12, "b": 13}, "succ": true}'
        foo = Dummy("12", 3, 4, C(12, 13))
        assert json.dumps(encode(foo)) == dummy_json_text

        bar = decode(json.loads(dummy_json_text), Dummy)
        assert foo == bar

    def test_skip_fields(self):
        @dataclass
        @mapping(x="a", y=None)
        class C:
            x: int
            y: int = 0

        @dataclass()
        @mapping(value="val", succ = None, __op="op", id="guid", c="sub")
        class Dummy:
            id: str
            __op: int
            value: int
            c:C
            succ: bool = False

        dummy_json_text = '{"guid": "12", "op": 3, "val": 4, "sub": {"a": 2}}'
        foo = Dummy("12", 3,  4, C(2, 7))
        assert json.dumps(encode(foo)) == dummy_json_text

        bar = decode(json.loads(dummy_json_text), Dummy)
        bar.c.y = 7
        assert foo == bar
        
    def test_renamed_fields_missing(self):
        with pytest.raises(LocatedValidationErrorCollection, match="Missing required field: y"):
            @dataclass
            @mapping(x="a", y="b")
            class C:
                x: int
                y: int

            dummy_json_text = '{"a": 12, "y": 13}'
            decode(json.loads(dummy_json_text), C)

    def test_encoding_skips_required_fields(self):
        with pytest.raises(Exception, match="Required field cannot be skipped: y"):
            @dataclass
            @mapping(x="a", y=None)
            class C:
                x: int
                y: int
            encode(C(2, 3))
            
    def test_decoding_skips_required_fields(self):
        with pytest.raises(Exception, match="Required field cannot be skipped: y"):
            @dataclass
            @mapping(x="a", y=None)
            class C:
                x: int
                y: int
            decode({"a": 2, "y": 3}, C)

    def test_dataclass_with_default(self):
        @dataclass()
        class Dummy:
            a: str
            b: int = 4

        dummy_json_text = '{"a": "123", "b": 4}'
        foo = Dummy("123")
        assert json.dumps(encode(foo)) == dummy_json_text

        bar = decode(json.loads(dummy_json_text), Dummy)
        assert foo == bar

    def test_dataclass_with_none(self):
        @dataclass()
        class Dummy:
            a: str
            b: Optional[int] = None

        dummy_json_text = '{"a": "123"}'
        foo = Dummy("123")

        bar = decode(json.loads(dummy_json_text), Dummy)
        assert foo == bar

    def test_bool_decoder(self):
        @dataclass
        class Dummy:
            a: bool
            b: bool
            c: bool
            d: bool
            e: bool
            f: bool
        
        parsed = decode({"a": False, "b":"TrUe", 
                         "c":"fAlSe", "d":123, 
                         "e":0, "f":" false"}, Dummy)
        assert parsed == Dummy(a=False, b=True, c=False, d=True, e=False, f=True)

    def test_raise_when_decoding_bool(self):
        @dataclass
        class Dummy:
            a: bool
        
        with pytest.raises(LocatedValidationErrorCollection, 
                           match="Expected type bool"):
            decode({"a": []}, Dummy)
        