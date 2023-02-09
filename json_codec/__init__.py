from dataclasses import MISSING, fields, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, TypeVar, Union, cast, Optional
from uuid import UUID
from copy import deepcopy

from typing_extensions import Type
from .codecs.date_codec import DateTypeDecoder, serialize_date
from .codecs.datetime_codec import (
    DateTimeTypeDecoder,
    serialize_datetime,
)

from .codecs.bool_codec import BoolTypeDecoder
from .codecs.dict_codec import DictTypeDecoder
from .codecs.list_codec import ListTypeDecoder as ListTypeParser
from .codecs.primitive_codec import (
    PrimitiveTypeDecoder,
)
from .codecs.set_codec import SetTypeDecoder as SetTypeParser
from .codecs.time_codec import (
    TimeTypeDecoder as TimeTypeParser,
    serialize_time,
)
from .codecs.tuple_codec import (
    TupleTypeDecoder as TupleTypeParser,
)
from .codecs.union_codec import (
    UnionTypeDecoder as UnionTypeParser,
)
from .types import (
    AssumeDataclass,
    AssumeGeneric,
    AssumeNewType,
    ParseProcessResult,
    TypeDecoder,
    ValidationError,
    ValidationErrorBase,
)

__all__ = [
    "decode",
    "encode",
    "mapping",

    "LocatedValidationError",
    "LocatedValidationErrorCollection",
]

T = TypeVar("T")

typers_parsers: Dict[Any, TypeDecoder[Any]] = {
    Decimal: PrimitiveTypeDecoder(Decimal),
    str: PrimitiveTypeDecoder(str),
    int: PrimitiveTypeDecoder(int),
    float: PrimitiveTypeDecoder(float),
    bool: BoolTypeDecoder(),
    dict: DictTypeDecoder(),
    list: ListTypeParser(),
    tuple: TupleTypeParser(),
    set: SetTypeParser(),
    UUID: PrimitiveTypeDecoder(UUID),
    Union: UnionTypeParser(),
    Any: PrimitiveTypeDecoder(lambda x: x),
    date: DateTypeDecoder(),
    datetime: DateTimeTypeDecoder(),
    time: TimeTypeParser(),
    type(None): PrimitiveTypeDecoder(lambda x: None),
}


def is_generic(type_: Type[Any]) -> bool:
    return (
        hasattr(type_, "__origin__")
        and cast(AssumeGeneric, type_).__origin__ is not None
    )


class LocatedValidationError(Exception):
    def __init__(self, message: str, json_path: str) -> None:
        super().__init__(message)
        self.json_path = json_path


class LocatedValidationErrorCollection(Exception):
    def __init__(self, errors: List[LocatedValidationError]) -> None:
        super().__init__("Located validation errors: %s" % errors)
        self.errors = errors

    def __str__(self) -> str:
        return "\n".join(["{}: {}".format(e.json_path, str(e)) for e in self.errors])


def __get_recursive_mapped_type(cls_type: Type[Any]) -> Type[Any]:
    if not hasattr(cls_type, "__bases__") or len(cls_type.__bases__) == 0:
        return cls_type

    while cls_type not in typers_parsers:
        cls_type = cls_type.__bases__[0]
        if cls_type not in typers_parsers:
            return __get_recursive_mapped_type(cls_type)
    return cls_type


def is_new_type(type_: Type[Any]) -> bool:
    return hasattr(type_, "__supertype__")


def get_new_type_supertype(type_: Type[Any]) -> Type[Any]:
    return cast(AssumeNewType, type_).__supertype__


_MAPPING_FIELD = "__json_codec_mapping__"

def mapping(**kwargs: Optional[str]):
    """
    Rename or skip certain fields

    Example usage::

      @dataclass
      @mapping(x='a', y=None)
      class C:
          x: int
          y: int = 0
      assert encode(C(1, 2)) == {'a': 1}
    """
    def wrap(cls):
        mapping = getattr(cls, _MAPPING_FIELD, None)
        if mapping is None:
            mapping = {}
            setattr(cls, _MAPPING_FIELD, mapping)
        for field_name, field_json_name in kwargs.items():
            if field_name.startswith("__"):
                field_name = f"_{cls.__name__}{field_name}"
            mapping[field_name] = field_json_name
        return cls

    return wrap

def __parse_value(
    value: Any,
    type_: Type[T],
    json_path: str = "$",
    located_errors: List[LocatedValidationError] = [],
    skip_raise: bool = False,
) -> ParseProcessResult[T]:
    real_type = type_
    target_type = type_
    type_args: Tuple[Type[Any], ...] = ()
    if is_generic(type_):
        real_type = cast(AssumeGeneric, type_).__origin__
        target_type = real_type
        type_args = cast(AssumeGeneric, type_).__args__
    elif is_new_type(type_):
        target_type = get_new_type_supertype(type_)
    elif not is_dataclass(type_) and not issubclass(real_type, Enum):
        target_type = __get_recursive_mapped_type(type_)

    if target_type in typers_parsers:
        parser = typers_parsers[target_type]
        parser_generator = parser.parse(value, *type_args)
        try:
            parsed_yield = parser_generator.send(cast(Any, None))
            while True:
                parsed_value = __parse_value(
                    parsed_yield.value,
                    parsed_yield.type_,
                    "{}{}".format(json_path, parsed_yield.json_path),
                    located_errors,
                    parsed_yield.skip_raise,
                )
                parsed_yield = parser_generator.send(parsed_value)
        except StopIteration as e:

            final = e.value
            if not isinstance(final, ParseProcessResult):
                raise ValueError(f"Parser {parser} did not return a ParseProcessResult")
            if isinstance(final.result, Exception) and not skip_raise:
                located_errors.append(
                    LocatedValidationError(
                        message=str(final.result),
                        json_path=json_path,
                    )
                )

            if target_type != real_type:
                final = ParseProcessResult(
                    result=cast(Type[Any], real_type)(final.result),
                )

            return cast(ParseProcessResult[T], final)

    elif is_dataclass(real_type):
        try:
            return ParseProcessResult(
                __parse_dataclass(
                    value,
                    real_type,
                    json_path,
                    located_errors,
                )
            )
        except AssertionError as e:

            error = ValidationError(
                str(e),
            )
            if not skip_raise:
                located_errors.append(
                    LocatedValidationError(
                        message=str(error),
                        json_path=json_path,
                    )
                )
            return ParseProcessResult(error)
    elif issubclass(real_type, Enum):
        try:
            value = real_type(value)
            return ParseProcessResult(value)
        except ValueError as e:
            error = ValidationError(
                "Invalid enum value for {}: {} | valid types: {}".format(
                    real_type,
                    value,
                    ", ".join(k for k, v in real_type.__members__.items()),
                )
            )
            if not skip_raise:
                located_errors.append(
                    LocatedValidationError(
                        message=str(error),
                        json_path=json_path,
                    )
                )
            return ParseProcessResult(error)

    else:
        raise ValueError(f"Unsupported type: {type_}")

def __get_default(field):
    if field.default is not MISSING:
        return field.default
    elif field.default_factory is not MISSING:  # type: ignore
        return field.default_factory()  # type: ignore
    else:
        return MISSING

def __parse_dataclass(
    value: Any,
    type_: Type[T],
    json_path: str = "$",
    located_errors: List[LocatedValidationError] = [],
) -> T:
    assert isinstance(value, dict), "Value must be a dict"

    assert is_dataclass(type_), "Type must be a dataclass"

    fields = cast(AssumeDataclass, type_).__dataclass_fields__

    kwargs: Dict[str, Any] = {}

    mapping = getattr(type_, _MAPPING_FIELD, {})

    for field_name, field in fields.items():

        if field_name in mapping:
            field_json_name = mapping[field_name]
            if field_json_name is None:
                default = __get_default(field)
                if default is MISSING:
                    default = None
                    located_errors.append(
                        LocatedValidationError(
                            message=f"Required field cannot be skipped: {field_name}",
                            json_path=json_path,
                        )
                    )
                kwargs[field_name] = default
                continue
        else:
            field_json_name = field_name

        field_json_path = f"{json_path}.{field_json_name}"

        if field_json_name not in value:
            default = __get_default(field)
            if default is MISSING:
                default = None
                located_errors.append(
                    LocatedValidationError(
                        message=f"Missing required field: {field_name}",
                        json_path=json_path,
                    )
                )
            kwargs[field_name] = default
            continue

        parsed_value = __parse_value(
            value[field_json_name],
            field.type,
            field_json_path,
            located_errors,
        )

        kwargs[field_name] = parsed_value.result

    return cast(Callable[..., T], type_)(**kwargs)

def __asdict(obj, dict_factory=dict):
    if is_dataclass(obj):
        mapping = getattr(obj, _MAPPING_FIELD, {})
        result = []
        for f in fields(obj):
            field_json_name = f.name
            if f.name in mapping:
                field_json_name = mapping[f.name]
                if field_json_name is None:
                    # skip field
                    if __get_default(f) is MISSING:
                        raise Exception(f"Required field cannot be skipped: {f.name}")
                    continue
            value = __asdict(getattr(obj, f.name), dict_factory)
            result.append((field_json_name, value))
        return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        return type(obj)(*[__asdict(v, dict_factory) for v in obj])
    elif isinstance(obj, (list, tuple)):
        return type(obj)(__asdict(v, dict_factory) for v in obj)
    elif isinstance(obj, dict):
        return type(obj)((__asdict(k, dict_factory),
                        __asdict(v, dict_factory))
                        for k, v in obj.items())
    else:
        return deepcopy(obj)

def get_class_or_type_name(type_: Type[Any]) -> str:
    if is_generic(type_):
        return cast(AssumeGeneric, type_).__repr__()

    if is_dataclass(type_):
        return type_.__name__

    return type_.__qualname__


def decode(value: Any, type_: Type[T]) -> T:
    errors: List[LocatedValidationError] = []
    parsed_value = __parse_value(value, type_, located_errors=errors)
    if len(errors):
        raise LocatedValidationErrorCollection(errors)

    if isinstance(parsed_value.result, Exception):
        raise parsed_value.result

    return parsed_value.result


def __encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return serialize_datetime(value)
    if isinstance(value, date):
        return serialize_date(value)
    if isinstance(value, time):
        return serialize_time(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (Decimal, UUID, str)):
        return str(value)
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [__encode(v) for v in value]
    if isinstance(value, dict):
        return {__encode(k): __encode(v) for k, v in value.items()}
    if is_dataclass(value):
        return __encode(__asdict(value))
    if value is None:
        return None
    raise ValueError(f"Unsupported type: {type(value)}")


def encode(value: Any) -> Any:
    return __encode(value)
