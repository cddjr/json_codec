from typing import Any, Generator, Type

from json_codec.types import (
    ParseProcessResult,
    ParseProcessYield,
    TypeDecoder,
    ValidationError,
)


class BoolTypeDecoder(TypeDecoder[bool]):
    def parse(
        self, value: Any, *types: Type[Any]
    ) -> Generator[
        ParseProcessYield[Any],
        ParseProcessResult[Any],
        ParseProcessResult[bool],
    ]:
        try:
            if isinstance(value, str):
                if value.lower() == "true":
                    return self._success(True)
                elif value.lower() == "false":
                    return self._success(False)
                else:
                    # FIXME
                    return self._success(bool(value))
            elif isinstance(value, bool):
                return self._success(value)
            elif isinstance(value, int):
                return self._success(bool(value))
            raise ValueError
        except ValueError:
            return self._failure(
                ValidationError(f"Expected type bool, but {value} is not a valid value")
            )
        yield
