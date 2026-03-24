from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from typing import TypeVar

    T = TypeVar('T')

    def IsInstance(arg: type[T]) -> T: ...
    def IsDatetime(*args: Any, **kwargs: Any) -> datetime: ...
    def IsFloat(*args: Any, **kwargs: Any) -> float: ...
    def IsInt(*args: Any, **kwargs: Any) -> int: ...
    def IsNow(*args: Any, **kwargs: Any) -> datetime: ...
    def IsStr(*args: Any, **kwargs: Any) -> str: ...
    def IsBytes(*args: Any, **kwargs: Any) -> bytes: ...
    def IsList(*args: T, **kwargs: Any) -> list[T]: ...
else:
    from dirty_equals import IsBytes, IsDatetime, IsFloat, IsInstance, IsInt, IsList, IsNow as _IsNow, IsStr

    def IsNow(*args: Any, **kwargs: Any):
        if 'delta' not in kwargs:
            kwargs['delta'] = 10
        return _IsNow(*args, **kwargs)

__all__ = (
    'IsDatetime',
    'IsFloat',
    'IsNow',
    'IsStr',
    'IsBytes',
    'IsInt',
    'IsInstance',
    'IsList',
)

pytest_plugins = []
