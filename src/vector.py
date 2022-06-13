from __future__ import annotations

import math
from fractions import Fraction
from numbers import Number
from operator import add, floordiv, mul, sub
from typing import Callable, Iterable, Sequence, TypeVar, overload

from typing_extensions import Self

Numeric = int | float | Fraction

T = TypeVar("T", bound=Numeric)


class Vector(Sequence[T]):
    def __init__(self, data: Iterable) -> None:
        self.data = list(data)

    def __str__(self) -> str:
        return str(list(map(str, self))).replace("'", "")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.data})"

    def __getitem__(self, i: int | slice):
        if isinstance(i, int):
            return self.data[i]
        return type(self)(self.data[i])

    def __setitem__(self, i: int, value: Number) -> None:
        self.data[i] = value

    def __len__(self) -> int:
        return len(self.data)

    def __add__(self, other: Iterable) -> Self:
        if isinstance(other, int):
            return type(self)(x + other for x in self)
        return type(self)(map(add, self, other))

    def origin(self) -> Self:
        return type(self)([0] * len(self))

    def dist(self, other: Iterable | None = None) -> float:
        """Returns the euclidean distance of the vector given its coordinates."""
        if other is None:
            return math.dist(self, self.origin())
        return math.dist(self, other)

    def __matmul__(self, other) -> int:
        return dot(self, other)

    __rmatmul__ = __matmul__

    def normalize(self) -> Self:
        """Returns a normalized vector (magnitude of 1)."""
        dist = self.dist()
        return type(self)(i / dist for i in self)

    @overload
    def __sub__(self, other: Number) -> Self:
        ...

    @overload
    def __sub__(self, other: Self) -> Self:
        ...

    def __sub__(self, other):
        return type(self)(map(sub, other, self))

    def __mul__(self, other) -> Self:
        try:
            return type(self)(map(mul, self, other))
        except TypeError:
            return type(self)(i * other for i in self)

    __rmul__ = __mul__

    def __truediv__(self, other: Self) -> Self:
        return type(self)(Fraction(i, other) for i in self)  # type: ignore

    def __floordiv__(self, other: Self) -> Self:
        return type(self)(map(floordiv, self, other))

    def __pow__(self, exp: Number, mod: Number) -> Self:
        ...

    def __mod__(self, mod: Number) -> Self:
        ...

    def map(self, func: Callable) -> Self:
        ...

    def intersection(self, other) -> Self:
        ...


def cross(v1, v2):
    ...


def dot(v1, v2):
    return sum(i * j for i, j in zip(v1, v2))


if __name__ == "__main__":
    v1 = Vector([3, 1, 9])
    print(v2 := v1.normalize())
    print(v2.dist([0, 0, 0]))
