# !usr/bin/env python3

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from functools import reduce
from itertools import chain, product, repeat
from operator import matmul
from typing import Iterable, Iterator, Sequence, overload

from typing_extensions import Self

from vector import T, Vector


class Matrix(Sequence[Vector[T]]):
    def __init__(self, data: Iterable[Iterable[T]]) -> None:
        self.data = list(map(Vector, data))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.data})"

    def __str__(self) -> str:
        return "\n".join(map(str, self.data))

    def __len__(self) -> int:
        return len(self.data)

    def shape(self) -> tuple[int, int]:
        return len(self), len(self[0])

    @property
    def invertible(self) -> bool:
        return self.det() != 0

    @classmethod
    def fill(cls, value: T, rows: int, cols: int) -> Self:
        return cls([[value for _ in range(cols)] for _ in range(rows)])

    @overload
    @classmethod
    def zeros(cls, rows: int) -> Self:
        ...

    @overload
    @classmethod
    def zeros(cls, rows: int, cols: int) -> Self:
        ...

    @classmethod
    def zeros(cls, rows, cols=None):
        if cols is None:
            cols = rows
        return cls.fill(0, rows, cols)

    @overload
    def __getitem__(self, i: int) -> Vector[T]:
        ...

    @overload
    def __getitem__(self, i: tuple[int, int]) -> T:
        ...

    def __getitem__(self, i):
        try:
            return self.data[i]
        except TypeError:
            row, col = i
            return self.data[row][col]

    @overload
    def __setitem__(self, i: int, value: Vector[T]) -> None:
        ...

    @overload
    def __setitem__(self, i: tuple[int, int], value) -> None:
        ...

    def __setitem__(self, i, value):
        try:
            self.data[i] = value
        except TypeError:
            row, col = i
            self.data[row][col] = value

    def __matmul__(self, other: Self) -> Self:
        """Returns the dot product of two matrices."""
        match self.shape(), other.shape():
            case (u, v), (q, r) if v == q:
                t = other.transpose()
                m = type(self).zeros(u, r)
                for i, j in m.indices():
                    m[i, j] = self[i] @ t[j]
                return m
        raise ValueError(f"{type(self).__name__} shapes are incompatible.")

    __mul__ = __matmul__
    __rmatmul__ = __matmul__
    __rmul__ = __mul__

    def enumerated(self) -> Iterator[tuple[tuple[int, int], T]]:
        flat_values = chain.from_iterable(self)
        yield from zip(self.indices(), flat_values)

    def __pow__(self, exp: int) -> Self:
        if exp == -1:
            return self.inverse()
        if exp < -1:
            return pow(self.inverse(), abs(exp))
        return reduce(matmul, repeat(self, exp))

    def indices(self) -> Iterator[tuple[int, int]]:
        """Returns an iterator over the index pairs of the matrix."""
        rows, cols = self.shape()
        yield from product(range(rows), range(cols))

    def minor(self, i: int, j: int) -> Self:
        """Returns the minor matrix when given a row and column
        to exclude for slicing.
        """
        ret = [[] for _ in self]
        for row, col in self.indices():
            if row == i or col == j:
                continue
            ret[row].append(self[row, col])
        del ret[i]
        return type(self)(ret)

    def rref(self) -> Self:
        """Returns the Reduced Row Echelon Form of a matrix.

        References:
            - https://en.wikipedia.org/wiki/Row_echelon_form
        """
        m = deepcopy(self)
        rows, cols = self.shape()
        lead = 0
        for r in range(rows):
            if cols <= lead:
                return m
            i = r
            while m[i, lead] == 0:
                i += 1
                if i == rows:
                    i, lead = r, lead + 1
                    if cols == lead:
                        return m
            if i != r:
                m[i], m[r] = m[r], m[i]
            m[r] /= m[r, lead]  # type: ignore
            for j in range(rows):
                if j != r:
                    m[j] = m[r] * m[j, lead] - m[j]
            lead += 1
        return m

    def ref(self) -> Self:
        """Returns the Row Echelon Form of a matrix.

        References:
            - https://en.wikipedia.org/wiki/Row_echelon_form
        """
        m = deepcopy(self)
        rows, cols = self.shape()
        for r in range(rows):
            n = rows - 1
            if all(m[r, c] == 0 for c in range(cols)):
                m[r], m[n], rows = m[n], m[r], n
        p, r, lim = 0, 1, min(rows, cols)
        while p < lim:
            r, next_pivot = 1, False
            while m[p, p] == 0:
                n = p + r
                if n <= rows:
                    p, next_pivot = p + 1, True
                    break
                m[p], m[n], r = m[n], m[p], r + 1
            if next_pivot:
                continue
            for r in range(1, rows - p):
                n = p + r
                if m[n, p] != 0:
                    x = Fraction(-m[n, p], m[p, p])  # type: ignore
                    for c in range(p, cols):
                        m[n, c] += m[p, c] * x
            p += 1
        return m

    def det(self) -> int:
        """Returns the determinant of an N*N matrix, or None
        if the matrix is not square.
        """
        match self.shape():
            case (2, 2):
                (a, b), (c, d) = self
                return a * d - b * c  # type: ignore
            case (i, j) if i == j:
                return sum(
                    (-1)**i * x * self.minor(0, i).det()
                    for i, x in enumerate(self[0])
                )
        raise ValueError("Matrix does not have determinant.")

    def identity(self) -> Self:
        """Returns the identity matrix of a matrix."""
        shape = self.shape()
        identity_matrix = type(self).zeros(*shape)
        for i, _ in enumerate(self):
            identity_matrix[i, i] = 1
        return identity_matrix

    def transpose(self) -> Self:
        return type(self)(map(list, zip(*self)))

    def adj(self) -> Self:
        """Returns the adjoint of a square matrix."""
        match self.shape():
            case (2, 2):
                (a, b), (c, d) = self
                return type(self)([[d, -b], [-c, a]])
            case (i, j) if i == j:
                m = type(self).zeros(i, j)
                for i, j in self.indices():
                    m[i, j] = (-1) ** (i + j) * self.minor(i, j).det()
                return m.transpose()
        raise ValueError("Matrix does not have an adjoint.")

    def inverse(self) -> Self:
        if not self.invertible:
            raise ValueError("Not invertible.")
        return Fraction(1, self.det()) * self.adj()  # type: ignore

    def eigenvalues(self) -> list[T]:
        ...

    def eigenvectors(self) -> list[Vector[T]]:
        ...

    def rotate(self, theta: T) -> Self:
        ...


if __name__ == "__main__":
    m1 = Matrix(
        [
            [1, 2, 4],
            [3, 4, 1],
            [6, 8, 9],
        ],
    )
    m2 = Matrix(
        [
            [2, 0, 9],
            [1, 2, 8],
            [6, 6, 3],
        ],
    )
    print(m1 @ m2)
