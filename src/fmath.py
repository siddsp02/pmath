"""Math module for inspecting double precision
floating point numbers according to IEEE 754."""

import struct
from typing import NamedTuple


class Float(NamedTuple):
    value: float

    def __str__(self) -> str:
        return str(self.value)

    @property
    def sign_bit(self) -> int:
        value_bin = struct.pack(">d", self.value)
        value_int = struct.unpack(">Q", value_bin)[0]
        mask = 1 << 63
        return (mask & value_int) >> 63

    @property
    def bias(self) -> int:
        return 1023

    @property
    def exponent(self) -> int:
        value_bin = struct.pack(">d", self.value)
        value_int = struct.unpack(">Q", value_bin)[0]
        mask = 0x7FF << 52
        return ((mask & value_int) >> 52) - self.bias

    @property
    def mantissa(self) -> int:
        value_bin = struct.pack(">d", self.value)
        value_int = struct.unpack(">Q", value_bin)[0]
        mask = (1 << 52) - 1
        return mask & value_int

    def __float__(self) -> float:
        return self.value

    def __bytes__(self) -> bytes:
        return struct.pack(">d", self.value)

    def scientific_notation(self) -> str:
        return f"{'+-'[self.sign_bit]}1.{self.mantissa:0>52b} * 2**{self.exponent}"


x = Float(0.1)
r = f"{x.sign_bit} {x.exponent:0>11b} {x.mantissa:0>52b}"
print(x.scientific_notation())
print(x)
