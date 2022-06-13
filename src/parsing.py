from collections import deque
from functools import total_ordering
from numbers import Number
import re
from typing import Any
from typing_extensions import Self

OPERATORS = [
    "(",
    ")",
    "**",
    "*",
    "@",
    "//",
    "/",
    "+",
    "-",
    "^",
    "|",
    "%",
    "<<",
    ">>",
    "&",
    "~",
]


class Expression:
    def __init__(self, string: str, vars: dict[str, Any]) -> None:
        self.string = string
        self.vars = vars

    def value(self) -> Any:
        ret = self.string
        d = {k: str(v) for (k, v) in self.vars.items()}
        for k, v in d.items():
            ret = ret.replace(k, v)
        return eval(ret)

    def rpn(self) -> deque[str]:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(string={self.string!r}, vars={self.vars})"

    def __str__(self) -> str:
        return self.string

    def __eq__(self, other: Self) -> bool:
        ...

    def __gt__(self, other: Self) -> bool:
        ...


@total_ordering
class var:
    def __init__(self, name: str, value: Number | None = None) -> None:
        self.name = name
        self.value = value

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        return self.name

    def bind(self, value: Number) -> None:
        ...

    def unbound(self) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    def __gt__(self, other) -> bool:
        ...


def priority(op: str) -> int:
    match op:
        case "(" | ")":
            return 0
        case "**":
            return 1
        case "~":
            return 2
        case "*" | "@" | "/" | "//" | "%":
            return 3
        case "+" | "-":
            return 4
        case "<<" | ">>":
            return 5
        case "&":
            return 6
        case "^":
            return 7
        case "|":
            return 8
        case _:
            return -1


def parse_tokens(expr: str):
    ops = "(" + "|".join(map(re.escape, sorted(OPERATORS, reverse=True))) + ")"
    expr = re.sub(r"\s+", "", expr)
    return filter(None, re.split(ops, expr))


def isfloat(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def rpn(tokens: list[str]) -> deque[str]:
    """Converts a list of tokens to the equivalent expression into
    Reverse Polish Notation using the Shunting Yard Algorithm.

    References:
        - https://en.wikipedia.org/wiki/Shunting_yard_algorithm
    """
    operator_stack = deque()
    output_queue = deque()
    for token in tokens:
        if token.isalnum() or isfloat(token):
            output_queue.append(token)
        elif token == "(":
            operator_stack.append(token)
        elif token == ")":
            while operator_stack[-1] != "(":
                assert operator_stack
                output_queue.append(operator_stack.pop())
            assert operator_stack[-1] == "("
            operator_stack.pop()
        elif token in OPERATORS:
            while (
                operator_stack
                and (op := operator_stack[-1]) != "("
                and (
                    priority(token) > priority(op)
                    or (
                        # If the precedence of both operators is the same,
                        # check if the operator token is left-associative.
                        priority(token) == priority(op)
                        and token in {"*", "/", "//", "+", "-"}
                    )
                )
            ):
                output_queue.append(operator_stack.pop())
            operator_stack.append(token)
    while operator_stack:
        assert operator_stack[-1] != "("
        output_queue.append(operator_stack.pop())
    return output_queue


def eval_rpn(expr: deque[str]) -> Any:
    """Evaluates an expression given in Reverse Polish Notation."""
    stack = []
    while expr:
        while expr[0] not in OPERATORS:
            stack.append(expr.popleft())
        *stack, m, n = stack
        op = expr.popleft()
        s = eval(f"({m}){op}({n})")  # Preserves negation for operands.
        stack.append(s)
    return stack.pop()


if __name__ == "__main__":
    s = var("s", value=12)  # type: ignore
    q = var("q", value=13)  # type: ignore
