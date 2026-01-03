from dataclasses import dataclass
from enum import Enum


class TokenType(Enum):
    INT = 0
    FLOAT = 1
    IDENTIFIER = 2
    KEYWORD = 3
    STRING = 4

    PLUS = 10
    MINUS = 11
    MUL = 12
    DIV = 13
    EQ = 14
    NEQ = 15
    LT = 16
    LE = 17
    GE = 18
    GT = 19

    EOF = 50


@dataclass
class Token:
    type: TokenType
    value: str | int | float | None = None
