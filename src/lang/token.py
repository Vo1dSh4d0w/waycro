from dataclasses import dataclass, replace
from enum import Enum

from lang.position import Position


class TokenType(Enum):
    """
    Enum for every token type.
    """

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
    DOT = 20
    COMMA = 21

    COLON = 30
    LPAREN = 31
    RPAREN = 32
    LBRACE = 33
    RBRACE = 34

    EOF = 50


@dataclass
class Token:
    """
    A singular token.

    type
        the type of the token

    value
        an optional value of the token, defaults to None
    """

    type: TokenType
    pos_start: Position
    pos_end: Position | None = None
    value: str | int | float | None = None

    def __post_init__(self):
        if self.pos_end is None:
            self.pos_end = replace(self.pos_start)
