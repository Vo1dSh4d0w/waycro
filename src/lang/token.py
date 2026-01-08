from dataclasses import dataclass, replace
from enum import Enum, auto

from lang.position import Position


class TokenType(Enum):
    """
    Enum for every token type.
    """

    INT = auto()
    FLOAT = auto()
    IDENTIFIER = auto()
    KEYWORD = auto()
    STRING = auto()

    ASSIGN = auto()
    PLUS = auto()
    MINUS = auto()
    MUL = auto()
    DIV = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    LE = auto()
    GE = auto()
    GT = auto()
    OR = auto()
    AND = auto()
    DOT = auto()
    COMMA = auto()
    SEMI = auto()

    COLON = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()

    EOF = auto()


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

    def matches(self, type: TokenType, value: str | int | float):
        return type == self.type and value == self.value

    def __post_init__(self):
        if self.pos_end is None:
            self.pos_end = replace(self.pos_start)
