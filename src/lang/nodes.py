from dataclasses import replace
from enum import Enum
from typing import cast, override

from lang.position import Position
from lang.token import Token

type StatementContent = BinOp | NumberNode | Scope


class BinOperation(Enum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3


class UnaryOperation(Enum):
    PLUS = 0
    MINUS = 1


class Node:
    def __init__(self, pos_start: Position, pos_end: Position | None = None) -> None:
        self.pos_start: Position = pos_start
        self.pos_end: Position = pos_end or replace(pos_start)


class NumberNode(Node):
    def __init__(self, number_tok: Token) -> None:
        super().__init__(number_tok.pos_start, number_tok.pos_end)

        self.value: int | float = cast(int | float, number_tok.value)

    @override
    def __repr__(self) -> str:
        return f"NumberNode(value={self.value})"


class UnaryOp(Node):
    def __init__(self, pos_start: Position, operand: Node, op: UnaryOperation) -> None:
        super().__init__(pos_start, operand.pos_end)

        self.operand: Node = operand
        self.op: UnaryOperation = op

    @override
    def __repr__(self) -> str:
        return f"UnaryOp(op={self.op}, operand={self.operand})"


class BinOp(Node):
    def __init__(self, lhs: Node, rhs: Node, op: BinOperation) -> None:
        super().__init__(lhs.pos_start, rhs.pos_start)

        self.lhs: Node = lhs
        self.rhs: Node = rhs
        self.op: BinOperation = op

    @override
    def __repr__(self) -> str:
        return f"BinOp(lhs={self.lhs}, op={self.op}, rhs={self.rhs})"


class Statement(Node):
    def __init__(self, inner: StatementContent) -> None:
        super().__init__(inner.pos_start, inner.pos_end)

        self.inner: StatementContent = inner

    @override
    def __repr__(self) -> str:
        return f"Statement(inner={self.inner})"


class Scope(Node):
    def __init__(
        self, pos_start: Position, pos_end: Position, stmt: list[Statement]
    ) -> None:
        super().__init__(pos_start, pos_end)

        self.stmt: list[Statement] = stmt

    @override
    def __repr__(self) -> str:
        return f"Scope(stmt={self.stmt})"


class Program(Node):
    def __init__(
        self, pos_start: Position, pos_end: Position, stmt: list[Statement]
    ) -> None:
        super().__init__(pos_start, pos_end)

        self.stmt: list[Statement] = stmt

    @override
    def __repr__(self) -> str:
        return f"Program(stmt={self.stmt})"
