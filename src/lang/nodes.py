from dataclasses import replace
from enum import Enum, auto
from typing import cast, override

from lang.position import Position
from lang.token import Token

type StatementContent = (
    UnaryOp | BinOp | IntLiteral | Scope | SymbolDeclaration | Assignment
)


class BinOperation(Enum):
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()

    EQ = auto()
    NEQ = auto()
    LT = auto()
    LE = auto()
    GE = auto()
    GT = auto()


class UnaryOperation(Enum):
    PLUS = auto()
    MINUS = auto()


class SymbolDeclarationScope(Enum):
    LOCAL = auto()
    GLOBAL = auto()
    EXPORT = auto()


class Node:
    def __init__(self, pos_start: Position, pos_end: Position | None = None) -> None:
        self.pos_start: Position = pos_start
        self.pos_end: Position = pos_end or replace(pos_start)


class IntLiteral(Node):
    def __init__(self, number_tok: Token) -> None:
        super().__init__(number_tok.pos_start, number_tok.pos_end)

        self.value: int = cast(int, number_tok.value)

    @override
    def __repr__(self) -> str:
        return f"IntLiteral(value={self.value})"


class FloatLiteral(Node):
    def __init__(self, number_tok: Token) -> None:
        super().__init__(number_tok.pos_start, number_tok.pos_end)

        self.value: float = cast(float, number_tok.value)

    @override
    def __repr__(self) -> str:
        return f"FloatLiteral(value={self.value})"


class StringLiteral(Node):
    def __init__(self, string_tok: Token) -> None:
        super().__init__(string_tok.pos_start, string_tok.pos_end)

        self.value: str = cast(str, string_tok.value)

    @override
    def __repr__(self) -> str:
        return f"StringLiteral(value={self.value})"


class SymbolAccess(Node):
    def __init__(self, identifier_tok: Token) -> None:
        super().__init__(identifier_tok.pos_start, identifier_tok.pos_end)

        self.identifier: str = cast(str, identifier_tok.value)

    @override
    def __repr__(self) -> str:
        return f"SymbolAccess(identifer={self.identifier})"


class SymbolDeclaration(Node):
    def __init__(
        self,
        pos_start: Position,
        pos_end: Position,
        scope: SymbolDeclarationScope,
        identifier_tok: Token,
        initial_value: Node,
    ) -> None:
        super().__init__(pos_start, pos_end)

        self.scope: SymbolDeclarationScope = scope
        self.identifier: str = cast(str, identifier_tok.value)
        self.initial_value: Node = initial_value

    @override
    def __repr__(self) -> str:
        return f"SymbolDeclaration(scope={self.scope}, identifier={self.identifier}, initial_value={self.initial_value})"


class Assignment(Node):
    def __init__(self, pos_end: Position, lhs: Node, rhs: Node) -> None:
        super().__init__(lhs.pos_start, pos_end)

        self.lhs: Node = lhs
        self.rhs: Node = rhs

    @override
    def __repr__(self) -> str:
        return f"Assignment(lhs={self.lhs}, rhs={self.rhs})"


class IfStatement(Node):
    def __init__(
        self,
        pos_start: Position,
        condition: Node,
        body: Node,
        else_node: Node | None = None,
    ) -> None:
        super().__init__(pos_start, else_node.pos_end if else_node else body.pos_end)

        self.condition: Node = condition
        self.body: Node = body
        self.else_node: Node | None = else_node

    @override
    def __repr__(self) -> str:
        return f"IfStatement(condition={self.condition}, body={self.body}, else_node={self.else_node})"


class Attribute(Node):
    def __init__(self, lhs: Node, identifier_tok: Token) -> None:
        super().__init__(lhs.pos_start, identifier_tok.pos_end)

        self.lhs: Node = lhs
        self.identifier: str = cast(str, identifier_tok.value)

    @override
    def __repr__(self) -> str:
        return f"Attribute(lhs={self.lhs}, identifier={self.identifier})"


class Call(Node):
    def __init__(self, pos_end: Position, lhs: Node, args: list[Node]) -> None:
        super().__init__(lhs.pos_start, pos_end)

        self.lhs: Node = lhs
        self.args: list[Node] = args

    @override
    def __repr__(self) -> str:
        return f"Call(lhs={self.lhs}, args={self.args})"


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
