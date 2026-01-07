from typing import Self, cast

from lang.error import Error, SyntaxError
from lang.nodes import (
    Attribute,
    BinOp,
    BinOperation,
    Call,
    FloatLiteral,
    IntLiteral,
    Node,
    Program,
    Scope,
    Statement,
    StatementContent,
    StringLiteral,
    SymbolAccess,
    SymbolDeclaration,
    SymbolDeclarationScope,
    UnaryOp,
    UnaryOperation,
)
from lang.position import Position
from lang.token import Token, TokenType


class ParseResult:
    def __init__(self) -> None:
        self.error: Error | None = None
        self.result: Node | None = None
        self.consume_count: int = 0

    def register_consume(self):
        self.consume_count += 1

    def register(self, result: Self) -> Self:
        self.consume_count += result.consume_count
        return result

    def try_register(self, result: Self) -> tuple[Self, int]:
        if not result.error:
            self.consume_count += result.consume_count
            return result, 0
        return result, result.consume_count

    def success(self, result: Node) -> Self:
        self.result = result
        return self

    def failure(self, error: Error) -> Self:
        self.error = error
        return self


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens: list[Token] = tokens
        self.len: int = len(tokens)
        self.idx: int = 0

    def consume(self, parse_result: ParseResult | None = None) -> Token:
        if self.idx >= self.len:
            raise Exception("Tokens exhausted")

        tok = self.tokens[self.idx]
        self.idx += 1

        if parse_result:
            parse_result.register_consume()

        return tok

    def reverse(self, count: int):
        self.idx -= count

    def peek(self, ahead: int = 0) -> Token:
        return self.tokens[self.idx + ahead]

    def tok_to_bin_operation(self, tok: Token) -> BinOperation:
        match tok.type:
            case TokenType.PLUS:
                return BinOperation.ADD
            case TokenType.MINUS:
                return BinOperation.SUB
            case TokenType.MUL:
                return BinOperation.MUL
            case TokenType.DIV:
                return BinOperation.DIV
            case _:
                raise Exception(
                    f"Cannot convert token ({tok.type}) to binary operation."
                )

    def tok_to_unary_operation(self, tok: Token) -> UnaryOperation:
        match tok.type:
            case TokenType.PLUS:
                return UnaryOperation.PLUS
            case TokenType.MINUS:
                return UnaryOperation.MINUS
            case _:
                raise Exception(
                    f"Cannot convert token ({tok.type}) to unary operation."
                )

    def tok_to_symbol_declaration_scope(self, tok: Token) -> SymbolDeclarationScope:
        match tok.value:
            case "local":
                return SymbolDeclarationScope.LOCAL
            case "global":
                return SymbolDeclarationScope.GLOBAL
            case "export":
                return SymbolDeclarationScope.EXPORT
            case _:
                raise Exception(
                    f"Cannot convert token ({tok.value}) to symbol declaration scope."
                )

    def parse(self) -> ParseResult:
        return self.parse_program()

    def parse_program(self) -> ParseResult:
        res = ParseResult()

        stmts: list[Statement] = []

        while self.peek().type != TokenType.EOF:
            stmt = res.register(self.parse_statement())
            if stmt.error:
                return res.failure(stmt.error)

            stmts.append(cast(Statement, stmt.result))

        return res.success(Program(stmts[0].pos_start, stmts[-1].pos_end, stmts))

    def parse_statement(self) -> ParseResult:
        res = ParseResult()

        next_tok = self.peek()

        if next_tok.type == TokenType.LBRACE:
            scope = res.register(self.parse_scope())
            if scope.error:
                return res.failure(scope.error)
            return res.success(cast(Scope, scope.result))
        elif next_tok.type == TokenType.KEYWORD and next_tok.value in (
            "local",
            "global",
            "export",
        ):
            symbol_declaration = res.register(self.parse_symbol_declaration())
            if symbol_declaration.error:
                return res.failure(symbol_declaration.error)
            return res.success(
                Statement(cast(StatementContent, symbol_declaration.result))
            )
        else:
            expr = res.register(self.parse_expr())
            if expr.error:
                return res.failure(expr.error)

            next_tok = self.peek()
            if next_tok.type != TokenType.SEMI:
                return res.failure(
                    SyntaxError("expected ';'", next_tok.pos_start, next_tok.pos_end)
                )

            _ = self.consume(res)

            return res.success(Statement(cast(StatementContent, expr.result)))

    def parse_scope(self) -> ParseResult:
        res = ParseResult()
        lbrace = self.consume(res)

        stmts: list[Statement] = []

        while self.peek().type != TokenType.RBRACE:
            stmt = res.register(self.parse_statement())
            if stmt.error:
                return res.failure(stmt.error)

            stmts.append(cast(Statement, stmt.result))

        rbrace = self.consume(res)

        return res.success(
            Scope(lbrace.pos_start, cast(Position, rbrace.pos_end), stmts)
        )

    def parse_symbol_declaration(self) -> ParseResult:
        res = ParseResult()

        scope = self.consume(res)

        identifier_tok = self.peek()
        if identifier_tok.type != TokenType.IDENTIFIER:
            return res.failure(
                SyntaxError(
                    "expected identifier",
                    identifier_tok.pos_start,
                    cast(Position, identifier_tok.pos_end),
                )
            )

        _ = self.consume(res)

        assign_tok = self.peek()
        if assign_tok.type != TokenType.ASSIGN:
            return res.failure(
                SyntaxError(
                    "expected '='",
                    assign_tok.pos_start,
                    cast(Position, assign_tok.pos_end),
                )
            )

        _ = self.consume(res)

        initial_value = res.register(self.parse_expr())
        if initial_value.error:
            return res.failure(initial_value.error)

        semi = self.peek()
        if semi.type != TokenType.SEMI:
            return res.failure(
                SyntaxError(
                    "expected ';'", semi.pos_start, cast(Position, semi.pos_end)
                )
            )

        _ = self.consume(res)

        return res.success(
            SymbolDeclaration(
                scope.pos_start,
                cast(Position, semi.pos_end),
                self.tok_to_symbol_declaration_scope(scope),
                identifier_tok,
                cast(Node, initial_value.result),
            )
        )

    def parse_expr(self) -> ParseResult:
        res = ParseResult()

        lhs = res.register(self.parse_term())
        if lhs.error:
            return res.failure(lhs.error)

        if self.peek().type not in (
            TokenType.PLUS,
            TokenType.MINUS,
        ):
            return res.success(cast(Node, lhs.result))

        op = self.consume(res)

        rhs = res.register(self.parse_term())
        if rhs.error:
            return res.failure(rhs.error)

        bin_op = BinOp(
            cast(Node, lhs.result),
            cast(Node, rhs.result),
            self.tok_to_bin_operation(op),
        )
        while self.peek().type in (TokenType.PLUS, TokenType.MINUS):
            op = self.consume(res)
            rhs = res.register(self.parse_term())
            if rhs.error:
                return res.failure(rhs.error)

            bin_op = BinOp(
                bin_op,
                cast(Node, rhs.result),
                self.tok_to_bin_operation(op),
            )

        return res.success(bin_op)

    def parse_term(self) -> ParseResult:
        res = ParseResult()

        lhs = res.register(self.parse_postfix())
        if lhs.error:
            return res.failure(lhs.error)

        if self.peek().type not in (
            TokenType.MUL,
            TokenType.DIV,
        ):
            return res.success(cast(Node, lhs.result))

        op = self.consume(res)

        rhs = res.register(self.parse_postfix())
        if rhs.error:
            return res.failure(rhs.error)

        bin_op = BinOp(
            cast(Node, lhs.result),
            cast(Node, rhs.result),
            self.tok_to_bin_operation(op),
        )
        while self.peek().type in (TokenType.MUL, TokenType.DIV):
            op = self.consume(res)
            rhs = res.register(self.parse_postfix())
            if rhs.error:
                return res.failure(rhs.error)

            bin_op = BinOp(
                bin_op,
                cast(Node, rhs.result),
                self.tok_to_bin_operation(op),
            )

        return res.success(bin_op)

    def parse_postfix(self) -> ParseResult:
        res = ParseResult()

        lhs = res.register(self.parse_atom())
        if lhs.error:
            return res.failure(lhs.error)

        if self.peek().type not in (TokenType.DOT, TokenType.LPAREN):
            return res.success(cast(Node, lhs.result))

        op = self.consume(res)

        final_node: Node

        if op.type == TokenType.DOT:
            next_tok = self.peek()
            if next_tok.type != TokenType.IDENTIFIER:
                return res.failure(
                    SyntaxError(
                        "expected identifier", next_tok.pos_start, next_tok.pos_end
                    )
                )

            identifier = self.consume(res)

            final_node = Attribute(cast(Node, lhs.result), identifier)
        else:
            args: list[Node] = []

            while self.peek().type != TokenType.RPAREN:
                arg = res.register(self.parse_expr())

                if arg.error:
                    return res.failure(arg.error)

                args.append(cast(Node, arg.result))

                if self.peek().type != TokenType.COMMA:
                    break
                _ = self.consume(res)

            next_tok = self.peek()
            if next_tok.type != TokenType.RPAREN:
                return res.failure(
                    SyntaxError("expected ')'", next_tok.pos_start, next_tok.pos_end)
                )

            _ = self.consume(res)

            final_node = Call(
                cast(Position, next_tok.pos_end), cast(Node, lhs.result), args
            )

        while self.peek().type in (TokenType.DOT, TokenType.LPAREN):
            op = self.consume(res)

            if op.type == TokenType.DOT:
                next_tok = self.peek()
                if next_tok.type != TokenType.IDENTIFIER:
                    return res.failure(
                        SyntaxError(
                            "expected identifier", next_tok.pos_start, next_tok.pos_end
                        )
                    )

                identifier = self.consume(res)

                final_node = Attribute(final_node, identifier)
            else:
                args = []

                while self.peek().type != TokenType.RPAREN:
                    arg = res.register(self.parse_expr())

                    if arg.error:
                        return res.failure(arg.error)

                    args.append(cast(Node, arg.result))

                    if self.peek().type != TokenType.COMMA:
                        break
                    _ = self.consume(res)

                next_tok = self.peek()
                if next_tok.type != TokenType.RPAREN:
                    return res.failure(
                        SyntaxError(
                            "expected ')'", next_tok.pos_start, next_tok.pos_end
                        )
                    )

                _ = self.consume(res)

                final_node = Call(cast(Position, next_tok.pos_end), final_node, args)

        return res.success(final_node)

    def parse_atom(self) -> ParseResult:
        res = ParseResult()

        next_tok = self.peek()

        if next_tok.type == TokenType.LPAREN:
            _ = self.consume(res)
            expr = res.register(self.parse_expr())

            if expr.error:
                return res.failure(expr.error)

            next_tok = self.consume(res)
            if next_tok.type != TokenType.RPAREN:
                return res.failure(
                    SyntaxError("expected ')'", next_tok.pos_start, next_tok.pos_end)
                )

            return res.success(cast(Node, expr.result))
        elif next_tok.type in (TokenType.PLUS, TokenType.MINUS):
            op = self.consume(res)
            operand = res.register(self.parse_atom())

            if operand.error:
                return res.failure(operand.error)

            return res.success(
                UnaryOp(
                    op.pos_start,
                    cast(Node, operand.result),
                    self.tok_to_unary_operation(op),
                )
            )
        elif next_tok.type == TokenType.INT:
            _ = self.consume(res)
            return res.success(IntLiteral(next_tok))
        elif next_tok.type == TokenType.FLOAT:
            _ = self.consume(res)
            return res.success(FloatLiteral(next_tok))
        elif next_tok.type == TokenType.STRING:
            _ = self.consume(res)
            return res.success(StringLiteral(next_tok))
        elif next_tok.type == TokenType.IDENTIFIER:
            _ = self.consume(res)
            return res.success(SymbolAccess(next_tok))
        else:
            return res.failure(
                SyntaxError(
                    "expected number, string, identifier or '('",
                    next_tok.pos_start,
                    next_tok.pos_end,
                )
            )
