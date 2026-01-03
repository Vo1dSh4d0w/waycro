from dataclasses import replace
from string import ascii_letters, digits

from lang.error import Error, InvalidCharError
from lang.position import Position
from lang.token import Token, TokenType

DIGITS = digits + "."
ALLOWED_CHARS_FIRST = ascii_letters + "_"
ALLOWED_CHARS = ALLOWED_CHARS_FIRST + digits
WHITESPACE = "\n\t "

KEYWORDS = {
    "if",
    "else",
    "while",
    "until",
    "repeat",
    "times",
    "forever",
    "fn",
    "return",
}


class Lexer:
    def __init__(self, txt: str, fn: str) -> None:
        self.txt: str = txt
        self.len: int = len(txt)
        self.pos: Position = Position(fn, txt, 0, 1, 1)

    def peek(self, ahead: int = 0) -> str | None:
        if self.pos.idx + ahead >= self.len:
            return None
        return self.txt[self.pos.idx + ahead]

    def consume(self) -> str:
        if self.pos.idx >= self.len:
            raise Exception("EOF")

        char = self.txt[self.pos.idx]
        self.pos.col += 1
        self.pos.idx += 1

        if char == "\n":
            self.pos.ln += 1
            self.pos.col = 1

        return char

    def tokenize(self) -> tuple[list[Token] | None, Error | None]:
        char: str | None
        toks: list[Token] = []

        while (char := self.peek()) is not None:
            match char:
                case v if v in DIGITS:
                    tok, err = self.make_number()
                    if err or tok is None:
                        return None, err
                    toks.append(tok)
                case v if v in ALLOWED_CHARS_FIRST:
                    toks.append(self.make_identifier())
                case "+":
                    _ = self.consume()
                    toks.append(Token(TokenType.PLUS))
                case "-":
                    _ = self.consume()
                    toks.append(Token(TokenType.MINUS))
                case "*":
                    _ = self.consume()
                    toks.append(Token(TokenType.MUL))
                case "/":
                    _ = self.consume()
                    toks.append(Token(TokenType.DIV))
                case "<":
                    _ = self.consume()
                    if self.peek() == "=":
                        _ = self.consume()
                        toks.append(Token(TokenType.LE))
                    else:
                        toks.append(Token(TokenType.LT))
                case ">":
                    _ = self.consume()
                    if self.peek() == "=":
                        _ = self.consume()
                        toks.append(Token(TokenType.GE))
                    else:
                        toks.append(Token(TokenType.GT))
                case v if v in WHITESPACE:
                    _ = self.consume()
                case _:
                    return None, InvalidCharError(char, self.pos)

        toks.append(Token(TokenType.EOF))
        return toks, None

    def make_number(self) -> tuple[Token | None, Error | None]:
        char: str | None
        num: str = ""
        is_float: bool = False

        while (char := self.peek()) is not None and char in DIGITS + ".":
            if char == ".":
                if is_float:
                    return None, InvalidCharError(char, self.pos)
                is_float = True
            num += self.consume()

        if is_float:
            return Token(TokenType.FLOAT, float(num)), None
        else:
            return Token(TokenType.INT, int(num)), None

    def make_identifier(self) -> Token:
        char: str | None
        val: str = ""

        while (char := self.peek()) is not None and char in ALLOWED_CHARS:
            val += self.consume()

        if val in KEYWORDS:
            return Token(TokenType.KEYWORD, val)
        else:
            return Token(TokenType.IDENTIFIER, val)
