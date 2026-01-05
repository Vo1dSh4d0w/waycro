from dataclasses import replace
from string import ascii_letters, digits

from lang.error import Error, InvalidCharError
from lang.position import Position
from lang.token import Token, TokenType

DIGITS: str = digits + "."
ALLOWED_CHARS_FIRST: str = ascii_letters + "_"
ALLOWED_CHARS: str = ALLOWED_CHARS_FIRST + digits
WHITESPACE: str = "\n\t "

KEYWORDS: set[str] = {
    "if",
    "else",
    "while",
    "until",
    "repeat",
    "times",
    "forever",
    "fn",
    "return",
    "exit",
    "using",
}


class Lexer:
    """Class for converting a text to the corresponding tokens."""

    def __init__(self, txt: str, fn: str) -> None:
        """

        txt
            the text to be converted to tokens

        fn
            the name of the file containing the text
        """
        self.txt: str = txt
        self.len: int = len(txt)
        self.pos: Position = Position(fn, txt, 0, 1, 1)

    def peek(self, ahead: int = 0) -> str | None:
        """
        fetches the character from the text on index + ahead without changing position

        ahead
            the offset of current position, defaults to 0 (current position)

        returns the fetched character
        """
        if self.pos.idx + ahead >= self.len:
            return None
        return self.txt[self.pos.idx + ahead]

    def consume(self) -> str:
        """
        advances the current position to the next character

        throws if there are no characters left

        returns the consumed character
        """
        if self.pos.idx >= self.len:
            raise Exception("consume: EOF")

        char = self.txt[self.pos.idx]
        self.pos.col += 1
        self.pos.idx += 1

        if char == "\n":
            self.pos.ln += 1
            self.pos.col = 1

        return char

    def tokenize(self) -> tuple[list[Token] | None, Error | None]:
        """
        converts the given text into tokens

        returns a tuple, where the first element is the list of tokens or None if an error occured and the second element is the error or None if no error occured
        """
        char: str | None
        toks: list[Token] = []

        while (char := self.peek()) is not None:
            match char:
                case ".":
                    next_char = self.peek(1)
                    if next_char is not None and next_char not in DIGITS:
                        _ = self.consume()
                        toks.append(Token(TokenType.DOT, replace(self.pos)))
                    else:
                        tok, err = self.make_number()
                        if err or tok is None:
                            return None, err
                        toks.append(tok)
                case v if v in DIGITS:
                    tok, err = self.make_number()
                    if err or tok is None:
                        return None, err
                    toks.append(tok)
                case v if v in ALLOWED_CHARS_FIRST:
                    toks.append(self.make_identifier())
                case "+":
                    toks.append(Token(TokenType.PLUS, replace(self.pos)))
                    _ = self.consume()
                case "-":
                    toks.append(Token(TokenType.MINUS, replace(self.pos)))
                    _ = self.consume()
                case "*":
                    toks.append(Token(TokenType.MUL, replace(self.pos)))
                    _ = self.consume()
                case "/":
                    toks.append(Token(TokenType.DIV, replace(self.pos)))
                    _ = self.consume()
                case "<":
                    pos_start = replace(self.pos)
                    _ = self.consume()
                    if self.peek() == "=":
                        toks.append(Token(TokenType.LE, pos_start, replace(self.pos)))
                        _ = self.consume()
                    else:
                        toks.append(Token(TokenType.LT, pos_start))
                case ">":
                    pos_start = replace(self.pos)
                    _ = self.consume()
                    if self.peek() == "=":
                        toks.append(Token(TokenType.GE, pos_start, replace(self.pos)))
                        _ = self.consume()
                    else:
                        toks.append(Token(TokenType.GT, pos_start))
                case ":":
                    toks.append(Token(TokenType.COLON, replace(self.pos)))
                    _ = self.consume()
                case ":":
                    toks.append(Token(TokenType.SEMI, replace(self.pos)))
                    _ = self.consume()
                case "(":
                    toks.append(Token(TokenType.LPAREN, replace(self.pos)))
                    _ = self.consume()
                case ")":
                    toks.append(Token(TokenType.RPAREN, replace(self.pos)))
                    _ = self.consume()
                case "{":
                    toks.append(Token(TokenType.LBRACE, replace(self.pos)))
                    _ = self.consume()
                case "}":
                    toks.append(Token(TokenType.RBRACE, replace(self.pos)))
                    _ = self.consume()
                case ",":
                    toks.append(Token(TokenType.COMMA, replace(self.pos)))
                    _ = self.consume()
                case v if v in WHITESPACE:
                    _ = self.consume()
                case _:
                    return None, InvalidCharError(char, replace(self.pos))

        toks.append(Token(TokenType.EOF, replace(self.pos)))
        return toks, None

    def make_number(self) -> tuple[Token | None, Error | None]:
        """
        Creates a number assuming the current position is at the first digit of the number.
        This method differentiates between an integer and a floating point number, returning the corresponding TokenType (INT or FLOAT),
        depending on whether the number contains a decimal point or not.
        If the number starts with decimal point, a zero is implicitly assumed before the decimal point.
        If the number ends with a decimal point, a zero is implicitly assumed after the decimal point.
        The current position will be at the first character following the number after the method finishes.

        returns a tuple, where the first element is the token or None if an error occured and the second element is the error or None if no error occured
        """
        char: str | None
        num: str = ""
        is_float: bool = False
        pos_start = replace(self.pos)
        pos_end = replace(self.pos)

        while (char := self.peek()) is not None and char in DIGITS + ".":
            if char == ".":
                if is_float:
                    return None, InvalidCharError(char, self.pos)
                is_float = True
            pos_end = replace(self.pos)
            num += self.consume()

        if is_float:
            return Token(TokenType.FLOAT, pos_start, pos_end, float(num)), None
        else:
            return Token(TokenType.INT, pos_start, pos_end, int(num)), None

    def make_identifier(self) -> Token:
        """
        Creates either an identifier or a keyword token assuming the current position is at the first character of the identifier/token.
        The function differentiates between an identifier and an keyword depending on whether the name is in the KEYWORDS set.
        The identifier may start with an ascii letter or an underscore, every other character may additionally be a digit.
        The current position will be at the first character following the identifier after the method finishes.

        returns a tuple, where the first element is the token or None if an error occured and the second element is the error or None if no error occured
        """
        char: str | None
        val: str = ""
        pos_start = replace(self.pos)
        pos_end = replace(self.pos)

        while (char := self.peek()) is not None and char in ALLOWED_CHARS:
            pos_end = replace(self.pos)
            val += self.consume()

        if val in KEYWORDS:
            return Token(TokenType.KEYWORD, pos_start, pos_end, val)
        else:
            return Token(TokenType.IDENTIFIER, pos_start, pos_end, val)
