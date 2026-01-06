from dataclasses import replace
from typing import override

from lang.position import Position


def generate_string_with_arrows(pos_start: Position, pos_end: Position | None = None):
    """
    Gennerates a string containing the relevant lines between pos_start and pos_end.
    Every character between pos_start and pos_end will have arrows ('^') underneath.

    pos_start
        the starting position

    pos_end
        the ending position, defaults to pos_start

    returns the string with the arrows as described above
    """
    if pos_end is None:
        pos_end = replace(pos_start)

    lines = list(pos_start.ftxt.split("\n")[pos_start.ln - 1 : pos_end.ln])
    txt = ""

    if pos_start.ln == pos_end.ln:
        txt += lines[0] + "\n"
        txt += " " * (pos_start.col - 1) + "^" * (pos_end.col - pos_start.col + 1)
    else:
        txt += lines[0] + "\n"
        txt += (
            " " * (pos_start.col - 1) + "^" * (len(lines[0]) - pos_start.col + 1) + "\n"
        )
        for line in lines[1:-1]:
            txt += line + "\n"
            txt += "^" * (len(line)) + "\n"
        txt += lines[-1] + "\n"
        txt += "^" * pos_end.col

    return txt


def add_indent(txt: str, n: int = 1) -> str:
    """
    Adds n whitespaces before each line of txt.

    txt
        the text to indent

    n
        number of whitespaces to add

    returns the indented text
    """
    return "\n".join(map(lambda ln: " " * n + ln, txt.split("\n")))


class Error:
    """
    Base class for every error, should not be used directly.
    """

    def __init__(self, name: str, msg: str) -> None:
        """
        name
            the name of the error

        msg
            the error message
        """
        self.name: str = name
        self.msg: str = msg

    @override
    def __repr__(self) -> str:
        return f"{self.name}: {self.msg}"


class PositionalError(Error):
    """
    Base class for errors which contain information about the position of the error, should not be used directly.
    """

    def __init__(
        self, name: str, msg: str, pos_start: Position, pos_end: Position | None = None
    ) -> None:
        """
        name
            the name of the error

        msg
            the error message

        pos_start
            the starting position of the error

        pos_end
            the ending position of the error, defaults to the starting position
        """
        super().__init__(name, msg)

        self.pos_start: Position = pos_start
        self.pos_end: Position = pos_end or replace(pos_start)

    @override
    def __repr__(self) -> str:
        return f"{self.name}: {self.msg} at {self.pos_start.fn}, line {self.pos_start.ln}, col {self.pos_start.col}\n{add_indent(generate_string_with_arrows(self.pos_start, self.pos_end))}"


class InvalidCharError(PositionalError):
    """
    Error used by the lexer when it detects an invalid character.
    """

    def __init__(self, char: str, pos: Position) -> None:
        """
        char
            the invalid character

        pos
            the position of the invalid character
        """
        super().__init__("InvalidCharError", char, pos)


class SyntaxError(PositionalError):
    def __init__(
        self, message: str, pos_start: Position, pos_end: Position | None = None
    ) -> None:
        super().__init__(
            "SyntaxError", message, pos_start, pos_end or replace(pos_start)
        )
