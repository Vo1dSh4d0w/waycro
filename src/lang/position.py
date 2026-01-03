from dataclasses import dataclass


@dataclass
class Position:
    """
    Data class representing a position in text, usually a file.

    fn
        the file name

    ftxt
        the file content

    idx
        the index in ftxt

    ln
        the line

    col
        the column
    """

    fn: str
    ftxt: str
    idx: int
    ln: int
    col: int
