from dataclasses import dataclass


@dataclass
class Position:
    fn: str
    ftxt: str
    idx: int
    ln: int
    col: int
