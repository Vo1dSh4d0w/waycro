import argparse
from typing import cast

from lang.lexer import Lexer
from lang.parser import Parser
from lang.token import Token


class Args:
    filename: str = ""
    verbose: bool = False


def main():
    parser = argparse.ArgumentParser(
        prog="waycro", description="Macro tool and language for Wayland Compositors"
    )
    _ = parser.add_argument("filename", type=str)
    _ = parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose logging"
    )

    args = parser.parse_args(namespace=Args())
    with open(args.filename, "r") as f:
        txt = f.read()
        lexer = Lexer(txt, args.filename)
        toks, err = lexer.tokenize()
        if err:
            print(err)
        else:
            parser = Parser(cast(list[Token], toks))
            res = parser.parse()
            if res.error:
                print(res.error)
            else:
                print(res.result)


if __name__ == "__main__":
    main()
