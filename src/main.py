import argparse

from lang.lexer import Lexer


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
            print(toks)


if __name__ == "__main__":
    main()
