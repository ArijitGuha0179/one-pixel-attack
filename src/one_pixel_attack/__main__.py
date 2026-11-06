"""Run as python -m one_pixel_attack {evaluate,train}."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="One Pixel Attack research toolkit")
    parser.add_argument("command", choices=["evaluate", "train"])
    args, rest = parser.parse_known_args()
    if args.command == "train":
        from .train import main as command
    else:
        from .evaluate import main as command
    command(rest)


if __name__ == "__main__":
    main()
