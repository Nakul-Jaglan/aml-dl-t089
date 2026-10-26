"""Explicit commands keep final testing separate from development."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "freeze", "test", "figures"])
    parser.add_argument("--profile", choices=["smoke", "budget", "full"], default="budget")
    parser.add_argument("--search", choices=["baseline", "compact"], default="compact")
    parser.add_argument("--selection-from", choices=["budget"])
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare
        prepare()
    elif args.command == "train":
        from .experiments import train
        train(args.profile, args.search, args.selection_from)
    elif args.command == "freeze":
        from .experiments import freeze
        freeze(args.profile)
    elif args.command == "test":
        from .experiments import final_test
        final_test(args.profile, args.allow_test)
    else:
        from .plots import generate_figures
        generate_figures(args.profile)

