"""Phase 2 commands. Development commands never load official test data.

  python -m clothing_ml.phase2 train  --config configs/phase2/smoke.json
  python -m clothing_ml.phase2 verify --profile smoke --run-id <run_id>
  python -m clothing_ml.phase2 select --profile full            # validation-only CNN choice
  python -m clothing_ml.phase2 fusion --profile full            # optional hybrid + LR controls
  python -m clothing_ml.phase2 curves --profile full            # learning-curve figures
  python -m clothing_ml.phase2 freeze --profile full            # nominate rows for test
  python -m clothing_ml.phase2 test   --profile full --allow-test
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "verify", "select", "fusion", "curves", "freeze", "test"])
    parser.add_argument("--config", nargs="+", help="one or more JSON configs (train)")
    parser.add_argument("--profile", choices=["smoke", "full"], default="full")
    parser.add_argument("--run-id")
    parser.add_argument("--device", help="override config device (cpu, mps, cuda)")
    parser.add_argument("--seed", type=int, help="override config seed (for predeclared seed repeats)")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()

    if args.command == "train":
        from ..data import load_official
        from .runner import load_config, train_run
        if not args.config:
            parser.error("train needs --config")
        images, labels = load_official("train")
        for path in args.config:
            train_run(load_config(path, device=args.device, seed=args.seed), images, labels)
    elif args.command == "verify":
        from .runner import verify_run
        if not args.run_id:
            parser.error("verify needs --run-id")
        verify_run(args.profile, args.run_id, device=args.device or "cpu")
    elif args.command == "select":
        from .select import select_cnn
        select_cnn(args.profile)
    elif args.command == "fusion":
        from .fusion import run_fusion
        run_fusion(args.profile)
    elif args.command == "curves":
        from .plots import learning_curves
        learning_curves(args.profile)
    elif args.command == "freeze":
        from .select import freeze
        freeze(args.profile)
    else:
        from .select import final_test
        final_test(args.profile, args.allow_test)


if __name__ == "__main__":
    main()
