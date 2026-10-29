from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .common import load_config, read_json, root_path, tokenizer_for


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tagore Bengali continued-pretraining project (explicit actions only)")
    parser.add_argument("--root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("environment")
    sub.add_parser("collect")
    manual = sub.add_parser("import-manual")
    manual.add_argument("manifest", type=Path)
    sub.add_parser("prepare")
    sub.add_parser("audit")
    training = sub.add_parser("train")
    training.add_argument("--training-config", type=Path)
    training.add_argument("--stop-after", type=int)
    training.add_argument("--no-export", action="store_true")
    ck = sub.add_parser("checkpoints")
    ck.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = args.root.resolve() if args.root else root_path()
    config = load_config(root)
    if args.command == "environment":
        from .modeling import environment_report
        result = environment_report(root)
    elif args.command == "collect":
        from .collection import collect
        result = collect(root, config)
    elif args.command == "import-manual":
        from .collection import import_manual
        result = import_manual(root, args.manifest)
    elif args.command in {"prepare", "audit"}:
        from .corpus import prepare, validate_dataset
        result = (prepare if args.command == "prepare" else validate_dataset)(root, config, tokenizer_for(config))
    elif args.command == "checkpoints":
        from .checkpoints import checkpoint_inventory
        run_dir = args.run_dir.resolve()
        result = checkpoint_inventory(run_dir, read_json(run_dir / "RUN.json"))
    else:
        from .training import train
        tc = read_json(args.training_config) if args.training_config else load_config(root, "training")
        result = train(root, config, tc, stop_after=args.stop_after, export=not args.no_export)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
