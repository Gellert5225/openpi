#!/usr/bin/env python
"""Push an existing local LeRobot dataset to HuggingFace Hub.

Prerequisites
─────────────
1. Activate the venv:  .\.umi-env\Scripts\activate
2. Log in once:        python -m huggingface_hub.commands.huggingface_cli login

Usage
─────
  python scripts/push_to_hub.py --repo myuser/umi_sim_pick
  python scripts/push_to_hub.py --repo myuser/umi_sim_pick --private --tags ur5e openpi
"""

import argparse
import sys

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(
        description="Push a local LeRobot dataset to HuggingFace Hub."
    )
    parser.add_argument(
        "--repo", required=True, type=str,
        help="HuggingFace repo id (e.g. 'username/dataset_name'). "
             "Must match the repo_id used during conversion.",
    )
    parser.add_argument(
        "--tags", nargs="*", default=["ur5e", "umi-sim", "openpi"],
        help="Tags to attach to the dataset card (default: ur5e umi-sim openpi).",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Upload as a private dataset (default: public).",
    )
    parser.add_argument(
        "--license", type=str, default="apache-2.0",
        help="License identifier for the dataset card (default: apache-2.0).",
    )
    args = parser.parse_args()

    print(f"Loading local dataset  '{args.repo}' …")
    try:
        dataset = LeRobotDataset(args.repo)
    except Exception as e:
        print(f"Error: could not load dataset '{args.repo}': {e}", file=sys.stderr)
        print(
            "\nMake sure the dataset exists locally at "
            "~/.cache/huggingface/lerobot/<repo_id>/\n"
            "Run convert_zarr_to_lerobot.py first if needed.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Episodes : {dataset.num_episodes}")
    print(f"  Frames   : {dataset.num_frames}")
    print(f"  Features : {list(dataset.features.keys())}")
    print()

    print(f"Pushing to HuggingFace Hub as '{'private' if args.private else 'public'}' …")
    dataset.push_to_hub(
        tags=args.tags,
        private=args.private,
        license=args.license,
    )
    print(f"Done — https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
