"""
Concatenate two LeRobot datasets into a new dataset.

Usage
-----
  # Concatenate two datasets, keeping original task strings:
  python scripts/concat_lerobot_datasets.py \
      --dataset1 user/dataset_a \
      --dataset2 user/dataset_b \
      --output   user/merged_dataset

  # Concatenate with a new task string for the combined dataset:
  python scripts/concat_lerobot_datasets.py \
      --dataset1 user/dataset_a \
      --dataset2 user/dataset_b \
      --output   user/merged_dataset \
      --task     "pick up the cup and place it on the shelf"

  # Use local paths (relative to HF_LEROBOT_HOME):
  python scripts/concat_lerobot_datasets.py \
      --dataset1 local/pick_dataset \
      --dataset2 local/place_dataset \
      --output   local/combined

Dependencies
------------
  pip install lerobot pillow numpy
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PIL import Image
except ImportError:
    sys.exit("Pillow is required:  pip install pillow")

# LeRobot — try both import paths (v3.0+ and v2.1/OpenPI).
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
except ImportError:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.constants import HF_LEROBOT_HOME
    except ImportError:
        sys.exit(
            "lerobot is required.\n"
            "  uv pip install lerobot   OR   pip install lerobot"
        )


def features_compatible(f1: dict, f2: dict) -> tuple[bool, str]:
    """Check if two feature dicts are compatible for concatenation."""
    keys1 = set(f1.keys())
    keys2 = set(f2.keys())
    
    if keys1 != keys2:
        missing = keys1.symmetric_difference(keys2)
        return False, f"Feature keys differ: {missing}"
    
    for key in keys1:
        if f1[key].get("dtype") != f2[key].get("dtype"):
            return False, f"dtype mismatch for '{key}': {f1[key].get('dtype')} vs {f2[key].get('dtype')}"
        
        shape1 = tuple(f1[key].get("shape", []))
        shape2 = tuple(f2[key].get("shape", []))
        if shape1 != shape2:
            return False, f"shape mismatch for '{key}': {shape1} vs {shape2}"
    
    return True, ""


def get_episode_indices(dataset: LeRobotDataset, ep_idx: int) -> tuple[int, int]:
    """Get the start and end frame indices for an episode."""
    # LeRobot stores episode_data_index with 'from' and 'to' keys
    if hasattr(dataset, 'episode_data_index'):
        from_idx = int(dataset.episode_data_index["from"][ep_idx])
        to_idx = int(dataset.episode_data_index["to"][ep_idx])
        return from_idx, to_idx
    else:
        # Fallback for older API
        ep_starts = [0]
        for i in range(dataset.num_episodes - 1):
            # Count frames until episode changes
            pass
        # This is a simplified fallback - real implementation would need
        # to inspect the dataset structure
        raise NotImplementedError("Could not determine episode boundaries")


def copy_dataset_frames(
    src_dataset: LeRobotDataset,
    dst_dataset: LeRobotDataset,
    task_override: Optional[str] = None,
    dataset_name: str = "dataset",
) -> int:
    """Copy all frames from source to destination dataset.
    
    Returns the number of frames copied.
    """
    # Keys added by LeRobot that should not be copied back
    SKIP_KEYS = {"index", "task_index", "episode_index", "frame_index", "timestamp"}
    
    total_frames = 0
    
    for ep_idx in range(src_dataset.num_episodes):
        start_idx, end_idx = get_episode_indices(src_dataset, ep_idx)
        ep_len = end_idx - start_idx
        
        for frame_idx in range(start_idx, end_idx):
            # Get frame from source dataset
            src_frame = src_dataset[frame_idx]
            
            # Build output frame dict - only include features expected by destination
            out_frame = {}
            for key in dst_dataset.features.keys():
                if key in SKIP_KEYS:
                    continue
                if key not in src_frame:
                    continue
                    
                val = src_frame[key]
                feat_info = dst_dataset.features[key]
                
                # Convert tensors to numpy if needed
                if hasattr(val, 'numpy'):
                    val = val.numpy()
                # Handle PIL images
                if isinstance(val, Image.Image):
                    val = np.array(val)
                
                # Handle image features: LeRobot returns CHW, but add_frame expects HWC
                if feat_info.get("dtype") == "image":
                    expected_shape = tuple(feat_info.get("shape", []))
                    if len(val.shape) == 3 and len(expected_shape) == 3:
                        # Check if we have CHW format (C, H, W) but expect HWC (H, W, C)
                        if val.shape[0] in (1, 3, 4) and expected_shape[-1] in (1, 3, 4):
                            if val.shape[0] == expected_shape[-1] and val.shape[0] != val.shape[-1]:
                                # Transpose from CHW to HWC
                                val = np.transpose(val, (1, 2, 0))
                
                out_frame[key] = val
            
            # Set task string
            if task_override is not None:
                out_frame["task"] = task_override
            elif "task" in src_frame:
                # Keep original task
                task_val = src_frame["task"]
                if hasattr(task_val, 'item'):
                    task_val = task_val.item()
                out_frame["task"] = task_val
            else:
                out_frame["task"] = "manipulation"
            
            dst_dataset.add_frame(out_frame)
        
        dst_dataset.save_episode()
        total_frames += ep_len
        print(f"  {dataset_name} episode {ep_idx + 1}/{src_dataset.num_episodes} "
              f"({ep_len} frames, total {total_frames})")
    
    return total_frames


def concat_datasets(
    dataset1_repo: str,
    dataset2_repo: str,
    output_repo: str,
    task: Optional[str] = None,
    overwrite: bool = False,
    push_to_hub: bool = False,
    robot_type: Optional[str] = None,
):
    """Concatenate two LeRobot datasets into a new one.
    
    Parameters
    ----------
    dataset1_repo : First dataset repo_id
    dataset2_repo : Second dataset repo_id  
    output_repo   : Output dataset repo_id
    task          : If specified, all episodes get this task string.
                    If None, original task strings are preserved.
    overwrite     : Delete existing output dataset before writing
    push_to_hub   : Push final dataset to HuggingFace Hub
    robot_type    : Override robot_type (default: use from dataset1)
    """
    print(f"Loading dataset 1: {dataset1_repo}")
    ds1 = LeRobotDataset(dataset1_repo)
    print(f"  Episodes: {ds1.num_episodes}, Frames: {ds1.num_frames}")
    print(f"  Features: {list(ds1.features.keys())}")
    
    print(f"\nLoading dataset 2: {dataset2_repo}")
    ds2 = LeRobotDataset(dataset2_repo)
    print(f"  Episodes: {ds2.num_episodes}, Frames: {ds2.num_frames}")
    print(f"  Features: {list(ds2.features.keys())}")
    
    # Check compatibility
    compatible, msg = features_compatible(ds1.features, ds2.features)
    if not compatible:
        sys.exit(f"\nError: Datasets are not compatible.\n{msg}")
    
    # Check FPS match
    fps1 = ds1.fps
    fps2 = ds2.fps
    if fps1 != fps2:
        print(f"\nWarning: FPS mismatch ({fps1} vs {fps2}). Using {fps1} from dataset1.")
    
    # Use robot_type from first dataset or override
    if robot_type is None:
        robot_type = getattr(ds1, 'robot_type', 'unknown')
    
    print(f"\nCreating output dataset: {output_repo}")
    print(f"  Robot type: {robot_type}")
    print(f"  FPS: {fps1}")
    if task:
        print(f"  Task override: '{task}'")
    else:
        print(f"  Task: preserving original task strings")
    
    # Prepare output path
    output_path = HF_LEROBOT_HOME / output_repo
    if overwrite and output_path.exists():
        print(f"  Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    # Create new dataset with same features
    out_dataset = LeRobotDataset.create(
        repo_id=output_repo,
        robot_type=robot_type,
        fps=fps1,
        features=dict(ds1.features),
        image_writer_threads=4,
        image_writer_processes=2,
    )
    
    # Copy frames from both datasets
    print(f"\nCopying from dataset 1 ({ds1.num_episodes} episodes)...")
    frames1 = copy_dataset_frames(ds1, out_dataset, task_override=task, dataset_name="DS1")
    
    print(f"\nCopying from dataset 2 ({ds2.num_episodes} episodes)...")
    frames2 = copy_dataset_frames(ds2, out_dataset, task_override=task, dataset_name="DS2")
    
    # Finalize
    if hasattr(out_dataset, "finalize"):
        out_dataset.finalize()
    else:
        out_dataset.stop_image_writer()
    
    total_episodes = ds1.num_episodes + ds2.num_episodes
    total_frames = frames1 + frames2
    
    print(f"\n{'='*60}")
    print(f"Done — concatenated {total_episodes} episodes, {total_frames} frames")
    print(f"Output: {output_path}")
    
    # Optional push
    if push_to_hub:
        print("\nPushing to HuggingFace Hub...")
        out_dataset.push_to_hub(
            tags=[robot_type, "umi-sim", "concatenated"],
            private=False,
            license="apache-2.0",
        )
        print(f"Push complete: https://huggingface.co/datasets/{output_repo}")


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate two LeRobot datasets into a new dataset."
    )
    parser.add_argument(
        "--dataset1", required=True, type=str,
        help="First dataset repo_id (e.g. 'user/dataset_a').",
    )
    parser.add_argument(
        "--dataset2", required=True, type=str,
        help="Second dataset repo_id (e.g. 'user/dataset_b').",
    )
    parser.add_argument(
        "--output", required=True, type=str,
        help="Output dataset repo_id (e.g. 'user/merged_dataset').",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Task string for the combined dataset. "
             "If not specified, original task strings are preserved.",
    )
    parser.add_argument(
        "--robot-type", type=str, default=None,
        help="Override robot type (default: use from dataset1).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output dataset.",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push the concatenated dataset to HuggingFace Hub.",
    )
    
    args = parser.parse_args()
    
    concat_datasets(
        dataset1_repo=args.dataset1,
        dataset2_repo=args.dataset2,
        output_repo=args.output,
        task=args.task,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
        robot_type=args.robot_type,
    )


if __name__ == "__main__":
    main()
