"""
Convert UMI-Sim Zarr recordings to LeRobot dataset format for OpenPI fine-tuning.

UMI-Sim records demonstrations as .zarr.zip files with:
  data/
    camera0_rgb              (N, H, W, 3)  uint8   overhead camera
    camera1_rgb              (N, H, W, 3)  uint8   wrist camera (optional)
    camera0_depth            (N, H, W)     float32 metres  (optional)
    camera1_depth            (N, H, W)     float32 metres  (optional)
    robot0_eef_pos           (N, 3)        float32
    robot0_eef_rot_6d        (N, 6)        float32 6D rotation (first 2 cols of rotation matrix)
    robot0_gripper_width     (N, 1)        float32
    action                   (N, 10)       float32 [pos(3)+rot_6d(6)+grip(1)]  absolute
  meta/
    episode_ends             (E,)          int64

OpenPI expects a LeRobot v2 dataset whose "actions" are **absolute** EE
targets.  The conversion from absolute→delta is handled by OpenPI's
``DeltaActions`` transform at training time, so we do NOT compute
deltas here.

Usage
-----
  # IMPORTANT: Run from the OpenPI repo so you use their pinned LeRobot (v2.1).
  # This ensures the dataset codebase_version matches what OpenPI expects.
  cd /path/to/openpi
  uv run python scripts/convert_zarr_to_lerobot.py \
      --zarr  /mnt/i/umi-sim/data/vr_demonstrations/dataset_session.zarr.zip \
      --repo  myuser/umi_sim_pick \
      --fps   10

  # Or from a standalone env with 'pip install lerobot' (produces v3.0 — NOT
  # compatible with OpenPI without reconversion):
  python scripts/convert_zarr_to_lerobot.py \
      --zarr  data/vr_demonstrations/dataset_session.zarr.zip \
      --repo  myuser/umi_sim_pick

Dependencies
------------
  pip install lerobot zarr numcodecs
  # (lerobot is installed as part of the openpi environment)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import zarr
except ImportError:
    sys.exit("zarr is required:  pip install zarr")

# Try new import path (LeRobot ≥ v3.0) then fall back to old path (v2.1, used by OpenPI).
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # v3.0+
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
except ImportError:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # v2.1
        from lerobot.common.constants import HF_LEROBOT_HOME
    except ImportError:
        sys.exit(
            "lerobot is required.  Install it from the openpi environment:\n"
            "  uv pip install lerobot   OR   pip install lerobot"
        )


# ── Zarr reader (self-contained, no Isaac Sim dependency) ──────────────

class _ZarrReader:
    """Minimal reader for UMI-Sim .zarr.zip files."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(path)

        self._store = zarr.storage.ZipStore(str(self.path), mode="r")
        self._root = zarr.open_group(store=self._store, mode="r")

        self.episode_ends: np.ndarray = self._root["meta/episode_ends"][:]
        self.num_episodes: int = len(self.episode_ends)
        self.total_frames: int = int(self.episode_ends[-1]) if self.num_episodes else 0

        # Detect available arrays
        self._has_wrist = "data/camera1_rgb" in self._root
        self._has_depth = "data/camera0_depth" in self._root
        self._has_action = "data/action" in self._root
        self._has_rot_6d = "data/robot0_eef_rot_6d" in self._root

        # Figure out the task name stored in the zarr (if any).
        # ZarrDatasetWriter doesn't persist per-episode task names in a
        # dedicated array, so we fall back to a CLI flag.
        self._task_name: Optional[str] = None

        print(f"Opened {self.path.name}  "
              f"({self.num_episodes} episodes, {self.total_frames} frames, "
              f"wrist_cam={'yes' if self._has_wrist else 'no'}, "
              f"depth={'yes' if self._has_depth else 'no'})")

    # ── episode helpers ─────────────────────────────────────────────────

    def episode_range(self, ep_idx: int) -> tuple[int, int]:
        start = 0 if ep_idx == 0 else int(self.episode_ends[ep_idx - 1])
        end = int(self.episode_ends[ep_idx])
        return start, end

    def episode_length(self, ep_idx: int) -> int:
        s, e = self.episode_range(ep_idx)
        return e - s

    # ── frame access (single frame  →  dict) ───────────────────────────

    def get_frame(self, idx: int) -> dict:
        d = self._root["data"]
        if self._has_rot_6d:
            eef_rot = d["robot0_eef_rot_6d"][idx]                    # (6,)
        else:
            eef_rot = d["robot0_eef_rot_axis_angle"][idx]            # (3,) legacy
        frame: dict = {
            "image_overhead": d["camera0_rgb"][idx],                  # (H,W,3) uint8
            "eef_pos": d["robot0_eef_pos"][idx],                     # (3,)
            "eef_rot": eef_rot,
            "gripper": d["robot0_gripper_width"][idx],               # (1,)
        }
        if self._has_wrist:
            frame["image_wrist"] = d["camera1_rgb"][idx]              # (H,W,3) uint8
        if self._has_action:
            frame["action"] = d["action"][idx]                        # (10,) or (7,) legacy
        if self._has_depth:
            frame["depth"] = d["camera0_depth"][idx]                  # (H,W)
        return frame

    def close(self):
        if self._store:
            self._store.close()


# ── Conversion ─────────────────────────────────────────────────────────

def convert(
    zarr_path: str | Path,
    repo_id: str,
    fps: int = 10,
    task: str = "manipulation",
    robot_type: str = "xarm6",
    overwrite: bool = False,
    push_to_hub: bool = False,
    image_size: tuple[int, int] | None = None,
):
    """Convert a single .zarr.zip file into a LeRobot v2 dataset.

    Parameters
    ----------
    zarr_path : path to UMI-Sim .zarr.zip
    repo_id   : HuggingFace-style repo id  (e.g. ``"user/dataset"``)
    fps       : recording FPS (must match the zarr's actual rate)
    task      : default task/language instruction for every episode
    robot_type: LeRobot robot_type tag
    overwrite : delete existing local dataset before writing
    push_to_hub: push final dataset to HuggingFace Hub
    image_size: (H, W) to validate / override; auto-detected if None
    """
    reader = _ZarrReader(zarr_path)

    # Detect image resolution from first frame
    sample = reader.get_frame(0)
    h, w = sample["image_overhead"].shape[:2]
    if image_size is not None:
        h, w = image_size
    has_wrist = "image_wrist" in sample
    print(f"Image resolution: {h}×{w}, wrist camera: {'yes' if has_wrist else 'no'}")

    # Detect rotation format from zarr
    is_6d = reader._has_rot_6d
    rot_dim = 6 if is_6d else 3
    print(f"Rotation format: {'6D (continuous)' if is_6d else 'axis-angle (3D, legacy)'}")

    # State dim = eef_pos(3) + rot(6) + gripper(1) = 10
    state_dim = 10
    # Action dim = same 10 (absolute EE target)
    action_dim = 10

    # ── Create LeRobot dataset ──────────────────────────────────────────
    output_path = HF_LEROBOT_HOME / repo_id
    if overwrite and output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    features = {
        "image_overhead": {
            "dtype": "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["actions"],
        },
    }
    if has_wrist:
        features["image_wrist"] = {
            "dtype": "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=4,
        image_writer_processes=2,
    )

    # ── Write episodes ──────────────────────────────────────────────────
    total_written = 0
    for ep_idx in range(reader.num_episodes):
        start, end = reader.episode_range(ep_idx)
        ep_len = end - start

        for i in range(start, end):
            frame = reader.get_frame(i)

            rot = frame["eef_rot"].astype(np.float32)

            # If legacy 3D axis-angle, convert to 6D rotation
            if rot.shape[0] == 3:
                from scipy.spatial.transform import Rotation
                mat = Rotation.from_rotvec(rot).as_matrix()
                rot = mat[:, :2].T.flatten().astype(np.float32)  # (6,)

            # State: [eef_pos(3), rot_6d(6), gripper_width(1)] = 10D
            state = np.concatenate([
                frame["eef_pos"].astype(np.float32),
                rot,
                frame["gripper"].astype(np.float32).flatten()[:1],
            ])

            # Action: absolute EE target (10D)
            if "action" in frame:
                action = frame["action"].astype(np.float32)
                # If legacy 7D action, convert rotation part to 6D
                if action.shape[0] == 7:
                    from scipy.spatial.transform import Rotation as R2
                    a_pos = action[:3]
                    a_rot = action[3:6]
                    a_grip = action[6:7]
                    a_mat = R2.from_rotvec(a_rot).as_matrix()
                    a_rot_6d = a_mat[:, :2].T.flatten().astype(np.float32)
                    action = np.concatenate([a_pos, a_rot_6d, a_grip])
            else:
                # Fallback: use current state as action
                action = state.copy()

            lerobot_frame = {
                "image_overhead": frame["image_overhead"],  # (H, W, 3) uint8
                "state": state,                             # (10,) float32
                "actions": action,                          # (10,) float32 – absolute
                "task": task,
            }
            if has_wrist:
                lerobot_frame["image_wrist"] = frame["image_wrist"]

            dataset.add_frame(lerobot_frame)

        dataset.save_episode()
        total_written += ep_len
        print(f"  Episode {ep_idx + 1}/{reader.num_episodes}  "
              f"({ep_len} frames, total {total_written})")

    # Flush background image-writer threads.
    # v3.0 has finalize(); v2.1 only has stop_image_writer().
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    else:
        dataset.stop_image_writer()

    reader.close()

    print(f"\nDone — {reader.num_episodes} episodes, "
          f"{total_written} frames written to {output_path}")

    # ── Optional Hub push ───────────────────────────────────────────────
    if push_to_hub:
        print("Pushing to HuggingFace Hub …")
        dataset.push_to_hub(
            tags=[robot_type, "umi-sim", "openpi"],
            private=False,
            license="apache-2.0",
        )
        print("Push complete.")


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert UMI-Sim .zarr.zip → LeRobot dataset for OpenPI fine-tuning."
    )
    parser.add_argument(
        "--zarr", required=True, type=str,
        help="Path to the .zarr.zip file (or directory containing multiple).",
    )
    parser.add_argument(
        "--repo", required=True, type=str,
        help="LeRobot / HuggingFace repo id, e.g. 'myuser/umi_sim_pick'.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Recording FPS (default: 30).")
    parser.add_argument("--task", type=str, default="manipulation",
                        help="Language instruction / task name for all episodes.")
    parser.add_argument("--robot-type", type=str, default="xarm6",
                        help="Robot type tag (default: xarm6).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing local dataset.")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push finished dataset to HuggingFace Hub.")

    args = parser.parse_args()

    zarr_path = Path(args.zarr)

    if zarr_path.is_dir():
        # If a directory is given, process all .zarr.zip files in it
        zarr_files = sorted(zarr_path.glob("*.zarr.zip"))
        if not zarr_files:
            sys.exit(f"No .zarr.zip files found in {zarr_path}")
        print(f"Found {len(zarr_files)} zarr file(s) in {zarr_path}")
        for zf in zarr_files:
            convert(
                zarr_path=zf,
                repo_id=args.repo,
                fps=args.fps,
                task=args.task,
                robot_type=args.robot_type,
                overwrite=args.overwrite,
                push_to_hub=False,  # push once at the end
            )
            args.overwrite = False  # only overwrite on first file
        if args.push_to_hub:
            # Re-open and push the combined dataset
            ds = LeRobotDataset(args.repo)
            ds.push_to_hub(
                tags=[args.robot_type, "umi-sim", "openpi"],
                private=False,
                license="apache-2.0",
            )
    else:
        convert(
            zarr_path=zarr_path,
            repo_id=args.repo,
            fps=args.fps,
            task=args.task,
            robot_type=args.robot_type,
            overwrite=args.overwrite,
            push_to_hub=args.push_to_hub,
        )


if __name__ == "__main__":
    main()
