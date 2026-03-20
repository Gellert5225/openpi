"""
OpenPI data config for UMI-Sim → π₀ / π₀-FAST / π₀.₅ fine-tuning.

Copy this file into the OpenPI repo at:
    src/openpi/policies/umi_sim_policy.py

Then register the training configs (see umi_sim_configs.py).

Dataset layout (LeRobot v2, created by convert_zarr_to_lerobot.py):
    image_overhead — (224, 224, 3) uint8   overhead / third-person camera
    image_wrist    — (224, 224, 3) uint8   wrist-mounted camera
    state          — (10,) float32         [eef_pos(3), rot_6d(6), gripper_width(1)]
    actions        — (10,) float32         absolute EE target (same 10-dim as state)
    task           — str                  language instruction / task label
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# ─── Helpers ────────────────────────────────────────────────────────────

def make_umi_sim_example() -> dict:
    """Creates a random input example for smoke-testing the UMI-Sim policy."""
    return {
        "observation/image_overhead": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/image_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(10).astype(np.float32),
        "prompt": "pick up the red object",
    }


def _parse_image(image) -> np.ndarray:
    """Ensure image is uint8 (H, W, C)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# ─── Inputs transform ──────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class UmiSimInputs(transforms.DataTransformFn):
    """
    Map UMI-Sim observations → π₀ model input format.

    π₀ supports three image slots:
        base_0_rgb          — third-person / overhead view
        left_wrist_0_rgb    — left wrist camera
        right_wrist_0_rgb   — right wrist camera

    UMI-Sim has two cameras:
        image_overhead → base_0_rgb
        image_wrist    → left_wrist_0_rgb
    right_wrist_0_rgb is zero-filled.
    """

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # State: [eef_pos(3), rot_6d(6), gripper_width(1)] = 10-dim
        state = np.asarray(data["observation/state"], dtype=np.float32)

        # Overhead camera → base slot, wrist camera → left_wrist slot
        overhead_image = _parse_image(data["observation/image_overhead"])
        wrist_image = _parse_image(data["observation/image_wrist"])
        zero_image = np.zeros_like(overhead_image)

        inputs: dict = {
            "state": state,
            "image": {
                "base_0_rgb": overhead_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": zero_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # For PI0_FAST all slots must be True; for PI0/PI05 unused slots are False.
                "right_wrist_0_rgb": (
                    np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
                ),
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        else:
            print("Warning: no prompt found in data; using empty string!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        return inputs


# ─── Outputs transform ─────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class UmiSimOutputs(transforms.DataTransformFn):
    """Extract the first 10 action dims (the rest is padding)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :10])}
