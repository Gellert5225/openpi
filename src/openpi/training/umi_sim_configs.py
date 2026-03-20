"""
OpenPI training configs for UMI-Sim datasets.

Copy this file into the OpenPI repo at:
    src/openpi/training/umi_sim_configs.py

Then add the configs to the _CONFIGS list in src/openpi/training/config.py:
    import openpi.training.umi_sim_configs as umi_sim_config
    ...
    _CONFIGS = [
        ...
        *umi_sim_config.get_umi_sim_configs(),
    ]

After registration, run:
    uv run scripts/compute_norm_stats.py --config-name pi0_umi_sim
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_umi_sim --exp-name=my_experiment --overwrite
"""

import dataclasses
import pathlib

from typing_extensions import override

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.policies.umi_sim_policy as umi_sim_policy
import openpi.transforms as _transforms
import openpi.training.config as _config
import openpi.training.weight_loaders as weight_loaders
import openpi.training.optimizer as _optimizer

ModelType = _model.ModelType


# ─── Data config factory ───────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class LeRobotUmiSimDataConfig(_config.DataConfigFactory):
    """
    Data pipeline config for UMI-Sim LeRobot datasets.

    The LeRobot dataset produced by ``convert_zarr_to_lerobot.py`` has:
        image_overhead — (224,224,3)  dtype="image"
        image_wrist    — (224,224,3)  dtype="image"
        state          — (10,)      dtype="float32" [eef_pos(3), rot_6d(6), gripper(1)]
        actions        — (10,)      dtype="float32" (absolute EE targets, same layout)
        task           — str          language instruction

    The repack transform maps dataset keys → observation keys used during
    inference.  ``DeltaActions`` converts absolute → delta at training time;
    ``AbsoluteActions`` reverses this during inference.
    """

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
    ) -> _config.DataConfig:

        # ── Repack: dataset keys → inference-style keys ────────────────
        # Left side  = target key name (what the Inputs transform reads)
        # Right side = source key in the LeRobot dataset
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image_overhead": "image_overhead",
                        "observation/image_wrist": "image_wrist",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # ── Data transforms (applied to both train & inference) ────────
        data_transforms = _transforms.Group(
            inputs=[umi_sim_policy.UmiSimInputs(model_type=model_config.model_type)],
            outputs=[umi_sim_policy.UmiSimOutputs()],
        )

        # UMI-Sim actions are absolute EE targets → convert to delta at
        # train time.  Gripper (dim 9) stays absolute.
        #   dims 0-8: [eef_x, eef_y, eef_z, r00, r10, r20, r01, r11, r21] → delta
        #   dim  9:   gripper_width → absolute (no delta)
        delta_action_mask = _transforms.make_bool_mask(9, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # ── Model transforms (tokenisation, padding, etc.) ─────────────
        model_transforms = _config.ModelTransformFactory()(model_config)

        #print(f"[UmiSim] prompt_from_task={self.base_config.prompt_from_task}")

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


# ─── Training configs ──────────────────────────────────────────────────

def get_umi_sim_configs() -> list[_config.TrainConfig]:
    """Return all UMI-Sim training configs to be registered in _CONFIGS."""
    return [
        # ── π₀ full fine-tune ──────────────────────────────────────────
        _config.TrainConfig(
            name="pi0_umi_sim",
            model=pi0_config.Pi0Config(),
            data=LeRobotUmiSimDataConfig(
                # ▸ Replace with your HF repo id
                repo_id="gellert5225/umi_sim_sort",
                base_config=_config.DataConfig(
                    prompt_from_task=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_base/params"
            ),
            num_train_steps=30_000,
        ),

        # ── π₀ LoRA fine-tune (low-memory, fits on RTX 4090) ──────────
        _config.TrainConfig(
            name="pi0_umi_sim_lora",
            model=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=LeRobotUmiSimDataConfig(
                repo_id="gellert5225/umi_sim_green_cube_v6",
                base_config=_config.DataConfig(
                    prompt_from_task=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_base/params"
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=200,
                decay_steps=15_000,
                peak_lr=2.5e-5,
                decay_lr=2.5e-6,
            ),
            num_train_steps=15_000,
            batch_size=8,
            freeze_filter=pi0_config.Pi0Config(
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),

        # ── π₀-FAST full fine-tune ────────────────────────────────────
        _config.TrainConfig(
            name="pi0_fast_umi_sim",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=10,
                action_horizon=10,
                max_token_len=180,
            ),
            data=LeRobotUmiSimDataConfig(
                repo_id="gellert5225/umi_sim_sort_v2",
                base_config=_config.DataConfig(
                    prompt_from_task=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_fast_base/params"
            ),
            num_train_steps=30_000,
        ),

        # ── π₀-FAST LoRA fine-tune ────────────────────────────────────
        _config.TrainConfig(
            name="pi0_fast_umi_sim_lora",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=10,
                action_horizon=10,
                max_token_len=180,
                paligemma_variant="gemma_2b_lora",
            ),
            data=LeRobotUmiSimDataConfig(
                repo_id="gellert5225/umi_sim_green_cube_v6",
                base_config=_config.DataConfig(
                    prompt_from_task=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi0_fast_base/params"
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=200,
                decay_steps=15_000,
                peak_lr=2.5e-5,
                decay_lr=2.5e-6,
            ),
            num_train_steps=15_000,
            batch_size=8,
            freeze_filter=pi0_fast.Pi0FASTConfig(
                action_dim=10,
                action_horizon=10,
                max_token_len=180,
                paligemma_variant="gemma_2b_lora",
            ).get_freeze_filter(),
            ema_decay=None,
        ),

        # ── π₀.₅ full fine-tune ───────────────────────────────────────
        _config.TrainConfig(
            name="pi05_umi_sim",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=10,
            ),
            data=LeRobotUmiSimDataConfig(
                repo_id="gellert5225/umi_sim_sort_v2",
                base_config=_config.DataConfig(
                    prompt_from_task=True,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params"
            ),
            num_train_steps=30_000,
        ),
    ]
