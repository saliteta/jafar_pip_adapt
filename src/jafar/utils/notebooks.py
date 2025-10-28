# utils/notebook.py
from __future__ import annotations
import os
import torch
from hydra import initialize, compose
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_module
from IPython.display import clear_output
from ..hydra_plugins.resolvers import register_resolvers

def load_model(backbone: str, project_root: str, model_path: str) -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Assumes your configs live under the installed package at: jafar/configs/
    e.g. src/jafar/configs/base.yaml, etc.
    """
    # --- Build Hydra overrides ---
    overrides = ["val_dataloader.batch_size=1", f"project_root={project_root}"]
    if "radio" in backbone:
        overrides += ["backbone=radio"]
    overrides += [f"backbone.name={backbone}"]

    # --- Always (re)initialize Hydra against the packaged configs dir ---
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    register_resolvers()

    # Locate installed config directory: jafar/configs
    with initialize_config_module(config_module="jafar.config", version_base=None):
        cfg = compose(config_name="base", overrides=overrides)

    # --- Instantiate modules ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone_obj = instantiate(cfg.backbone).to(device)
    model = instantiate(cfg.model).to(device).eval()

    # --- Load checkpoint (handles both nested and flat state dicts) ---
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("jafar", ckpt)
    model.load_state_dict(state, strict=True)

    try: clear_output(wait=True)
    except Exception: pass

    return model, backbone_obj

__all__ = ["load_model"]
