from omegaconf import OmegaConf

def _get_feature(target: str) -> int:
    target = target.lower()
    if "vit_small" in target or "vits" in target: return 384
    if "vit_base"  in target or "vitb" in target: return 768
    if "vit_large" in target or "vitl" in target: return 1024
    if target == "efficientnet_b4": return 128
    if target == "maskclip":        return 512
    if target == "radio_v2.5-h":    return 1280
    if target == "radio_v2.5-l":    return 1024
    if target == "radio_v2.5-b":    return 768
    raise ValueError(f"Unsupported backbone: {target}")

def register_resolvers():
    OmegaConf.register_new_resolver("get_feature", _get_feature, use_cache=True)
