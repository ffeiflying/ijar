from ezcolorlog import root_logger as logger

# LOCAL PATCH (IJAR/cambrian_run): the upstream file imports every vision-tower
# variant eagerly, which drags in heavy optional deps (diffusers, maws, ...).
# Cambrian-8B only needs CLIP / SigLIP / DINOv2 / ConvNeXt, so the rest are
# imported best-effort: a missing dep no longer breaks the whole import chain.
from .clip_encoder import ClipVisionTower
from .siglip_encoder import SiglipVisionTower
from .clip_convnext_encoder import CLIPConvNextTower
from .dino_encoder import DinoVisionTower


def _optional(module, name):
    try:
        mod = __import__(module, fromlist=[name])
        return getattr(mod, name)
    except Exception as e:  # missing optional dependency
        logger.warning(f"optional vision tower {name} unavailable: {e}")
        return None


DfnClipVisionTower = _optional("cambrian.model.multimodal_encoder.dfn_clip_encoder", "DfnClipVisionTower")
EvaClipVisionTower = _optional("cambrian.model.multimodal_encoder.eva_clip_encoder", "EvaClipVisionTower")
IJepaVisionTower = _optional("cambrian.model.multimodal_encoder.ijepa_encoder", "IJepaVisionTower")
MAEVisionTower = _optional("cambrian.model.multimodal_encoder.mae_encoder", "MAEVisionTower")
MiDaSVisionTower = _optional("cambrian.model.multimodal_encoder.midas_encoder", "MiDaSVisionTower")
MoCoVisionTower = _optional("cambrian.model.multimodal_encoder.moco_encoder", "MoCoVisionTower")
SupervisedViT_VisionTower = _optional("cambrian.model.multimodal_encoder.supervised_vit_encoder", "SupervisedViT_VisionTower")
SAMVisionTower = _optional("cambrian.model.multimodal_encoder.sam_encoder", "SAMVisionTower")
DiffusionVisionTower = _optional("cambrian.model.multimodal_encoder.diffusion_encoder", "DiffusionVisionTower")
MawsVisionTower = _optional("cambrian.model.multimodal_encoder.maws_encoder", "MawsVisionTower")


def load_vision_model(vision_tower_name: str, args, **kwargs):
    """
    Load a vision tower model based on the model name

    Args:
        vision_tower_name (str): The name of the vision tower model.
        args (argparse.Namespace): The arguments parsed from the command line.
        kwargs: Additional keyword arguments.
    """

    if vision_tower_name.lower().startswith("hybridmodel"):
        raise ValueError("HybridModels must be loaded using the `multimodal_encoder.builderbuild_vision_tower()` function.")

    # CLIP-based Vision Towers
    if "openai/clip" in vision_tower_name.lower():
        logger.info(f"Loading **OpenAI CLIP** Vision Tower: {vision_tower_name}")
        return ClipVisionTower(vision_tower_name, args=args, **kwargs)
    if "apple/dfn" in vision_tower_name.lower():
        logger.info(f"Loading **Apple DFN CLIP** Vision Tower: {vision_tower_name}")
        return DfnClipVisionTower(vision_tower_name, args=args, **kwargs)
    if "siglip" in vision_tower_name.lower():
        logger.info(f"Loading **SigLIP CLIP** Vision Tower: {vision_tower_name}")
        return SiglipVisionTower(vision_tower_name, args=args, **kwargs)
    if "eva/clip" in vision_tower_name.lower():
        logger.info(f"Loading **EVA CLIP** Vision Tower: {vision_tower_name}")
        return EvaClipVisionTower(vision_tower_name, args=args, **kwargs)
    if "clip-convnext" in vision_tower_name.lower():
        logger.info(f"Loading **ConvNeXt CLIP** Vision Tower: {vision_tower_name}")
        return CLIPConvNextTower(vision_tower_name, args=args, **kwargs)

    # SSL-based Vision Towers
    if "dinov2" in vision_tower_name.lower():
        logger.info(f"Loading **DINOv2** Vision Tower: {vision_tower_name}")
        return DinoVisionTower(vision_tower_name, args=args, **kwargs)
    if "maws/" in vision_tower_name.lower():
        logger.info(f"Loading **MAWS** Vision Tower: {vision_tower_name}")
        return MawsVisionTower(vision_tower_name, args=args, **kwargs)
    if "mae" in vision_tower_name.lower():
        logger.info(f"Loading **MAE** Vision Tower: {vision_tower_name}")
        return MAEVisionTower(vision_tower_name, args=args, **kwargs)
    if "moco" in vision_tower_name.lower():
        logger.info(f"Loading **MoCo** Vision Tower: {vision_tower_name}")
        return MoCoVisionTower(vision_tower_name, args=args, **kwargs)
    if "ijepa" in vision_tower_name.lower():
        logger.info(f"Loading **IJepa** Vision Tower: {vision_tower_name}")
        return IJepaVisionTower(vision_tower_name, args=args, **kwargs)

    # Supervised Vision Towers
    if "supervised-vit" in vision_tower_name.lower():
        logger.info(f"Loading **Supervised** Vision Tower: {vision_tower_name}")
        return SupervisedViT_VisionTower(vision_tower_name, args=args, **kwargs)

    # Other Vision Towers
    if "diffusion" in vision_tower_name.lower():
        logger.info(f"Loading **Diffusion CLIP** Vision Tower: {vision_tower_name}")
        return DiffusionVisionTower(vision_tower_name, args=args, **kwargs)
    if "midas" in vision_tower_name.lower():
        logger.info(f"Loading **MiDaS** Vision Tower: {vision_tower_name}")
        return MiDaSVisionTower(vision_tower_name, args=args, **kwargs)
    if "sam" in vision_tower_name.lower():
        logger.info(f"Loading **SAM Vision Tower: {vision_tower_name}")
        return SAMVisionTower(vision_tower_name, args=args, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower_name}')
