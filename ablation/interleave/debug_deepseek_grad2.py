"""
Deep debug: trace gradient flow through every stage.
"""
import sys, os, random
import torch
import torch.nn as nn
from PIL import Image

_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _IJAR_DIR not in sys.path:
    sys.path.insert(0, _IJAR_DIR)
_ABLATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ABLATION_DIR not in sys.path:
    sys.path.insert(0, _ABLATION_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "deepseek"))

from interleave.deepseek.modeling_deepseek_vl_v2_dino import DeepseekVLV2ForCausalLM
from interleave.deepseek.processing_deepseek_vl_v2 import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images

sys.path.insert(0, "/data/liuxiaofei/uncertainty/Defense-Prefix-main")
from datasets.configs.imagenet_config import classes as IMAGENET_CLASSES

MODEL_PATH = ("/data/public/models/deepseek/"
              "models--deepseek-ai--deepseek-vl2-tiny/"
              "snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa")
IMAGENET_VAL = "/data/public/datasets/ImageNet-val/val"
DEVICE = "cuda:0"
TEMPLATE = "This is a photo of a {class_name}."


def check_grad(name, tensor):
    """Return grad stats for a tensor."""
    if tensor is None:
        return f"{name}: None"
    has = tensor.requires_grad
    isnan = torch.isnan(tensor).any().item() if has else False
    norm = tensor.norm().item()
    return f"{name}: shape={list(tensor.shape)} req_grad={has} norm={norm:.4f} nan={isnan}"


def main():
    print("=" * 60)
    print("Deep Gradient Trace")
    print("=" * 60)

    # Load model
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, ignore_mismatched_sizes=True,
        trust_remote_code=True)
    model = model.to(DEVICE).eval()
    processor = DeepseekVLV2Processor.from_pretrained(MODEL_PATH)

    # Set up DINO
    model.set_dino_mode("fused")
    model.reset_dino_projector()
    for name, param in model.named_parameters():
        if 'dino_projector' in name or name == 'dino_gate':
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Get 1 sample
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(IMAGENET_VAL)
    rng = random.Random(42)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    for i in indices:
        path, label = dataset.samples[i]
        class_name = IMAGENET_CLASSES[label]
        img_path = path
        break

    img = Image.open(img_path).convert('RGB').resize((384, 384))
    text = TEMPLATE.format(class_name=class_name)
    conversation = [
        {"role": "<|User|>", "content": f"<image>\n{text}", "images": [img]},
        {"role": "<|Assistant|>", "content": text},
    ]
    result = processor.process_one(
        conversations=conversation, images=[img],
        inference_mode=False, system_prompt="")
    prep = processor.batchify([result]).to(DEVICE)

    # Trace prepare_inputs_embeds step by step
    input_ids = prep.input_ids
    images = prep.images
    images_seq_mask = prep.images_seq_mask
    images_spatial_crop = prep.images_spatial_crop

    bs, max_n_images, _ = images_spatial_crop.shape
    batch_num_tiles = [0 for _ in range(bs)]
    total_tiles = []
    for idx in range(bs):
        for jdx in range(max_n_images):
            nw, nh = images_spatial_crop[idx, jdx]
            if nw == 0 or nh == 0:
                break
            batch_num_tiles[idx] += (1 + nw * nh)
        total_tiles.append(images[idx, :batch_num_tiles[idx]])
    total_tiles = torch.cat(total_tiles, dim=0)
    print(f"\n[1] total_tiles: {check_grad('total_tiles', total_tiles)}")

    images_feature = model.vision(total_tiles)
    print(f"[2] images_feature: {check_grad('images_feature', images_feature)}")

    images_embeds = model.projector(images_feature)
    print(f"[3] images_embeds (native): {check_grad('images_embeds', images_embeds)}")

    # DINO fusion
    if model.dino_mode != "native_only":
        model._init_dino(total_tiles.device)
        images_embeds_dino = model.extract_dino_feature(total_tiles, images_embeds)
        print(f"[4] images_embeds (fused): {check_grad('images_embeds_dino', images_embeds_dino)}")

        # Check individual DINO components
        px = total_tiles.float()
        px = torch.nn.functional.interpolate(px, size=(336, 336), mode='bilinear', align_corners=False)
        dino_out = model.dino_encoder.forward_features(px)
        dino_feats = dino_out['x_norm_patchtokens']
        print(f"[4a] dino_feats: {check_grad('dino_feats', dino_feats)}")

        # Pool
        N_tiles, num_dino, dino_dim = dino_feats.shape
        grid_dino = int(num_dino ** 0.5)
        target_grid = int(images_embeds.shape[1] ** 0.5)
        dino_feats_2d = dino_feats.reshape(N_tiles, grid_dino, grid_dino, dino_dim)
        dino_feats_2d = dino_feats_2d.permute(0, 3, 1, 2)
        dino_pooled = torch.nn.functional.adaptive_avg_pool2d(dino_feats_2d, (target_grid, target_grid))
        dino_pooled = dino_pooled.permute(0, 2, 3, 1)
        dino_pooled = dino_pooled.reshape(N_tiles, images_embeds.shape[1], dino_dim)
        print(f"[4b] dino_pooled: {check_grad('dino_pooled', dino_pooled)}")

        dino_proj = model.dino_projector(dino_pooled)
        print(f"[4c] dino_proj: {check_grad('dino_proj', dino_proj)}")
        print(f"     dino_gate = {model.dino_gate.item():.6f}")

        dino_residual = model.dino_gate * dino_proj
        print(f"[4d] dino_residual: {check_grad('dino_residual', dino_residual)}")

        fused = images_embeds + dino_residual
        print(f"[4e] fused: {check_grad('fused', fused)}")

        px_dtype = total_tiles.dtype
        fused_typed = fused.to(px_dtype)
        print(f"[4f] fused_typed: {check_grad('fused_typed', fused_typed)}")

        images_embeds = fused_typed

    # Trace through formatting
    _, hw, n_dim = images_embeds.shape
    h = w = int(hw ** 0.5)
    print(f"\n[5] After fusion: hw={hw}, h={h}, w={w}, n_dim={n_dim}")

    input_embeds = model.language.get_input_embeddings()(input_ids).clone()
    print(f"[6] initial input_embeds: {check_grad('input_embeds', input_embeds)}")

    tile_index = 0
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=2e-3)
    optimizer.zero_grad()

    for idx in range(images_spatial_crop.shape[0]):
        images_in_this_batch = []
        for jdx in range(images_spatial_crop.shape[1]):
            nw, nh = images_spatial_crop[idx, jdx]
            if nw == 0 or nh == 0:
                break
            num_tiles_in_image = nw * nh

            global_features = images_embeds[tile_index]
            local_features = images_embeds[tile_index + 1: tile_index + 1 + num_tiles_in_image]
            tile_index += num_tiles_in_image + 1

            print(f"\n[7] batch {idx} tile {jdx}:")
            print(f"  {check_grad('global_features', global_features)}")
            print(f"  {check_grad('local_features', local_features)}")

            if model.tile_tag == "2D":
                global_features = global_features.view(h, w, n_dim)
                new_lines_in_global = model.image_newline.unsqueeze(0).expand(h, 1, -1)
                global_features = torch.cat([global_features, new_lines_in_global], dim=1)
                global_features = global_features.view(-1, n_dim)
                print(f"  {check_grad('global_after_newline', global_features)}")

                local_features = local_features.reshape(num_tiles_in_image, h, w, n_dim)
                local_features = local_features.permute(0, 3, 1, 2)
                from einops import rearrange as re
                local_features = local_features.reshape(num_tiles_in_image, h*w, n_dim)

                new_lines_in_local = model.image_newline.unsqueeze(0).expand(num_tiles_in_image * h, 1, -1)
                local_features = local_features.reshape(num_tiles_in_image * h, w, n_dim)
                local_features = torch.cat([local_features, new_lines_in_local], dim=1)
                local_features = local_features.reshape(-1, n_dim)
                print(f"  {check_grad('local_after_newline', local_features)}")

                if model.global_view_pos == "head":
                    global_local_features = torch.cat(
                        [global_features, model.view_seperator[None, :], local_features], dim=0)
                else:
                    global_local_features = torch.cat(
                        [local_features, model.view_seperator[None, :], global_features], dim=0)
                print(f"  {check_grad('global_local_features', global_local_features)}")

            global_features.retain_grad()
            local_features.retain_grad()
            global_local_features.retain_grad()
            images_in_this_batch.append(global_local_features)

        if len(images_in_this_batch) > 0:
            images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
            print(f"\n[8] before scatter:")
            print(f"  {check_grad('images_in_this_batch', images_in_this_batch)}")
            print(f"  mask True count: {images_seq_mask[idx].bool().sum().item()}")
            print(f"  images_in_this_batch len: {images_in_this_batch.shape[0]}")

            # Do scatter and check grad
            idx_embeds = input_embeds[idx]
            mask = images_seq_mask[idx].bool().unsqueeze(-1)
            images_in_this_batch.retain_grad()
            idx_embeds.masked_scatter_(mask, images_in_this_batch)
            print(f"  {check_grad('input_embeds after scatter', input_embeds)}")
            print(f"  images_in_this_batch is_leaf={images_in_this_batch.is_leaf}")

    print(f"\n[9] Forward through LM...")
    outputs = model.language(
        inputs_embeds=input_embeds,
        attention_mask=prep.attention_mask,
        labels=prep.labels,
    )
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = outputs.loss
    print(f" loss = {loss.item():.4f}")

    input_embeds.retain_grad()

    print(f"\n[10] Backward...")
    loss.backward()

    print(f"  input_embeds.grad norm = {input_embeds.grad.norm().item():.6f}")
    print(f"  input_embeds.grad abs sum = {input_embeds.grad.abs().sum().item():.6f}")
    # Check if images_in_this_batch got any gradient
    if hasattr(images_in_this_batch, 'grad') and images_in_this_batch.grad is not None:
        print(f"  images_in_this_batch.grad norm = {images_in_this_batch.grad.norm().item():.6f}")
    else:
        print(f"  images_in_this_batch.grad = None")
    # Check upstream grads
    if hasattr(global_local_features, 'grad') and global_local_features.grad is not None:
        print(f"  global_local_features.grad norm = {global_local_features.grad.norm().item():.6f}")
    else:
        print(f"  global_local_features.grad = None")
    if hasattr(global_features, 'grad') and global_features.grad is not None:
        print(f"  global_features.grad norm = {global_features.grad.norm().item():.6f}")
    else:
        print(f"  global_features.grad = None")
    if hasattr(local_features, 'grad') and local_features.grad is not None:
        print(f"  local_features.grad norm = {local_features.grad.norm().item():.6f}")
    else:
        print(f"  local_features.grad = None")

    # Check all trainable params
    total_grad = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            gn = param.grad.norm().item()
            total_grad += gn ** 2
            if gn > 0:
                print(f"  {name}: grad_norm = {gn:.6f}")
    total_grad = total_grad ** 0.5
    print(f"  Total grad norm: {total_grad:.6f}")

    print(f"\n[11] Final check:")
    print(f"  dino_gate.grad = {model.dino_gate.grad}")


if __name__ == "__main__":
    main()
