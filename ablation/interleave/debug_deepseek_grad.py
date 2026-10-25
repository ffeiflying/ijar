"""
Debug script: verify DINO projector training for DeepSeek-VL2-Tiny.
Tests on 4 images with 50 steps — loss should drop quickly if gradient flows.
"""
import sys, os, random
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

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


def main():
    print("=" * 60)
    print("DeepSeek-VL2 DINO Projector — Gradient Debug")
    print("=" * 60)

    # 1) Load model
    print("\n[1] Loading model...")
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=DEVICE, ignore_mismatched_sizes=True,
        trust_remote_code=True)
    model = model.to(DEVICE).eval()
    processor = DeepseekVLV2Processor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer

    # 2) Set up DINO
    model.set_dino_mode("fused")
    model.reset_dino_projector()

    for name, param in model.named_parameters():
        if 'dino_projector' in name or name == 'dino_gate':
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,}")

    # Print gate initial value
    print(f"  dino_gate = {model.dino_gate.item():.6f}")

    # 3) Get 4 training samples
    print("\n[2] Loading 4 training samples...")
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(IMAGENET_VAL)
    rng = random.Random(42)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    samples = []
    for i in indices:
        path, label = dataset.samples[i]
        class_name = IMAGENET_CLASSES[label]
        samples.append({"image_path": path, "class_name": class_name})
        if len(samples) >= 4:
            break

    # 4) Run native_only baseline
    print("\n[3] Native-only baseline...")
    model.set_dino_mode("native_only")
    losses_native = []
    with torch.no_grad():
        for s in samples:
            img = Image.open(s["image_path"]).convert('RGB').resize((384, 384))
            text = TEMPLATE.format(class_name=s["class_name"])
            conversation = [
                {"role": "<|User|>", "content": f"<image>\n{text}",
                 "images": [img]},
                {"role": "<|Assistant|>", "content": text},
            ]
            result = processor.process_one(
                conversations=conversation, images=[img],
                inference_mode=False, system_prompt="")
            prep = processor.batchify([result]).to(DEVICE)
            embeds = model.prepare_inputs_embeds(
                input_ids=prep.input_ids, images=prep.images,
                images_seq_mask=prep.images_seq_mask,
                images_spatial_crop=prep.images_spatial_crop)
            out = model.language(inputs_embeds=embeds,
                                attention_mask=prep.attention_mask,
                                labels=prep.labels)
            if isinstance(out, tuple):
                out = out[0]
            losses_native.append(out.loss.item())
    print(f"  Native-only losses: {[f'{l:.4f}' for l in losses_native]}")
    print(f"  Mean: {sum(losses_native)/len(losses_native):.4f}")

    # 5) Training check: 50 steps on these 4 samples
    print("\n[4] Training 50 steps (fused mode)...")
    model.set_dino_mode("fused")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=2e-3)

    all_losses = []
    grad_norms = []

    for step in range(50):
        step_losses = []
        optimizer.zero_grad()

        for s in samples:
            img = Image.open(s["image_path"]).convert('RGB').resize((384, 384))
            text = TEMPLATE.format(class_name=s["class_name"])
            conversation = [
                {"role": "<|User|>", "content": f"<image>\n{text}",
                 "images": [img]},
                {"role": "<|Assistant|>", "content": text},
            ]
            result = processor.process_one(
                conversations=conversation, images=[img],
                inference_mode=False, system_prompt="")
            prep = processor.batchify([result]).to(DEVICE)
            embeds = model.prepare_inputs_embeds(
                input_ids=prep.input_ids, images=prep.images,
                images_seq_mask=prep.images_seq_mask,
                images_spatial_crop=prep.images_spatial_crop)
            out = model.language(inputs_embeds=embeds,
                                attention_mask=prep.attention_mask,
                                labels=prep.labels)
            if isinstance(out, tuple):
                out = out[0]
            loss = out.loss / 4.0  # average over 4 samples
            loss.backward()
            step_losses.append(out.loss.item())

        # Check gradient
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None and p.requires_grad:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

        optimizer.step()
        avg_loss = sum(step_losses) / len(step_losses)
        all_losses.append(avg_loss)

        if step < 5 or step % 10 == 0:
            gate_val = model.dino_gate.item()
            proj_norm = sum(
                p.data.norm(2).item()
                for n, p in model.named_parameters()
                if 'dino_projector' in n and p.requires_grad)
            print(f"  step {step:3d}: loss={avg_loss:.4f}  "
                  f"grad={total_norm:.6f}  gate={gate_val:.6f}  "
                  f"proj_norm={proj_norm:.4f}")

    print(f"\n[5] Summary:")
    print(f"  Loss: {all_losses[0]:.4f} -> {all_losses[-1]:.4f} "
          f"(delta = {all_losses[-1] - all_losses[0]:.4f})")
    print(f"  Gate: {model.dino_gate.item():.6f}")
    print(f"  Grad norm range: {min(grad_norms):.6f} ~ {max(grad_norms):.6f}")


if __name__ == "__main__":
    main()
