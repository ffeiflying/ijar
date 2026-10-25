"""
Train DINO Projector for Qwen2.5-VL / InternVL2_5 (IJAR 3.1)
===============================================================
Only trains the DINO projector MLP. Freezes: LLM, native vision encoder,
native projector, DINOv2 encoder.

Training data: ImageNet validation subset (excluding STA images).
Objective: Next-token prediction on class-name prompts.

Usage:
    python train_dino_projector.py --model qwen --mode concat
    python train_dino_projector.py --model internvl --mode interleaved
"""

import sys, os, argparse, json, random
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
import numpy as np

_IJAR_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _IJAR_DIR not in sys.path:
    sys.path.insert(0, _IJAR_DIR)

# Also add ablation/ so "interleave" package is importable
_ABLATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ABLATION_DIR not in sys.path:
    sys.path.insert(0, _ABLATION_DIR)

# Import ImageNet class names
sys.path.insert(0, "/data/liuxiaofei/uncertainty/Defense-Prefix-main")
from datasets.configs.imagenet_config import classes as IMAGENET_CLASSES

# ── Config ──────────────────────────────────────────────────────────────
IMAGENET_VAL = "/data/public/datasets/ImageNet-val/val"
STA_IMAGE_DIR = os.path.join(_IJAR_DIR, "datasets", "STA", "images")
OUTPUT_DIR = os.path.join(_IJAR_DIR, "results", "interleave")

TRAIN_SIZE = 2000
BATCH_SIZE = 2
LEARNING_RATE = 2e-3
NUM_EPOCHS = 3
SEED = 42
GRAD_ACCUM = 8  # effective batch = 16

# Template for training
TEMPLATES = [
    "This is a photo of a {class_name}.",
    "The image shows a {class_name}.",
    "A {class_name} is in the picture.",
    "a photo of a {class_name}.",
]


def get_train_images():
    """Get training image paths and class names (excluding STA images)."""
    # Load STA image IDs to exclude
    sta_csv = os.path.join(_IJAR_DIR, "datasets", "STA", "sta_mini.csv")
    import pandas as pd
    sta_df = pd.read_csv(sta_csv)
    sta_filenames = set(sta_df["index"].tolist())

    dataset = ImageFolder(IMAGENET_VAL)
    rng = random.Random(SEED)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    train_entries = []
    for i in indices:
        if len(train_entries) >= TRAIN_SIZE:
            break
        path, label = dataset.samples[i]
        basename = os.path.basename(path)
        if basename in sta_filenames:
            continue
        class_name = IMAGENET_CLASSES[label]
        train_entries.append({"image_path": path, "class_name": class_name})

    print(f"  Training images: {len(train_entries)} (STA images excluded)")
    return train_entries


class ImageNetTrainDataset(Dataset):
    def __init__(self, entries, model_type="qwen"):
        self.entries = entries
        self.model_type = model_type
        # For InternVL, we need the dynamic preprocess
        if model_type == "internvl":
            self.transform = None  # handled by load_image in main
        else:
            self.transform = transforms.Compose([
                transforms.Resize((336, 336)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        template = random.choice(TEMPLATES)
        target_text = template.format(class_name=entry["class_name"])
        return {
            "image_path": entry["image_path"],
            "target_text": target_text,
            "class_name": entry["class_name"],
        }


def train_qwen(train_entries, mode, device="cuda:1"):
    """Train DINO projector for Qwen2.5-VL."""
    print("\n" + "=" * 60)
    print(f"Training Qwen2.5-VL DINO projector (mode={mode})")
    print("=" * 60)

    # --- Load model ---
    from interleave.qwen.qwen_dino_wrapper import (
        load_qwen_dino, QwenVLWithDINO, double_image_tokens)

    model = load_qwen_dino(dino_mode=mode, device_map=device)
    model = model.to(device)
    model.train()

    # Freeze everything except DINO projector
    for name, param in model.named_parameters():
        if 'dino_projector' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE)

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        cache_dir="/data/public/models/Qwen")

    dataset = ImageNetTrainDataset(train_entries, "qwen")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    all_step_losses = []
    optimizer.zero_grad()

    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0.0
        for step, batch in enumerate(pbar):
            # Build messages
            messages_list = []
            for i in range(len(batch["image_path"])):
                messages_list.append([
                    {"role": "user", "content": [
                        {"type": "image", "image": batch["image_path"][i]},
                        {"type": "text", "text": batch["target_text"][i]},
                    ]},
                ])

            # Tokenize
            texts = []
            for msgs in messages_list:
                # Use simple format: image + target text
                text = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False)
                texts.append(text)

            # For training, use the text as labels
            inputs = processor(
                text=texts, images=[m[0]["content"][0]["image"] for m in messages_list],
                padding=True, return_tensors="pt",
            ).to(device)

            # Double image tokens for DINO mode
            inputs = double_image_tokens(inputs)

            # Forward: compute loss on the target text
            labels = inputs["input_ids"].clone()
            # Mask out image tokens in labels (set to -100)
            image_token_id = model.config.image_token_id
            labels[labels == image_token_id] = -100

            output = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                labels=labels,
                use_cache=False,
            )

            loss = output.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

            loss_val = loss.item() * GRAD_ACCUM
            all_step_losses.append(loss_val)
            epoch_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    return model, all_step_losses


def train_internvl(train_entries, mode, device="cuda:0"):
    """Train DINO projector for InternVL2_5-8B."""
    print("\n" + "=" * 60)
    print(f"Training InternVL2_5 DINO projector (mode={mode})")
    print("=" * 60)

    # --- Load original InternVL model first ---
    from transformers import AutoModel, AutoTokenizer

    model_id = "OpenGVLab/InternVL2_5-8B"
    model_cache = "/data/public/models/InternVL/InternVL2_5-8B"

    # We need to use our DINO-augmented model
    _INTERLEAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_INTERLEAVE_DIR, "internvl"))
    from modeling_internvl_chat_dino import InternVLChatModelDINO

    MODEL_DIR = "/data/liuxiaofei/huggingface/modules/transformers_modules/" \
                "OpenGVLab/InternVL2_5_hyphen_8B/e9e4c0dc1db56bfab10458671519b7fa3dd29463"
    CONFIG_PATH = os.path.join(MODEL_DIR, "configuration_internvl_chat.py")

    print(f"  Loading from: {model_id} (cache: {model_cache})")
    model = InternVLChatModelDINO.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        use_flash_attn=True,
        trust_remote_code=True, cache_dir=model_cache,
        ignore_mismatched_sizes=True)
    model = model.to(device).eval()

    # Set DINO mode and re-init projector for training
    model.set_dino_mode(mode)
    model.reset_dino_projector()

    # Freeze everything except DINO projector (and the gate, for "fused")
    for name, param in model.named_parameters():
        if 'dino_projector' in name or name == 'dino_gate':
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=False,
        cache_dir=model_cache)

    # For InternVL, we manually build inputs
    dataset = ImageNetTrainDataset(train_entries, "internvl")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    all_step_losses = []
    optimizer.zero_grad()

    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode
    import torchvision.transforms as T

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0.0
        for step, batch in enumerate(pbar):
            pixel_values_list = []
            input_ids_list = []

            for i in range(len(batch["image_path"])):
                img = Image.open(batch["image_path"][i]).convert('RGB')
                img = img.resize((448, 448))
                img_tensor = T.ToTensor()(img)
                img_tensor = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(img_tensor)
                pixel_values_list.append(img_tensor)

                target = batch["target_text"][i]
                # Build input: <image>\n{target}
                text = f"<image>\n{target}"
                model.img_context_token_id = tokenizer.convert_tokens_to_ids(
                    '<IMG_CONTEXT>')

                # Create image tokens
                img_token_count = model.num_image_token
                if mode not in ("native_only", "fused"):
                    img_token_count *= 2
                image_tokens = '<img>' + '<IMG_CONTEXT>' * img_token_count + '</img>'
                text_with_img = text.replace('<image>', image_tokens, 1)

                tok = tokenizer(text_with_img, return_tensors='pt')
                input_ids_list.append(tok['input_ids'].squeeze(0))

            # Pad
            max_len = max(len(ids) for ids in input_ids_list)
            input_ids = torch.zeros(BATCH_SIZE, max_len, dtype=torch.long)
            attention_mask = torch.zeros(BATCH_SIZE, max_len, dtype=torch.long)
            for i, ids in enumerate(input_ids_list):
                input_ids[i, :len(ids)] = ids
                attention_mask[i, :len(ids)] = 1

            pixel_values = torch.stack(pixel_values_list).to(
                device, dtype=torch.bfloat16)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Labels: predict target text
            labels = input_ids.clone()
            # Mask image tokens
            img_ctx_id = model.img_context_token_id
            labels[labels == img_ctx_id] = -100
            labels[labels == tokenizer.convert_tokens_to_ids('<img>')] = -100
            labels[labels == tokenizer.convert_tokens_to_ids('</img>')] = -100

            image_flags = torch.ones(BATCH_SIZE, 1, dtype=torch.long, device=device)

            output = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                labels=labels,
                use_cache=False,
            )

            loss = output.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

            loss_val = loss.item() * GRAD_ACCUM
            all_step_losses.append(loss_val)
            epoch_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        if getattr(model, 'dino_gate', None) is not None:
            print(f"  Epoch {epoch+1} dino_gate: "
                  f"{float(model.dino_gate.item()):.5f}")

    return model, all_step_losses


def train_deepseek(train_entries, device="cuda:0"):
    """Train DINO projector + fusion layer for DeepSeek-VL2-Tiny."""
    print("\n" + "=" * 60)
    print("Training DeepSeek-VL2-Tiny DINO projector (mode=fused)")
    print("=" * 60)

    # Add deepseek directory to path
    _INTERLEAVE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_INTERLEAVE_DIR, "deepseek"))
    from modeling_deepseek_vl_v2_dino import DeepseekVLV2ForCausalLM

    MODEL_PATH = ("/data/public/models/deepseek/"
                  "models--deepseek-ai--deepseek-vl2-tiny/"
                  "snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa")

    print(f"  Loading from: {MODEL_PATH}")
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16,
        device_map=device, ignore_mismatched_sizes=True,
        trust_remote_code=True)
    model = model.to(device).eval()

    # Set DINO mode and re-init projector
    model.set_dino_mode("fused")
    model.reset_dino_projector()

    # Freeze everything except DINO projector and gate
    for name, param in model.named_parameters():
        if 'dino_projector' in name or name == 'dino_gate':
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE)

    # Use processor for tokenizer + image preprocessing
    from processing_deepseek_vl_v2 import DeepseekVLV2Processor
    processor = DeepseekVLV2Processor.from_pretrained(MODEL_PATH)
    tokenizer = processor.tokenizer

    dataset = ImageNetTrainDataset(train_entries, "deepseek")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    all_step_losses = []
    optimizer.zero_grad()

    from PIL import Image
    import torchvision.transforms as T

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Resize((384, 384), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Image token count: 1 global (210) + 1 view_sep (1) + 1 local (210) = 421
    N_IMG_TOKENS = 421

    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        epoch_loss = 0.0
        for step, batch in enumerate(pbar):
            images_list = []
            text_list = []
            for i in range(len(batch["image_path"])):
                img = Image.open(batch["image_path"][i]).convert('RGB')
                img_tensor = transform(img)
                images_list.append(img_tensor)
                text_list.append(batch["target_text"][i])

            # Build simple input: [PAD * 421] + tokenized_text
            PAD_ID = tokenizer.pad_token_id or 0
            BOS_ID = tokenizer.bos_token_id
            EOS_ID = tokenizer.eos_token_id

            input_ids_list, labels_list = [], []
            for text in text_list:
                text_ids = tokenizer.encode(text, add_special_tokens=False)
                # Prepend BOS + image placeholders + text + EOS
                ids = torch.tensor(
                    [BOS_ID] + [PAD_ID] * N_IMG_TOKENS + text_ids + [EOS_ID],
                    dtype=torch.long)
                input_ids_list.append(ids)
                lab = ids.clone()
                lab[1:1+N_IMG_TOKENS] = -100  # mask image tokens
                labels_list.append(lab)

            # Pad to batch
            max_len = max(len(ids) for ids in input_ids_list)
            input_ids = torch.zeros(BATCH_SIZE, max_len, dtype=torch.long)
            attention_mask = torch.zeros(BATCH_SIZE, max_len, dtype=torch.long)
            labels = torch.full((BATCH_SIZE, max_len), -100, dtype=torch.long)
            images_seq_mask = torch.zeros(BATCH_SIZE, max_len, dtype=torch.long)

            for i in range(len(input_ids_list)):
                L = len(input_ids_list[i])
                input_ids[i, :L] = input_ids_list[i]
                attention_mask[i, :L] = 1
                labels[i, :L] = labels_list[i]
                # Mark image positions (after BOS)
                images_seq_mask[i, 1:1+N_IMG_TOKENS] = 1

            # Build images: [B, 2, 3, 384, 384] — global + local per image
            images = torch.zeros(BATCH_SIZE, 2, 3, 384, 384)
            for i, img_t in enumerate(images_list):
                images[i, 0] = img_t
                images[i, 1] = img_t
            images_spatial_crop = torch.ones(BATCH_SIZE, 1, 2, dtype=torch.long)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            images_seq_mask = images_seq_mask.to(device)
            images = images.to(device, dtype=torch.bfloat16)
            images_spatial_crop = images_spatial_crop.to(device)

            # Compute input embeddings (DINO fusion happens here)
            inputs_embeds = model.prepare_inputs_embeds(
                input_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )

            # Forward through language model
            outputs = model.language(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = outputs.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()

            loss_val = loss.item() * GRAD_ACCUM
            all_step_losses.append(loss_val)
            epoch_loss += loss_val
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    return model, all_step_losses


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["qwen", "internvl", "deepseek"])
    parser.add_argument("--mode", default="concat",
                        choices=["concat", "interleaved", "fused"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get training images
    print("[1/3] Preparing training data...")
    train_entries = get_train_images()
    print(f"  {len(train_entries)} training samples")

    # Train
    print(f"[2/3] Training {args.model} DINO projector (mode={args.mode})...")
    if args.model == "qwen":
        model, losses = train_qwen(train_entries, args.mode, args.device)
    elif args.model == "internvl":
        model, losses = train_internvl(train_entries, args.mode, args.device)
    else:
        model, losses = train_deepseek(train_entries, args.device)

    # Save
    print("[3/3] Saving DINO projector weights...")
    save_path = os.path.join(OUTPUT_DIR,
                             f"{args.model}_dino_projector_{args.mode}.pth")
    if args.model == "deepseek":
        dino_state = {
            'dino_projector': model.dino_projector.state_dict(),
            'dino_gate': model.dino_gate.item(),
            'losses': losses,
            'mode': 'fused',
            'num_epochs': NUM_EPOCHS,
            'train_size': len(train_entries),
        }
    else:
        dino_state = {
            'dino_projector': model.dino_projector.state_dict(),
            'losses': losses,
            'mode': args.mode,
            'num_epochs': NUM_EPOCHS,
            'train_size': len(train_entries),
        }
        if getattr(model, 'dino_gate', None) is not None:
            dino_state['dino_gate'] = float(model.dino_gate.item())
    torch.save(dino_state, save_path)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    main()
