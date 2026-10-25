"""
Qwen2.5-VL-7B with DINOv2 augmentation (IJAR 3.1 — Feature Interleaving)
==========================================================================
Wraps the original Qwen2.5-VL model, adding a frozen DINOv2 encoder and
trainable projector. Supports: "native_only", "concat", "interleaved".

Usage:
    model = load_qwen_dino("native_only")        # original behaviour
    model = load_qwen_dino("concat")             # CLIP+DINO concatenated
    model = load_qwen_dino("interleaved")        # CLIP+DINO interleaved
    model.set_dino_mode("concat")                # switch at runtime
"""

import sys
import os
import torch
import torch.nn as nn


# ── DINOv2 lazy-load ────────────────────────────────────────────────────
def _get_dinov2():
    """Lazy-load DINOv2 ViT-L/14 encoder from local cache (no network needed)."""
    os.environ["XFORMERS_DISABLED"] = "1"
    _DINO_LOCAL = "/data/liuxiaofei/torch_cache/hub/facebookresearch_dinov2_main"
    _WEIGHT_PATH = "/data/liuxiaofei/torch_cache/hub/checkpoints/dinov2_vitl14_pretrain.pth"
    if _DINO_LOCAL not in sys.path:
        sys.path.insert(0, _DINO_LOCAL)
    # Build model architecture (no pretrained), then load local weights
    from dinov2.hub.backbones import dinov2_vitl14 as _make_model
    model = _make_model(pretrained=False)
    state = torch.load(_WEIGHT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ── DINO Projector ──────────────────────────────────────────────────────
class DINOProjector(nn.Module):
    """2-layer MLP: 1024 → hidden → hidden → llm_hidden, with learnable gate."""
    def __init__(self, llm_hidden_size=3584, intermediate_size=None):
        super().__init__()
        hidden = intermediate_size or 1024
        self.net = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, hidden),
            nn.GELU(),
            nn.Linear(hidden, llm_hidden_size),
        )
        self.gate = nn.Parameter(torch.zeros(1))  # starts silent

    def forward(self, x):
        return self.net(x) * self.gate


# ── Wrapper ─────────────────────────────────────────────────────────────
class QwenVLWithDINO(nn.Module):
    """
    Wraps Qwen2_5_VLForConditionalGeneration with optional DINOv2 features.

    DINO features are spatial-pooled to match the native token count, then
    projected to LLM hidden_size. The combined features replace the image
    tokens in the input embeddings.

    Token arrangement:
      native_only:  [nat0, nat1, ..., natN]           N tokens
      concat:       [nat0..natN, din0..dinN]          2N tokens (need 2x image tokens in input)
      interleaved:  [nat0, din0, nat1, din1, ...]    2N tokens
    """
    def __init__(self, base_model, dino_mode="native_only"):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.llm_hidden_size = base_model.config.hidden_size  # 3584

        self.dino_encoder = None  # lazy-loaded
        self.dino_projector = DINOProjector(self.llm_hidden_size)
        self._dino_mode = "native_only"
        self.set_dino_mode(dino_mode)

        # Store reference for ACF computation
        self.model = base_model.model
        self.lm_head = base_model.lm_head
        self.device = base_model.device

    def _init_dino(self, device):
        if self.dino_encoder is not None:
            return
        dino = _get_dinov2()
        self.dino_encoder = dino.to(device).float().eval()
        for p in self.dino_encoder.parameters():
            p.requires_grad = False
        # Projector: convert to fp32 and re-init to avoid bfloat16 NaN
        self.dino_projector = self.dino_projector.float()
        for m in self.dino_projector.net:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        nn.init.zeros_(self.dino_projector.net[-1].weight)
        nn.init.zeros_(self.dino_projector.net[-1].bias)
        print(f"  [DINO] dinov2_vitl14 loaded on {device} (fp32, gate=0)")

    def set_dino_mode(self, mode: str):
        assert mode in ("native_only", "concat", "interleaved")
        self._dino_mode = mode

    @property
    def dino_mode(self):
        return self._dino_mode

    def extract_dino_feature(self, pixel_values):
        """
        Extract DINOv2 features, pool to match native token count, project.

        Args:
            pixel_values: [total_patches, C, H, W]  (sum over images)
        Returns:
            dino_embeds: [total_patches * N_native, llm_hidden_size]
        """
        self._init_dino(pixel_values.device)

        # Resize to DINO native resolution (336)
        if pixel_values.shape[-1] != 336 or pixel_values.shape[-2] != 336:
            px = torch.nn.functional.interpolate(
                pixel_values, size=(336, 336), mode='bilinear',
                align_corners=False)
        else:
            px = pixel_values

        dino_out = self.dino_encoder.forward_features(px)
        dino_feats = dino_out['x_norm_patchtokens']  # [M, 576, 1024]
        M, num_dino, dino_dim = dino_feats.shape  # 576 for 336×336
        grid_dino = int(num_dino ** 0.5)  # 24

        # We need to match the native token count per image
        # For each image, native has image_grid_thw[i, 1]*image_grid_thw[i, 2] tokens
        # DINO has grid_dino^2 = 576 tokens at 336×336
        # Native has grid_h*grid_w after spatial merge
        # We don't know native count here — pool to a fixed ratio
        # Simplify: use adaptive pool to match native token count

        # Actually, we'll handle token matching in the forward pass
        # where we know the native token count
        dino_feats = dino_feats.reshape(M, grid_dino, grid_dino, dino_dim)
        dino_feats = dino_feats.permute(0, 3, 1, 2)  # [M, 1024, 24, 24]

        return dino_feats  # Return spatial features for later pooling

    def extract_dino_feature_pooled(self, pixel_values, native_img_tokens,
                                     image_grid_thw):
        """
        Extract DINO features and pool to match native token counts.

        Args:
            pixel_values: [M, C, H, W]
            native_img_tokens: total number of native image tokens
            image_grid_thw: [num_images, 3] — (t, h, w) for each image in LLM grid
        Returns:
            dino_flat: [native_img_tokens, llm_hidden_size]  (matches native count)
        """
        self._init_dino(pixel_values.device)
        px_dtype = pixel_values.dtype
        pixel_values = pixel_values.float()

        if pixel_values.shape[-1] != 336 or pixel_values.shape[-2] != 336:
            px = torch.nn.functional.interpolate(
                pixel_values, size=(336, 336), mode='bilinear',
                align_corners=False)
        else:
            px = pixel_values

        dino_out = self.dino_encoder.forward_features(px)
        dino_feats = dino_out['x_norm_patchtokens']  # [M, 576, 1024]

        # Pool per-image to match native token counts
        pooled_list = []
        for i, (t, h, w) in enumerate(image_grid_thw):
            n_tokens = int(h * w)  # native token count for this image
            sample_feats = dino_feats[i:i+1]  # [1, 576, 1024]

            # Reshape to 2D: [1, 24, 24, 1024]
            grid = int(sample_feats.shape[1] ** 0.5)
            sample_feats = sample_feats.reshape(1, grid, grid, -1)

            # Adaptive pool: [1, grid_h, grid_w, 1024] → [1, h, w, 1024]
            target_h, target_w = int(h), int(w)
            sample_feats = sample_feats.permute(0, 3, 1, 2)  # [1, 1024, 24, 24]
            sample_feats = torch.nn.functional.adaptive_avg_pool2d(
                sample_feats, (target_h, target_w))
            sample_feats = sample_feats.permute(0, 2, 3, 1)  # [1, h, w, 1024]
            sample_feats = sample_feats.reshape(1, n_tokens, -1)

            pooled_list.append(sample_feats.squeeze(0))

        dino_pooled = torch.cat(pooled_list, dim=0)  # [total_native_tokens, 1024]

        # Check
        assert dino_pooled.shape[0] == native_img_tokens, (
            f"DINO pooled tokens ({dino_pooled.shape[0]}) != native ({native_img_tokens})")

        dino_embeds = self.dino_projector(dino_pooled)
        return dino_embeds.to(px_dtype)  # [native_img_tokens, llm_hidden_size]

    # ── Forward delegation ──────────────────────────────────────────
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        rope_deltas=None,
        cache_position=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        if self._dino_mode == "native_only" or pixel_values is None:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                rope_deltas=rope_deltas,
                cache_position=cache_position,
                second_per_grid_ts=second_per_grid_ts,
                **kwargs,
            )

        # ── DINO mode: build combined embeddings ────────────────────
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None
                                else self.config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None
                       else self.config.use_return_dict)

        # 1) Native visual features
        inputs_embeds = self.base_model.model.embed_tokens(input_ids)
        pixel_values = pixel_values.type(self.base_model.visual.dtype)
        native_image_embeds = self.base_model.visual(
            pixel_values, grid_thw=image_grid_thw)
        n_native = native_image_embeds.shape[0]
        n_image_tokens_in_input = (input_ids == self.config.image_token_id).sum().item()

        # 2) DINO features
        dino_embeds = self.extract_dino_feature_pooled(
            pixel_values, n_native, image_grid_thw)
        dino_embeds = dino_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

        # 3) Merge
        if self._dino_mode == "interleaved":
            merged = torch.empty(2 * n_native, self.llm_hidden_size,
                                 dtype=inputs_embeds.dtype,
                                 device=inputs_embeds.device)
            merged[0::2] = native_image_embeds
            merged[1::2] = dino_embeds
        else:  # concat
            merged = torch.cat([native_image_embeds, dino_embeds], dim=0)

        n_merged = merged.shape[0]

        # 4) Replace image tokens in inputs_embeds
        # We need 2x image tokens for DINO mode
        n_expected = n_image_tokens_in_input
        if n_expected != n_merged:
            raise ValueError(
                f"DINO mode '{self._dino_mode}' requires {n_merged} image tokens "
                f"in input_ids, but found {n_expected}. "
                f"Use double_image_tokens() helper in the inference script.")

        mask = input_ids == self.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, merged)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        # 5) Standard forward (same as original)
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.base_model.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.base_model.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw,
                    second_per_grid_ts, attention_mask)
                self.base_model.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.base_model.rope_deltas).to(
                        inputs_embeds.device)
                    if cache_position is not None else 0)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(
                        batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.base_model.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        from transformers.modeling_outputs import CausalLMOutputWithPast
        hidden_states = outputs[0]
        logits = self.base_model.lm_head(hidden_states)

        loss = None
        if labels is not None:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.base_model.rope_deltas,
        )

    def generate(self, **kwargs):
        """Delegate to base model's generate."""
        return self.base_model.generate(**kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def __getattr__(self, name):
        """Fall through to base_model for any attribute not defined here."""
        # Avoid infinite recursion
        if name in ('base_model', '_dino_mode', '_modules', '_parameters',
                     '_buffers', 'training', 'dino_encoder', 'dino_projector'):
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


# ── Token helper ────────────────────────────────────────────────────────
def double_image_tokens(inputs):
    """
    Double image tokens in the tokenized inputs for DINO mode.

    Qwen2.5-VL uses <|vision_start|><|image_pad|>...<|vision_end|> format.
    This function doubles each <|image_pad|> token.

    Args:
        inputs: dict from processor (has 'input_ids', 'attention_mask')
    Returns:
        modified inputs dict
    """
    import copy
    inputs = copy.deepcopy(inputs)

    image_token_id = 151655  # <|image_pad|> for Qwen2.5-VL

    new_input_ids_list = []
    new_attention_mask_list = []

    for batch_idx in range(inputs['input_ids'].shape[0]):
        ids = inputs['input_ids'][batch_idx]
        mask = inputs['attention_mask'][batch_idx] if 'attention_mask' in inputs else None

        # Find image_pad tokens and double each
        is_img = (ids == image_token_id)
        indices = torch.where(is_img)[0]

        if len(indices) == 0:
            new_input_ids_list.append(ids)
            if mask is not None:
                new_attention_mask_list.append(mask)
            continue

        new_ids_list = []
        prev = 0
        for idx in indices:
            new_ids_list.append(ids[prev:idx+1])  # up to and including image token
            new_ids_list.append(ids[idx:idx+1])    # duplicate
            prev = idx + 1
        new_ids_list.append(ids[prev:])
        new_ids = torch.cat(new_ids_list, dim=0)

        new_input_ids_list.append(new_ids)
        if mask is not None:
            new_mask = torch.ones(len(new_ids), dtype=mask.dtype, device=mask.device)
            new_attention_mask_list.append(new_mask)

    # Pad to max length
    from torch.nn.utils.rnn import pad_sequence
    new_input_ids = pad_sequence(new_input_ids_list, batch_first=True,
                                  padding_value=0)
    result = {'input_ids': new_input_ids}
    if 'attention_mask' in inputs:
        new_attn_mask = pad_sequence(new_attention_mask_list, batch_first=True,
                                      padding_value=0)
        result['attention_mask'] = new_attn_mask

    return result


# ── Loader ──────────────────────────────────────────────────────────────
def load_qwen_dino(dino_mode="native_only", device_map="cuda:1",
                   torch_dtype=torch.float16,
                   model_name="Qwen/Qwen2.5-VL-7B-Instruct",
                   cache_dir="/data/public/models/Qwen"):
    """Load Qwen2.5-VL-7B and wrap with DINO support."""
    print(f"[1/2] Loading base Qwen2.5-VL-7B...")
    from transformers import Qwen2_5_VLForConditionalGeneration

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        cache_dir=cache_dir,
    )
    base.eval()
    print(f"  Base model loaded on {base.device}")

    print(f"[2/2] Adding DINOv2 augmentation (mode={dino_mode})...")
    wrapped = QwenVLWithDINO(base, dino_mode=dino_mode)
    print(f"  DINO mode: {dino_mode}")
    print(f"  DINO projector params: "
          f"{sum(p.numel() for p in wrapped.dino_projector.parameters()):,}")

    return wrapped
