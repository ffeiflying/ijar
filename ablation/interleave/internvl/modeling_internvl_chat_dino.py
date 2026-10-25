# --------------------------------------------------------
# InternVL with DINOv2 augmentation (IJAR 3.1 — Feature Interleaving)
# Modified from: modeling_internvl_chat.py
# --------------------------------------------------------
# Adds a frozen DINOv2 encoder + trainable projector for DINO features.
# Supports: "native_only" (original), "concat", "interleaved" modes.
# --------------------------------------------------------

import sys, os
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

# Import original InternVL modules — make them a fake package for relative imports
_ORIG_DIR = "/data/liuxiaofei/huggingface/modules/transformers_modules/" \
            "OpenGVLab/InternVL2_5_hyphen_8B/e9e4c0dc1db56bfab10458671519b7fa3dd29463"
_ORIG_PARENT = os.path.dirname(_ORIG_DIR)
_ORIG_PKG = os.path.basename(_ORIG_DIR)

# Register the original directory as a package (needed for relative imports inside)
import types as _types
if _ORIG_PKG not in sys.modules:
    _pkg = _types.ModuleType(_ORIG_PKG)
    _pkg.__path__ = [_ORIG_DIR]
    sys.modules[_ORIG_PKG] = _pkg

# Import from the package
from importlib import import_module as _import_module
_config = _import_module(f"{_ORIG_PKG}.configuration_internvl_chat")
InternVLChatConfig = _config.InternVLChatConfig
_conv = _import_module(f"{_ORIG_PKG}.conversation")
get_conv_template = _conv.get_conv_template
_vit = _import_module(f"{_ORIG_PKG}.modeling_intern_vit")
InternVisionModel = _vit.InternVisionModel
has_flash_attn = _vit.has_flash_attn
_lm = _import_module(f"{_ORIG_PKG}.modeling_internlm2")
InternLM2ForCausalLM = _lm.InternLM2ForCausalLM

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator
    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


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


class InternVLChatModelDINO(PreTrainedModel):
    """
    InternVL2_5-8B with optional DINOv2 feature augmentation.

    Extra params vs original:
      dino_mode: "native_only" | "concat" | "interleaved"

    DINO projector:  LayerNorm(1024) → Linear(1024→llm_hidden) → GELU → Linear(llm_hidden→llm_hidden)
    """
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None,
                 language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')

        # ── Vision model (original) ──────────────────────────────────
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)

        # ── Language model (original) ────────────────────────────────
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                from transformers import LlamaForCausalLM as _LFC
                self.language_model = _LFC(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        # ── DINOv2 encoder + projector (NEW) ─────────────────────────
        self.dino_encoder = None   # lazy-loaded via _init_dino()
        self.dino_projector = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        # Scale knob for the DINO branch, used only by the "fused" mode
        # (native-anchored residual).  Keeps the residual small so the frozen LLM
        # keeps behaving like native.  Unused by native_only/concat/interleaved.
        self.dino_gate = nn.Parameter(torch.tensor([0.01]))
        # Optional explicit bound for newly trained residuals. None preserves
        # historical checkpoints. Apply the identical function in train/generate.
        self.dino_max_rms_ratio = None
        self.dino_mode = "native_only"  # "native_only" | "concat" | "interleaved" | "fused"

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    # ── DINO helpers ─────────────────────────────────────────────────
    def _init_dino(self, device):
        """Lazy-load DINOv2 encoder once. Does NOT reset projector (call reset_dino_projector() for that)."""
        if self.dino_encoder is not None:
            return
        dino = _get_dinov2()
        self.dino_encoder = dino.to(device).float().eval()
        for p in self.dino_encoder.parameters():
            p.requires_grad = False
        # Ensure projector is fp32 (for DINO's fp32 output) but DON'T reset weights
        self.dino_projector = self.dino_projector.float()
        print(f"  [DINO] dinov2_vitl14 loaded on {device} (fp32)")

    def reset_dino_projector(self):
        """Re-init projector for training. Call BEFORE optimizer init."""
        self.dino_projector = self.dino_projector.float()
        for m in self.dino_projector:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        nn.init.zeros_(self.dino_projector[-1].weight)
        nn.init.zeros_(self.dino_projector[-1].bias)
        self.dino_gate.data = self.dino_gate.data.float().fill_(0.01)
        print(f"  [DINO] Projector reset: all weights re-initialized, last layer zeroed, gate=0.01")

    def extract_dino_residual(self, pixel_values):
        """
        Extract DINOv2 patch features, spatial-pool to match native token count,
        project through dino_projector.  Returns the RAW projected residual (in
        llm_hidden_size), i.e. without the <IMG_CONTEXT> anchor added.

        Args:
            pixel_values: [B, 3, H, W]  (same images as native encoder)
        Returns:
            dino_residual: [B, num_image_token, llm_hidden_size], fp32
        """
        # DINO runs in fp32; convert input, then convert result back
        px_dtype = pixel_values.dtype
        pixel_values = pixel_values.float()

        # Resize to DINOv2 native resolution (336) if needed
        if pixel_values.shape[-1] != 336 or pixel_values.shape[-2] != 336:
            px = torch.nn.functional.interpolate(
                pixel_values, size=(336, 336), mode='bilinear', align_corners=False)
        else:
            px = pixel_values

        B = px.shape[0]
        # DINOv2 forward → [B, 576+1, 1024] or [B, 1024+1, 1024] depending on input size
        # For 336×336: 24×24 = 576 patches + CLS
        dino_out = self.dino_encoder.forward_features(px)
        dino_feats = dino_out['x_norm_patchtokens']  # [B, num_patches, 1024]

        # Spatial pool to match native token count
        num_dino = dino_feats.shape[1]       # e.g. 576 for 336×336
        native_target = self.num_image_token  # e.g. 256 for 448×448
        grid_size_dino = int(num_dino ** 0.5)  # 24
        target_grid = int(native_target ** 0.5)  # 16

        if grid_size_dino != target_grid:
            dino_feats = dino_feats.reshape(B, grid_size_dino, grid_size_dino, -1)
            dino_feats = dino_feats.permute(0, 3, 1, 2)  # [B, C, H, W]
            dino_feats = torch.nn.functional.adaptive_avg_pool2d(
                dino_feats, (target_grid, target_grid))
            dino_feats = dino_feats.permute(0, 2, 3, 1)  # [B, H, W, C]
            dino_feats = dino_feats.reshape(B, native_target, -1)
            # Remove spatial dims → [B, num_image_token, 1024]

        # Project and convert back to input dtype
        dino_residual = self.dino_projector(dino_feats)  # [B, N, llm_hidden], fp32
        return dino_residual

    def extract_dino_feature(self, pixel_values):
        """
        Legacy concat/interleaved DINO token embeddings: anchored on the
        <IMG_CONTEXT> word embedding, i.e. <IMG_CONTEXT>_emb + dino_residual.
        """
        px_dtype = pixel_values.dtype
        dino_residual = self.extract_dino_residual(pixel_values)

        # Add residual to IMG_CONTEXT token embedding (neutral base)
        img_ctx_emb = self.language_model.get_input_embeddings()(
            torch.tensor([self.img_context_token_id], device=dino_residual.device))
        dino_embeds = img_ctx_emb.unsqueeze(0).expand(
            dino_residual.shape[0], dino_residual.shape[1], -1) + dino_residual

        return dino_embeds.to(px_dtype)

    def fuse_dino_residual(self, native, residual):
        delta = self.dino_gate.float() * residual.float()
        if self.dino_max_rms_ratio is not None:
            native_rms = native.float().square().mean((-2, -1), keepdim=True).sqrt()
            # Clamp BEFORE sqrt: a zero-initialized projector otherwise has an
            # infinite sqrt derivative, even when a later clamp masks the norm.
            delta_rms = delta.square().mean((-2, -1), keepdim=True).clamp_min(1e-16).sqrt()
            scale = (float(self.dino_max_rms_ratio) * native_rms /
                     delta_rms.clamp_min(1e-8)).clamp(max=1.)
            delta = delta * scale
        # Preserve legacy BF16 multiply/add when the bound is disabled.
        if self.dino_max_rms_ratio is None:
            return native + self.dino_gate.to(native.dtype) * residual.to(native.dtype)
        return native + delta.to(native.dtype)

    def set_dino_mode(self, mode: str):
        """mode: "native_only" | "concat" | "interleaved" | "fused" """
        assert mode in ("native_only", "concat", "interleaved", "fused")
        self.dino_mode = mode

    # ── Original methods (unchanged) ─────────────────────────────────
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        # ── DINO augmentation (same as in generate) ──────────────────
        if self.dino_mode != "native_only":
            self._init_dino(pixel_values.device)
            if self.dino_mode == "fused":
                # Native-anchored gated residual: keep the native tokens, add a
                # small DINO residual on top.  No token doubling.
                dino_res = self.extract_dino_residual(pixel_values)
                dino_res = dino_res[image_flags == 1] if dino_res.shape[0] > 1 else dino_res
                vit_embeds = self.fuse_dino_residual(vit_embeds, dino_res)
            else:
                dino_embeds = self.extract_dino_feature(pixel_values)
                dino_embeds = dino_embeds[image_flags == 1] if dino_embeds.shape[0] > 1 else dino_embeds

                if self.dino_mode == "interleaved":
                    BN_d, N_d, C_d = vit_embeds.shape
                    merged = torch.empty(BN_d, 2 * N_d, C_d, dtype=vit_embeds.dtype,
                                         device=vit_embeds.device)
                    merged[:, 0::2, :] = vit_embeds
                    merged[:, 1::2, :] = dino_embeds
                    vit_embeds = merged
                elif self.dino_mode == "concat":
                    vit_embeds = torch.cat([vit_embeds, dino_embeds], dim=1)

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        # InternLM2ForCausalLM may return nested ModelOutputs — unwrap to tensor
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs[0]
        # Unwrap until we get a plain tensor
        while hasattr(logits, 'to_tuple') and not isinstance(logits, torch.Tensor):
            logits = logits.to_tuple()[0]

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if hasattr(outputs, 'past_key_values'):
            pkv, hs, attn = outputs.past_key_values, outputs.hidden_states, outputs.attentions
        else:
            pkv = outputs[1] if len(outputs) > 1 else None
            hs = outputs[2] if len(outputs) > 2 else None
            attn = outputs[3] if len(outputs) > 3 else None
        return CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=pkv, hidden_states=hs, attentions=attn,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config,
                   num_patches_list=None, history=None, return_history=False,
                   IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False,
                   image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        # Double image tokens for the extra-token modes (native_only/fused add none)
        img_token_count = self.num_image_token
        if self.dino_mode not in ("native_only", "fused"):
            img_token_count *= 2

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * img_token_count * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config,
             history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
             IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False):
        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        # Double image tokens for the extra-token modes (native_only/fused add none)
        img_token_count = self.num_image_token
        if self.dino_mode not in ("native_only", "fused"):
            img_token_count *= 2

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * img_token_count * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id

        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
            output_hidden_states=True,
        )
        llm_head_hidden_states = generation_output[2]
        all_probs = generation_output[1]
        all_input_down_proj = generation_output[3]
        generation_output = generation_output[0]
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history, all_probs, llm_head_hidden_states, all_input_down_proj
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response, all_probs, llm_head_hidden_states, all_input_down_proj

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            # Native visual features
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)

            # ── DINO augmentation ────────────────────────────────────
            if self.dino_mode != "native_only":
                self._init_dino(pixel_values.device)
                if self.dino_mode == "fused":
                    # Native-anchored gated residual: native evidence preserved,
                    # DINO adds a bounded perturbation.  No token doubling.
                    dino_res = self.extract_dino_residual(pixel_values)
                    vit_embeds = self.fuse_dino_residual(vit_embeds, dino_res)
                else:
                    dino_embeds = self.extract_dino_feature(pixel_values)

                    if self.dino_mode == "interleaved":
                        # [nat0, din0, nat1, din1, ...]
                        BN, N, C = vit_embeds.shape
                        merged = torch.empty(BN, 2 * N, C, dtype=vit_embeds.dtype,
                                             device=vit_embeds.device)
                        merged[:, 0::2, :] = vit_embeds
                        merged[:, 1::2, :] = dino_embeds
                        vit_embeds = merged
                    elif self.dino_mode == "concat":
                        # [nat0..nat_N-1, din0..din_N-1]
                        vit_embeds = torch.cat([vit_embeds, dino_embeds], dim=1)

            # Replace image placeholder tokens with visual embeddings
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids_flat = input_ids.reshape(B * N)
            selected = (input_ids_flat == self.img_context_token_id)
            assert selected.sum() != 0, (
                f"No <IMG_CONTEXT> tokens found. "
                f"Make sure DINO mode is set before calling chat(). "
                f"Current mode: {self.dino_mode}")
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
