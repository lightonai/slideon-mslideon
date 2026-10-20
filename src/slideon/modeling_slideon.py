"""Complete SlideOn model: Transformer, token projections, and sparse latent head."""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel


class SlideOnConfig(PretrainedConfig):
    model_type = "slideon"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(self, backbone_config=None, projection_configs=None, head_config=None,
                 inference=None, tokenizer_settings=None, skip_token_ids=None,
                 checkpoint_type="retrieval", **kwargs):
        super().__init__(**kwargs)
        if isinstance(backbone_config, dict):
            backbone_config = dict(backbone_config)
            model_type = backbone_config.pop("model_type", "modernbert")
            backbone_config = AutoConfig.for_model(model_type, **backbone_config)
        self.backbone_config = backbone_config or AutoConfig.for_model("modernbert")
        self.projection_configs = projection_configs or []
        self.head_config = head_config or dict(d_model=768, n_latents=32768, d_lift=0)
        self.inference = inference or dict(
            token_k=32, query_max_active_dims=None, document_max_active_dims=None,
            query_length=8192, document_length=8192, head_chunk_size=2048,
            prompts={"query": "", "document": ""}, pooling="max_relu", score="dot")
        self.tokenizer_settings = tokenizer_settings or {}
        self.skip_token_ids = skip_token_ids or []
        self.checkpoint_type = checkpoint_type
        if self.inference.get("pooling") != "max_relu" or self.inference.get("score") != "dot":
            raise ValueError("SlideOn supports max_relu pooling and dot scoring")


class TokenProjection(torch.nn.Module):
    """The token-wise residual Dense layers used by mSlideOn."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = torch.nn.Linear(
            config["in_features"], config["out_features"], bias=config["bias"]
        )
        self.use_residual = config.get("use_residual", False)
        self.residual = None
        if self.use_residual and config["in_features"] != config["out_features"]:
            self.residual = torch.nn.Linear(
                config["in_features"], config["out_features"], bias=False
            )
        if config["activation_function"] != "torch.nn.modules.linear.Identity":
            raise ValueError("Unsupported projection activation")

    def forward(self, tokens):
        projected = self.linear(tokens)
        if self.use_residual:
            projected = projected + (
                self.residual(tokens) if self.residual is not None else tokens
            )
        return projected


class LatentHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model, n_latents = config["d_model"], config["n_latents"]
        d_lift = config.get("d_lift") or 0
        self.W_enc = torch.nn.Parameter(torch.empty(d_lift or d_model, n_latents))
        self.b_enc = torch.nn.Parameter(torch.zeros(n_latents))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_model))
        self.gate_relu_phi = torch.nn.Parameter(torch.ones(n_latents))
        self.lift = None
        if d_lift:
            hidden = config["hidden_dim"]
            self.lift = torch.nn.Linear(d_model, d_lift, bias=False)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d_lift, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, d_lift))
        if config.get("has_decoder", False):
            # Preserve the pretrained SAE decoder without optimizing it in retrieval SFT.
            self.W_dec = torch.nn.Parameter(torch.empty(n_latents, d_model), requires_grad=False)
            self.register_buffer("steps_since_fired", torch.zeros(n_latents, dtype=torch.long))

    def forward(self, tokens):
        centered = tokens - self.b_dec
        if self.lift is not None:
            lifted = self.lift(centered)
            centered = lifted + self.mlp(lifted)
        return centered @ self.W_enc + self.b_enc


class SlideOnModel(PreTrainedModel):
    config_class = SlideOnConfig
    base_model_prefix = "backbone"
    _supports_sdpa = True
    _supports_flash_attn = True
    _no_split_modules = ["ModernBertEncoderLayer", "LatentHead", "TokenProjection"]
    _keep_in_fp32_modules_strict = ["head"]

    def __init__(self, config):
        super().__init__(config)
        config.backbone_config._attn_implementation = config._attn_implementation
        self.backbone = AutoModel.from_config(config.backbone_config)
        self.projections = torch.nn.ModuleList(
            TokenProjection(item) for item in config.projection_configs)
        self.head = LatentHead(config.head_config)
        self.settings = config.inference
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, LatentHead):
            torch.nn.init.normal_(module.W_enc, std=0.02)
            if hasattr(module, "W_dec"):
                torch.nn.init.normal_(module.W_dec, std=0.02)
        else:
            super()._init_weights(module)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        result = super().from_pretrained(*args, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        # Match the original PyLate encoder's RoPE buffer precision.
        model.backbone.to(dtype=model.backbone.dtype)
        # Loading tensors can reset the decoder's requires_grad flag.
        if hasattr(model.head, "W_dec"):
            model.head.W_dec.requires_grad_(False)
        return result

    def get_embedding_dimension(self):
        return self.config.head_config["n_latents"]

    @staticmethod
    def _task(task):
        task = task or "document"
        if task not in ("query", "document"):
            raise ValueError("task must be 'query' or 'document'")
        return task


    def forward(self, input_ids, attention_mask, token_type_ids=None, task=None,
                slideon_task=None, max_active_dims=None, token_k=None, **kwargs):
        if "top_k" in kwargs:
            raise TypeError("top_k has been removed; use max_active_dims (None for unpruned)")
        features = {"input_ids": input_ids, "attention_mask": attention_mask}
        if slideon_task is not None or task is not None:
            features["slideon_task"] = slideon_task or task
        if token_type_ids is not None:
            features["token_type_ids"] = token_type_ids
        task = self._task(task or features.get("slideon_task"))
        if features.get("slideon_task", task) != task:
            raise ValueError("Preprocessing and forward tasks must match")
        n_latents = self.get_embedding_dimension()
        cap = self.settings.get(task + "_max_active_dims") if max_active_dims is None else max_active_dims
        k = self.settings["token_k"] if token_k is None else token_k
        if cap is not None and (type(cap) is not int or cap <= 0):
            raise ValueError("max_active_dims must be a positive integer or None")
        if not isinstance(k, int) or not 1 <= k <= n_latents:
            raise ValueError(f"token_k must be an integer between 1 and {n_latents}")
        model_inputs = {
            key: features[key]
            for key in ("input_ids", "attention_mask", "token_type_ids")
            if key in features
        }
        tokens = self.backbone(**model_inputs).last_hidden_state
        for projection in self.projections:
            tokens = projection(tokens)
        mask = features["attention_mask"].bool()
        if task == "document" and self.config.skip_token_ids:
            skip_ids = torch.tensor(self.config.skip_token_ids, device=input_ids.device)
            mask = mask & ~torch.isin(input_ids, skip_ids)
        owners = torch.arange(tokens.shape[0], device=tokens.device)[:, None].expand_as(
            mask
        )[mask]
        tokens = torch.nn.functional.normalize(tokens[mask], p=2, dim=-1)
        # Match the evaluation loader's FP16 token transfer before head pooling.
        if not self.training:
            tokens = tokens.half().float()
        probability_sum = tokens.new_zeros(n_latents, dtype=torch.float32)
        usage_sum = tokens.new_zeros(n_latents, dtype=torch.float32)
        pooled = torch.zeros(
            mask.shape[0] * n_latents, dtype=torch.float32, device=tokens.device
        )
        for start in range(0, tokens.shape[0], self.settings["head_chunk_size"]):
            end = start + self.settings["head_chunk_size"]
            # The reference evaluates the head under BF16 autocast on CUDA.
            def head_chunk(chunk):
                with torch.autocast(chunk.device.type, dtype=torch.bfloat16, enabled=chunk.is_cuda):
                    activations = self.head(chunk)
                    values, indices = activations.topk(k, dim=-1)
                probabilities = activations.float().softmax(-1).sum(0) if self.training else chunk.new_zeros(n_latents)
                return values, indices, probabilities
            if self.training and torch.is_grad_enabled():
                values, indices, probabilities = activation_checkpoint(head_chunk, tokens[start:end], use_reentrant=False)
            else:
                values, indices, probabilities = head_chunk(tokens[start:end])
            if self.training:
                probability_sum = probability_sum + probabilities
                usage_sum = usage_sum + torch.bincount(indices.flatten(), minlength=n_latents)
            values = values.float().relu()
            indices = owners[start:end, None] * n_latents + indices
            pooled = pooled.scatter_reduce(
                0, indices.flatten(), values.flatten(), reduce="amax", include_self=True
            )
        pooled = (
            pooled.view(mask.shape[0], n_latents)
            * self.head.gate_relu_phi.float().relu()
        )
        if cap is not None:
            # Prune before FP16 rounding
            values, indices = pooled.topk(min(cap, n_latents), dim=-1)
            pooled = torch.zeros_like(pooled).scatter(1, indices, values)
        # The reference stores retrieval weights in FP16; retain that rounding.
        features["sentence_embedding"] = pooled if self.training else pooled.half().float()
        if self.training:
            count = max(tokens.shape[0], 1)
            features["load_balance"] = n_latents * (
                probability_sum / count * usage_sum.detach() / (count * k)
            ).sum()
        return {key: features[key] for key in ("sentence_embedding", "load_balance") if key in features}


SlideOnConfig.register_for_auto_class()
SlideOnModel.register_for_auto_class("AutoModel")
