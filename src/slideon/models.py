"""Sentence Transformers adapter for the complete, flat SlideOn checkpoint."""
from pathlib import Path

import torch
from sentence_transformers.base.modules import InputModule
from transformers import AutoConfig, AutoModel, AutoTokenizer


class SlideOn(InputModule):
    config_file_name = "config.json"
    forward_kwargs = {"task", "top_k", "token_k"}
    save_in_root = True

    def __init__(self, checkpoint, model_kwargs=None, processor_kwargs=None,
                 config_kwargs=None, **hub_kwargs):
        super().__init__()
        config = AutoConfig.from_pretrained(
            checkpoint, trust_remote_code=True, **hub_kwargs, **(config_kwargs or {}))
        if config.model_type != "slideon":
            raise ValueError("Expected a flat SlideOn checkpoint; use slideon.export for legacy checkpoints")
        self.model = AutoModel.from_pretrained(
            checkpoint, config=config, trust_remote_code=True,
            **hub_kwargs, **(model_kwargs or {}))
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True, **hub_kwargs, **(processor_kwargs or {}))
        if self.tokenizer.mask_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.mask_token_id
        elif self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.settings = self.model.config.inference
        self.encoder_config = self.model.config.tokenizer_settings
        self.max_seq_length = max(self.settings["query_length"], self.settings["document_length"])

    def get_embedding_dimension(self):
        return self.model.get_embedding_dimension()

    @staticmethod
    def _task(task):
        task = task or "document"
        if task not in ("query", "document"):
            raise ValueError("task must be 'query' or 'document'")
        return task

    def preprocess(
        self, inputs, prompt=None, task=None, processing_kwargs=None, **kwargs
    ):
        task = self._task(task)
        if not all(isinstance(text, str) for text in inputs):
            raise TypeError("SlideOn accepts text strings only")
        if prompt is None:
            prompt = self.settings["prompts"][task]
        texts = [(prompt + text).strip() for text in inputs]
        prefix = self.encoder_config.get(task + "_prefix", "")
        length = min(self.max_seq_length, self.settings[task + "_length"])
        options = dict(
            padding=True,
            truncation=True,
            max_length=length - bool(prefix),
            return_tensors="pt",
        )
        options.update(processing_kwargs or {})
        features = dict(self.tokenizer(texts, **options))
        if prefix:
            prefix_id = self.tokenizer.convert_tokens_to_ids(prefix)
            for name, value in list(features.items()):
                fill = (
                    prefix_id
                    if name == "input_ids"
                    else (1 if name == "attention_mask" else 0)
                )
                marker = torch.full(
                    (value.shape[0], 1), fill, dtype=value.dtype, device=value.device
                )
                features[name] = torch.cat((value[:, :1], marker, value[:, 1:]), dim=1)
        features["slideon_task"] = task
        return features

    def forward(self, features, task=None, top_k=None, token_k=None, **kwargs):
        inputs = {key: features[key] for key in
                  ("input_ids", "attention_mask", "token_type_ids", "slideon_task") if key in features}
        features.update(self.model(**inputs, task=task, top_k=top_k, token_k=token_k))
        return features

    def save(self, output_path, safe_serialization=True, **kwargs):
        self.model.save_pretrained(output_path, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(output_path)

    @classmethod
    def load(cls, model_name_or_path, subfolder="", token=None, cache_folder=None,
             revision=None, local_files_only=False, model_kwargs=None,
             processor_kwargs=None, config_kwargs=None, backend="torch", **kwargs):
        if backend != "torch":
            raise ValueError("SlideOn currently supports the torch backend only")
        return cls(model_name_or_path, model_kwargs, processor_kwargs, config_kwargs,
                   subfolder=subfolder, token=token, cache_dir=cache_folder,
                   revision=revision, local_files_only=local_files_only)
