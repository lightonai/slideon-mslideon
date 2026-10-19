"""Import raw SAE checkpoints without a dependency on the research repository."""
import json
import shutil
import tempfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file
from sentence_transformers import SparseEncoder

def from_pretrained(checkpoint, revision=None, **kwargs):
    source = Path(checkpoint)
    hub_kwargs = dict(token=kwargs.get("token"), cache_dir=kwargs.get("cache_folder"),
                      local_files_only=kwargs.get("local_files_only", False))
    config_path = source / "config.json" if source.is_dir() else Path(
        hf_hub_download(str(checkpoint), "config.json", revision=revision, **hub_kwargs))
    if json.loads(config_path.read_text()).get("model_type") == "slideon":
        return SparseEncoder(str(checkpoint), revision=revision, trust_remote_code=True, **kwargs)
    if not source.is_dir():
        source = Path(snapshot_download(str(checkpoint), revision=revision, **hub_kwargs))
    from .export import export_legacy_checkpoint
    with tempfile.TemporaryDirectory(prefix="slideon-export-") as directory:
        export_legacy_checkpoint(source, directory)
        return SparseEncoder(directory, trust_remote_code=True, **kwargs)


def from_sae(checkpoint, revision=None, backbone_revision=None, query_length=128,
             document_length=512):
    source = Path(checkpoint)
    if not source.is_dir():
        config_path = Path(hf_hub_download(str(checkpoint), "config.json", revision=revision))
    else:
        config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("model_type") == "slideon":
        if config.get("checkpoint_type") != "sae":
            raise ValueError("from_sae requires an SAE initialization checkpoint")
        if backbone_revision is not None and backbone_revision != config.get("base_model_revision"):
            raise ValueError("The flat SAE bundles its backbone; a different revision requires a raw SAE")
        model = from_pretrained(checkpoint, revision=revision)
        model[0].settings.update(query_length=query_length, document_length=document_length)
        model[0].max_seq_length = max(query_length, document_length)
        return model
    if not source.is_dir():
        source = Path(snapshot_download(checkpoint, revision=revision,
                      allow_patterns=["config.json", "sae.safetensors"]))
    if not config.get("normalized_inputs", True):
        raise ValueError("Only normalized-token SAEs are supported")
    base_revision = backbone_revision or config.get("base_model_revision")
    if base_revision is None and config["base_model"] == "lightonai/DenseOn":
        base_revision = "cb9947ebccb33862d24e3c7ca2edb25e51acd887"
    base = Path(snapshot_download(config["base_model"], revision=base_revision))
    state = load_file(str(source / "sae.safetensors"))
    state = {key: value for key, value in state.items()
             if key not in ("W_dec", "steps_since_fired")}
    state["gate_relu_phi"] = torch.ones(config["n_latents"])
    with tempfile.TemporaryDirectory(prefix="slideon-sae-") as directory:
        root = Path(directory)
        encoder = root / "encoder"
        shutil.copytree(base, encoder)
        modules = json.loads((encoder / "modules.json").read_text())
        modules = modules[:config.get("truncate_after", 0) + 1]
        if modules[0]["path"]:
            raise ValueError("Expected a backbone with its Transformer at the root")
        (encoder / "modules.json").write_text(json.dumps(modules))
        encoder_config = json.loads((encoder / "config_sentence_transformers.json").read_text())
        if config.get("backbone_only"):
            encoder_config.update(query_prefix="", document_prefix="", skiplist_words=[])
        (encoder / "config_sentence_transformers.json").write_text(json.dumps(encoder_config))
        settings = dict(token_k=config["k"], query_top_k=0, document_top_k=0,
                        query_length=query_length, document_length=document_length,
                        head_chunk_size=256, prompts={
                            "query": config.get("query_prompt", ""),
                            "document": config.get("document_prompt", "")})
        (root / "slideon_config.json").write_text(json.dumps(settings))
        (root / "config.json").write_text(json.dumps(dict(
            config, pooling="max_relu", score="dot", sae_source=str(checkpoint),
            sae_revision=revision, resolved_backbone_revision=base_revision or base.name)))
        save_file(state, str(root / "sparse_head.safetensors"))
        from .export import export_legacy_checkpoint
        with tempfile.TemporaryDirectory(prefix="slideon-export-") as flat:
            export_legacy_checkpoint(root, flat, sae_checkpoint=source)
            return SparseEncoder(flat, trust_remote_code=True)
