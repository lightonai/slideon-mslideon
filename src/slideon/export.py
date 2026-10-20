"""Convert legacy encoder/head directories to a self-contained flat checkpoint."""
import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from .modeling_slideon import SlideOnConfig


def export_legacy_checkpoint(source, output, sae_checkpoint=None):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Export into a separate directory before replacing the source")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("The export directory must be empty")
    metadata_path = source / "retrieval_config.json"
    metadata = json.loads((metadata_path if metadata_path.exists() else source / "config.json").read_text())
    settings = json.loads((source / "slideon_config.json").read_text())
    # Translate historical checkpoint metadata; the public API uses max_active_dims.
    for task in ("query", "document"):
        legacy_cap = settings.pop(task + "_top_k", None)
        settings.setdefault(task + "_max_active_dims", legacy_cap or None)
    settings.update(pooling=metadata["pooling"], score=metadata["score"])
    encoder = source / "encoder"
    tokenizer = AutoTokenizer.from_pretrained(encoder, local_files_only=True)
    encoder_config = json.loads((encoder / "config_sentence_transformers.json").read_text())
    modules = json.loads((encoder / "modules.json").read_text())
    state = {"backbone." + k: v for k, v in load_file(str(encoder / "model.safetensors")).items()}
    projection_configs = []
    for index, entry in enumerate(modules[1:]):
        folder = encoder / entry["path"]
        projection_configs.append(json.loads((folder / "config.json").read_text()))
        state.update({f"projections.{index}." + k: v
                      for k, v in load_file(str(folder / "model.safetensors")).items()})
    head = load_file(str(source / "sparse_head.safetensors"))
    sae_path = Path(sae_checkpoint or source) / "sae.safetensors"
    if sae_path.exists():
        sae = load_file(str(sae_path))
        for key in head.keys() & sae.keys():
            if not torch.equal(head[key], sae[key]):
                raise ValueError(f"Raw SAE and inference head disagree: {key}")
        head.update(sae)
    head_config = dict(d_model=head["b_dec"].numel(), n_latents=head["b_enc"].numel(),
                       d_lift=head["lift.weight"].shape[0] if "lift.weight" in head else 0,
                       has_decoder="W_dec" in head)
    if head_config["d_lift"]:
        head_config["hidden_dim"] = head["mlp.0.weight"].shape[0]
    state.update({"head." + k: v for k, v in head.items()})
    provenance = {k: metadata[k] for k in ("base_model", "sae_path", "sae_source", "sae_revision") if k in metadata}
    provenance["base_model_revision"] = metadata.get("base_model_revision") or metadata.get("model_revision") or metadata.get("resolved_backbone_revision")
    config = SlideOnConfig(
        backbone_config=json.loads((encoder / "config.json").read_text()),
        projection_configs=projection_configs, head_config=head_config,
        inference=settings,
        tokenizer_settings={k: encoder_config[k] for k in
                            ("query_prefix", "document_prefix", "skiplist_words") if k in encoder_config},
        skip_token_ids=tokenizer.convert_tokens_to_ids(encoder_config.get("skiplist_words", [])),
        checkpoint_type="sae" if head_config["has_decoder"] else "retrieval",
        architectures=["SlideOnModel"],
        auto_map={"AutoConfig": "modeling_slideon.SlideOnConfig", "AutoModel": "modeling_slideon.SlideOnModel"},
        **provenance)
    config.save_pretrained(output)
    save_file(state, str(output / "model.safetensors"), metadata={"format": "pt"})
    tokenizer.save_pretrained(output)
    code = Path(__file__).parent
    shutil.copy2(code / "modeling_slideon.py", output / "modeling_slideon.py")
    shutil.copy2(code / "models.py", output / "slideon.py")
    (output / "modules.json").write_text(json.dumps([
        dict(idx=0, name="0", path="", type="slideon.SlideOn")], indent=2) + "\n")
    st_config = source / "config_sentence_transformers.json"
    st = json.loads(st_config.read_text()) if st_config.exists() else dict(
        model_type="SparseEncoder", prompts=settings["prompts"],
        default_prompt_name="document", similarity_fn_name="dot")
    (output / "config_sentence_transformers.json").write_text(json.dumps(st, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    export_legacy_checkpoint(args.source, args.output)
