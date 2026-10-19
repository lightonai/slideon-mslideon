"""Pretrain mSlideOn-SAE: frozen token preparation, model, loss, and training loop."""
import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from slideon.modeling_slideon import TokenProjection
from slideon.models import SlideOn


@dataclass(frozen=True)
class Recipe:
    name: str
    base_model: str
    base_model_revision: str
    d_model: int
    d_lift: int
    aux_k: int
    aux_coef: float
    balance_weight: float
    splits: tuple[str, ...]
    max_steps: int | None = None
    n_latents: int = 32768
    k: int = 32
    batch_size: int = 4096
    lr: float = 3e-4
    epochs: int = 3
    warmup_ratio: float = 0.05
    dead_steps: int = 2500
    seed: int = 42
    hidden_mult: int = 2
    balance_temperature: float = 1.0
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    schedule_token_fraction: float = 0.995
    bf16: bool = True
    query_length: int = 8192
    document_length: int = 8192
    query_prompt: str = "query: "
    document_prompt: str = "document: "
    dataset_prefix: str = "lightonai/embeddings-fine-tuning-filtered-"
    nv_threshold: float = 0.95
    num_negatives: int = 10
    sampled_negatives: int = 7
    init_docs_per_split: int = 128
    w_init: str = "data"

    def __post_init__(self):
        positive = ("d_model", "n_latents", "k", "batch_size", "epochs", "hidden_mult",
                    "query_length", "document_length", "num_negatives", "init_docs_per_split")
        for key in positive:
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("d_lift", "aux_k", "dead_steps", "sampled_negatives", "seed"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in ("lr", "aux_coef", "balance_weight", "balance_temperature", "weight_decay",
                    "adam_beta1", "adam_beta2", "adam_epsilon", "warmup_ratio",
                    "schedule_token_fraction", "nv_threshold"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
        if self.k > self.n_latents or self.sampled_negatives > self.num_negatives:
            raise ValueError("Invalid token top-k or sampled-negative count")
        if not 0 < self.lr or not 0 < self.balance_temperature or not 0 < self.adam_epsilon:
            raise ValueError("Learning rate, balance temperature, and Adam epsilon must be positive")
        if not (0 <= self.warmup_ratio < 1 and 0 < self.schedule_token_fraction <= 1):
            raise ValueError("Invalid warmup ratio or scheduled-token fraction")
        if not (0 <= self.adam_beta1 < 1 and 0 <= self.adam_beta2 < 1):
            raise ValueError("Adam betas must be in [0, 1)")
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps < 1):
            raise ValueError("max_steps must be null or a positive integer")
        if type(self.bf16) is not bool or self.w_init not in ("data", "data_centered"):
            raise ValueError("Invalid precision or SAE initialization")
        if not self.splits or any(not isinstance(split, str) or not split for split in self.splits):
            raise ValueError("splits must contain dataset split names")

    def checkpoint_config(self):
        return dict(asdict(self), normalized_inputs=True, backbone_only=not self.d_lift,
                    truncate_after=3 if self.d_lift else 0,
                    b_dec_init="mean", n_layers=1,
                    encoder="lift-residual-mlp" if self.d_lift else "linear")


RECIPE_KEYS = {field.name for field in fields(Recipe)} - {"name"}


def model_name(name):
    if name not in ("slideon", "mslideon"):
        raise ValueError(f"Unknown recipe: {name}")
    return "mSlideOn-SAE" if name == "mslideon" else "SlideOn-SAE"


def recipe_from_parameters(name, parameters):
    missing = RECIPE_KEYS - set(parameters)
    if missing:
        raise ValueError(f"Missing SAE parameters: {', '.join(sorted(missing))}")
    values = {key: parameters[key] for key in RECIPE_KEYS}
    if not isinstance(values["splits"], (list, tuple)):
        raise ValueError("splits must be a list")
    values["splits"] = tuple(values["splits"])
    return Recipe(name=name, **values)


class TopKSAE(nn.Module):
    def __init__(self, recipe):
        super().__init__()
        self.recipe = recipe
        decoder = F.normalize(torch.randn(recipe.n_latents, recipe.d_model), dim=-1)
        self.W_dec = nn.Parameter(decoder)
        if recipe.d_lift:
            self.lift = nn.Linear(recipe.d_model, recipe.d_lift, bias=False)
            nn.init.orthogonal_(self.lift.weight)
            self.mlp = nn.Sequential(nn.Linear(recipe.d_lift, recipe.hidden_mult * recipe.d_lift), nn.ReLU(),
                                     nn.Linear(recipe.hidden_mult * recipe.d_lift, recipe.d_lift))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
            encoder = self.lift(decoder).detach().T.contiguous()
        else:
            encoder = decoder.T.contiguous().clone()
        self.W_enc = nn.Parameter(encoder)
        self.b_enc = nn.Parameter(torch.zeros(recipe.n_latents))
        self.b_dec = nn.Parameter(torch.zeros(recipe.d_model))
        self.register_buffer("steps_since_fired", torch.zeros(recipe.n_latents, dtype=torch.long))

    @torch.no_grad()
    def initialize(self, document_tokens, rng=None):
        rng = rng or random.Random(self.recipe.seed)
        tokens = document_tokens.float()
        if self.recipe.w_init == "data_centered":
            tokens = tokens - tokens.mean(0)
        indices = (rng.sample(range(len(tokens)), self.recipe.n_latents) if len(tokens) >= self.recipe.n_latents
                   else rng.choices(range(len(tokens)), k=self.recipe.n_latents))
        atoms = F.normalize(tokens[indices], dim=-1).to(self.W_dec.device)
        self.W_dec.copy_(atoms)
        self.W_enc.copy_((self.lift(atoms) if self.recipe.d_lift else atoms).T)

    def forward(self, x):
        centered = x - self.b_dec
        if self.recipe.d_lift:
            lifted = self.lift(centered)
            centered = lifted + self.mlp(lifted)
        pre = centered @ self.W_enc + self.b_enc
        values, indices = pre.topk(self.recipe.k, dim=-1)
        values = values.relu()
        recon = (values.unsqueeze(-1) * self.W_dec[indices]).sum(1) + self.b_dec
        mse = (recon.float() - x).square().mean()
        aux, balance = mse.new_zeros(()), mse.new_zeros(())
        dead = self.steps_since_fired > self.recipe.dead_steps
        if self.recipe.aux_k and dead.any():
            count = min(self.recipe.aux_k, int(dead.sum()))
            av, ai = pre.masked_fill(~dead, -torch.inf).topk(count, dim=-1)
            # Chunking bounds the B x aux_k x d_model temporary without changing the sum.
            residual_hat = sum((av[:, start:start+32].relu().unsqueeze(-1)
                               * self.W_dec[ai[:, start:start+32]]).sum(1)
                               for start in range(0, count, 32))
            aux = (residual_hat.float() - (x - recon.float()).detach()).square().mean()
        if self.recipe.balance_weight:
            counts = torch.bincount(indices.detach().flatten(), minlength=self.recipe.n_latents).float()
            if dist.is_initialized():
                dist.all_reduce(counts)
            frequency = counts / counts.sum().clamp_min(1)
            probabilities = (pre.float() / self.recipe.balance_temperature).softmax(-1).mean(0)
            balance = self.recipe.n_latents * (frequency * probabilities).sum()
        loss = mse + self.recipe.aux_coef * aux + self.recipe.balance_weight * balance
        return loss, recon, values, indices, mse.detach(), aux.detach(), balance.detach()

    @torch.no_grad()
    def project_decoder_gradient(self):
        if self.W_dec.grad is not None:
            self.W_dec.grad.sub_((self.W_dec.grad * self.W_dec).sum(-1, keepdim=True) * self.W_dec)

    @torch.no_grad()
    def after_step(self, values, indices):
        self.W_dec.copy_(F.normalize(self.W_dec, dim=-1))
        fired = torch.zeros_like(self.steps_since_fired)
        fired[indices[values > 0].unique()] = 1
        if dist.is_initialized():
            dist.all_reduce(fired, op=dist.ReduceOp.MAX)
        self.steps_since_fired.add_(1)
        self.steps_since_fired[fired.bool()] = 0


def cache_recipe(config):
    """Only token-producing settings determine cache compatibility."""
    keys = ("name", "base_model", "base_model_revision", "d_model", "d_lift", "bf16",
            "query_length", "document_length", "query_prompt", "document_prompt",
            "dataset_prefix", "nv_threshold", "num_negatives", "sampled_negatives", "seed", "splits")
    return json.loads(json.dumps({key: config.get(key) for key in keys}))


class FrozenTokens(torch.nn.Module):
    preprocess = SlideOn.preprocess
    _task = staticmethod(SlideOn._task)

    def __init__(self, recipe, device):
        super().__init__()
        root = Path(snapshot_download(recipe.base_model, revision=recipe.base_model_revision))
        self.encoder_config = json.loads((root / "config_sentence_transformers.json").read_text())
        if not recipe.d_lift:
            self.encoder_config.update(query_prefix="", document_prefix="", skiplist_words=[])
        self.max_seq_length = max(recipe.query_length, recipe.document_length)
        self.settings = dict(query_length=recipe.query_length, document_length=recipe.document_length,
                             prompts={"query": recipe.query_prompt, "document": recipe.document_prompt})
        self.tokenizer = AutoTokenizer.from_pretrained(root)
        if self.tokenizer.mask_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.mask_token_id
        elif self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.encoder = AutoModel.from_pretrained(root, dtype=torch.bfloat16 if recipe.bf16 and device.type == "cuda" else torch.float32)
        self.encoder.to(dtype=self.encoder.dtype)
        self.projections = torch.nn.ModuleList()
        if recipe.d_lift:
            for entry in json.loads((root / "modules.json").read_text())[1:4]:
                folder = root / entry["path"]
                projection = TokenProjection(json.loads((folder / "config.json").read_text()))
                projection.load_state_dict(load_file(str(folder / "model.safetensors")))
                self.projections.append(projection.to(dtype=self.encoder.dtype))
        self.to(device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, texts, task):
        features = self.preprocess(texts, task=task)
        features = {k: v.to(self.encoder.device) for k, v in features.items() if isinstance(v, torch.Tensor)}
        tokens = self.encoder(**features).last_hidden_state
        for projection in self.projections:
            tokens = projection(tokens)
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
        mask = features["attention_mask"].bool()
        if task == "document" and self.encoder_config.get("skiplist_words"):
            skip = torch.tensor(self.tokenizer.convert_tokens_to_ids(self.encoder_config["skiplist_words"]), device=tokens.device)
            mask &= ~torch.isin(features["input_ids"], skip)
        return [row[keep].half().cpu().numpy() for row, keep in zip(tokens, mask)]


def dataset_repo(recipe, split):
    prefix = recipe.dataset_prefix
    if recipe.name == "slideon":
        return prefix + "en"
    if split.startswith("CodeEditSearch"):
        return prefix + "code-edit"
    if split.startswith("CodeSearchNet") or split in ("apps", "synthetictext2sql", "cosqa", "codefeedbackst",
                                                      "codefeedbackmt", "stackoverflowqa", "codetranscontest", "codetransdl"):
        return prefix + "code"
    return prefix + split.rsplit("_", 1)[1]


def filter_texts(scores, queries, documents, seed=42, nv_threshold=0.95,
                 num_negatives=10, sampled_negatives=7):
    """Filter negatives, sample the configured count, and deduplicate text IDs."""
    docs = dict(zip(documents["document_id"], documents["document"]))
    query_text = dict(zip(queries["query_id"], queries["query"]))
    kept_docs, kept_queries = {}, {}
    rng = random.Random(seed)
    for row in scores:
        ids, scores_, qid = row["document_ids"], row["scores"], row["query_id"]
        negatives = [did for did, score in zip(ids[1:], scores_[1:]) if score < nv_threshold * scores_[0] and docs.get(did)][:num_negatives]
        if len(negatives) < num_negatives or not query_text.get(qid) or not docs.get(ids[0]):
            continue
        kept_queries[qid] = query_text[qid]
        kept_docs[ids[0]] = docs[ids[0]]
        for did in rng.sample(negatives, sampled_negatives):
            kept_docs[did] = docs[did]
    return dict(documents=list(kept_docs.values()), queries=list(kept_queries.values()))


def prepare_tokens(recipe, output, encode_batch_size=4, dataset_revisions=None):
    if encode_batch_size < 1:
        raise ValueError("encode_batch_size must be positive")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    recipe_config = recipe.checkpoint_config()
    pins = (json.loads(Path(dataset_revisions).read_text()) if isinstance(dataset_revisions, (str, Path))
            else dict(dataset_revisions or {}))
    manifest_file = root / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        if cache_recipe(manifest["recipe"]) != cache_recipe(recipe_config):
            raise ValueError("Cache recipe mismatch")
        if pins:
            for repo, revision in pins.items():
                if HfApi().dataset_info(repo, revision=revision).sha != manifest["dataset_revisions"].get(repo):
                    raise ValueError(f"Dataset revision differs from the existing cache: {repo}")
    else:
        repos = sorted({dataset_repo(recipe, split) for split in recipe.splits})
        revisions = {repo: HfApi().dataset_info(repo, revision=pins.get(repo)).sha for repo in repos}
        manifest = dict(recipe=recipe_config, dataset_revisions=revisions, sources=[], complete=False)
        manifest_file.write_text(json.dumps(manifest, indent=2))
    encoder = FrozenTokens(recipe, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for split in recipe.splits:
        if all(any(s["split"] == split and s["kind"] == kind for s in manifest["sources"])
               for kind in ("documents", "queries")):
            continue
        repo = dataset_repo(recipe, split)
        tables = {}
        for kind in ("scores", "queries", "documents"):
            kwargs = dict(name=kind, revision=manifest["dataset_revisions"][repo])
            if recipe.name == "mslideon":
                kwargs.update(data_files=f"{kind}/{split}-*", split="train", verification_mode="no_checks")
            else:
                kwargs["split"] = split
            tables[kind] = load_dataset(repo, **kwargs)
        pools = filter_texts(**tables, seed=recipe.seed, nv_threshold=recipe.nv_threshold,
                            num_negatives=recipe.num_negatives, sampled_negatives=recipe.sampled_negatives)
        for kind, texts in pools.items():
            if any(s["split"] == split and s["kind"] == kind for s in manifest["sources"]):
                continue
            if not texts:
                raise ValueError(f"No usable texts in {split}/{kind}")
            name = f"{split}-{kind}"
            offsets = [0]
            scheduled_tokens = 0
            temporary = root / (name + ".bin.partial")
            with temporary.open("wb") as stream:
                for start in range(0, len(texts), encode_batch_size):
                    batch = texts[start:start+encode_batch_size]
                    # Historical schedules count raw text, before prompts/markers/skip lists.
                    scheduled_tokens += sum(len(ids) for ids in encoder.tokenizer(
                        batch, truncation=True, max_length=recipe.query_length if kind == "queries" else recipe.document_length)["input_ids"])
                    arrays = encoder.encode(batch, "query" if kind == "queries" else "document")
                    for array in arrays:
                        if array.shape[1] != recipe.d_model or not np.isfinite(array).all():
                            raise ValueError("Invalid frozen token embeddings")
                        stream.write(array.tobytes())
                        offsets.append(offsets[-1] + len(array))
            temporary.replace(root / (name + ".bin"))
            np.save(root / (name + "-offsets.npy"), np.array(offsets, dtype=np.int64))
            np.save(root / (name + "-chars.npy"), np.array([len(t) for t in texts], dtype=np.int64))
            manifest["sources"].append(dict(name=name, split=split, kind=kind, rows=len(texts),
                                            tokens=offsets[-1], scheduled_tokens=scheduled_tokens))
            tmp = root / "manifest.json.partial"
            tmp.write_text(json.dumps(manifest, indent=2))
            tmp.replace(manifest_file)
            print(f"Cached {split}/{kind}: {len(texts):,} texts, {offsets[-1]:,} tokens", flush=True)
    manifest["complete"] = True
    temporary = root / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(manifest_file)


class TokenCache:
    def __init__(self, root, recipe):
        self.root = Path(root)
        self.recipe = recipe
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if not self.manifest.get("complete") or cache_recipe(self.manifest["recipe"]) != cache_recipe(recipe.checkpoint_config()):
            raise ValueError("Token cache is incomplete or belongs to a different recipe")
        self.sources = self.manifest["sources"]
        self.arrays, self.offsets, self.chars = [], [], []
        for source in self.sources:
            name = source["name"]
            path = self.root / (name + ".bin")
            if path.stat().st_size != source["tokens"] * recipe.d_model * 2:
                raise ValueError(f"Invalid cache size: {path}")
            self.arrays.append(np.memmap(path, mode="r", dtype=np.float16, shape=(source["tokens"], recipe.d_model)))
            self.offsets.append(np.load(self.root / (name + "-offsets.npy"), mmap_mode="r"))
            self.chars.append(np.load(self.root / (name + "-chars.npy"), mmap_mode="r"))
            offsets = self.offsets[-1]
            if (len(offsets) != source["rows"] + 1 or offsets[0] != 0 or offsets[-1] != source["tokens"]
                    or np.any(np.diff(offsets) <= 0) or len(self.chars[-1]) != source["rows"]):
                raise ValueError(f"Invalid cache offsets: {name}")
        self.ends = np.cumsum([s["rows"] for s in self.sources])
        self.total_tokens = sum(s["tokens"] for s in self.sources)
        self.scheduled_tokens = sum(s["scheduled_tokens"] for s in self.sources)

    def location(self, index):
        part = int(np.searchsorted(self.ends, index, side="right"))
        return part, int(index - (self.ends[part-1] if part else 0))

    def row(self, part, index):
        return self.arrays[part][self.offsets[part][index]:self.offsets[part][index+1]]

    def initialization_tokens(self, rng):
        arrays = []
        for part, source in enumerate(self.sources):
            if source["kind"] == "documents":
                arrays.extend(self.row(part, i) for i in rng.sample(range(source["rows"]), min(self.recipe.init_docs_per_split, source["rows"])))
        return torch.from_numpy(np.concatenate(arrays))


class TokenBatches:
    """Reshuffle documents per pass, then shuffle tokens within each rank's buffer.

    Cursor and current document references suffice to restore a buffer without
    putting gigabytes of token embeddings in every training checkpoint.
    """
    def __init__(self, cache, recipe, rank, world, device, buffer_tokens=4_000_000, state=None):
        self.cache, self.recipe = cache, recipe
        self.rank, self.world, self.device = rank, world, device
        self.batch_size = recipe.batch_size // world
        self.buffer_tokens = buffer_tokens
        self.state = state or dict(epoch=0, cursor=0, fill=-1, references=[], batch=0)
        self.buffer = None
        self.order_epoch = -1

    def _next_document(self):
        while True:
            epoch = self.state["epoch"]
            if self.order_epoch != epoch:
                self.order = np.random.default_rng(self.recipe.seed + epoch).permutation(int(self.cache.ends[-1]))[self.rank::self.world]
                self.order_epoch = epoch
            if self.state["cursor"] < len(self.order):
                index = self.order[self.state["cursor"]]
                self.state["cursor"] += 1
                return self.cache.location(index)
            self.state.update(epoch=epoch+1, cursor=0)

    def _refill(self):
        references, chars = [], 0
        while chars < self.buffer_tokens * 4:
            block = [self._next_document() for _ in range(1024)]
            for part in sorted({p for p, _ in block}):
                selected = [(p, i) for p, i in block if p == part]
                references.extend(selected)
                chars += sum(int(self.cache.chars[p][i]) for p, i in selected)
        references = [ref for kind in ("documents", "queries") for ref in references
                      if self.cache.sources[ref[0]]["kind"] == kind]
        self.state.update(fill=self.state["fill"]+1, references=references, batch=0)

    def _load_buffer(self):
        arrays = [self.cache.row(p, i) for p, i in self.state["references"]]
        self.buffer = torch.from_numpy(np.concatenate(arrays)).to(self.device)
        available = torch.tensor(len(self.buffer) // self.batch_size, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(available, op=dist.ReduceOp.MIN)
        self.available = int(available)
        if not self.available:
            raise ValueError("Token buffer is smaller than a training batch")
        self.permutation = torch.randperm(len(self.buffer), generator=torch.Generator().manual_seed(
            self.recipe.seed + self.state["fill"])).to(self.device)

    def __next__(self):
        if self.buffer is None:
            if not self.state["references"]:
                self._refill()
            self._load_buffer()
        if self.state["batch"] >= self.available:
            self.buffer = None
            self._refill()
            self._load_buffer()
        start = self.state["batch"] * self.batch_size
        self.state["batch"] += 1
        return self.buffer[self.permutation[start:start+self.batch_size]].float()


def train(recipe, cache_path, output, *, resume=None, stop_after=None, save_steps=100000,
          log_steps=100, buffer_tokens=4_000_000, run_settings=None, dataset_revisions=None):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                                device_id=device if device.type == "cuda" else None)
    if recipe.batch_size % world:
        raise ValueError("The global token batch must be divisible by world size")
    torch.manual_seed(recipe.seed)
    cache = TokenCache(cache_path, recipe)
    for repo, revision in (dataset_revisions or {}).items():
        if cache.manifest["dataset_revisions"].get(repo) != revision:
            raise ValueError(f"Token-cache revision does not match the configuration: {repo}")
    if int(cache.ends[-1]) < world:
        raise ValueError("At least one cached document per rank is required")
    steps = int(cache.scheduled_tokens * recipe.schedule_token_fraction) // recipe.batch_size * recipe.epochs
    if recipe.max_steps:
        steps = min(steps, recipe.max_steps)
    if steps < 1:
        raise ValueError("Cache is smaller than one global batch")
    output = Path(output)
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory or explicitly --resume")
    if world > 1:
        dist.barrier()
    output.mkdir(parents=True, exist_ok=True)
    model = TopKSAE(recipe).to(device)
    state = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if state:
        if (state["recipe"] != recipe.checkpoint_config() or state["cache_manifest"] != cache.manifest
                or state["world_size"] != world or state["buffer_tokens"] != buffer_tokens):
            raise ValueError("Resume requires the same recipe, token cache, buffer size, and world size")
        model.load_state_dict(state["model"])
    elif rank == 0:
        rng = random.Random(recipe.seed)
        model.initialize(cache.initialization_tokens(rng), rng)
    wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None) if world > 1 else model
    batches = TokenBatches(cache, recipe, rank, world, device, buffer_tokens,
                          state["data_states"][rank] if state else None)
    pending = None
    if state is None:
        pending = next(batches)
        mean = pending.mean(0)
        if world > 1:
            dist.all_reduce(mean)
            mean /= world
        with torch.no_grad():
            model.b_dec.copy_(mean)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay,
                                 betas=(recipe.adam_beta1, recipe.adam_beta2), eps=recipe.adam_epsilon)
    warmup = max(1, int(steps * recipe.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: s / warmup if s < warmup else
        0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    start = 0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = state["step"]
    if rank == 0:
        settings = dict(run_settings or {key: getattr(recipe, key) for key in RECIPE_KEYS})
        settings.setdefault("encode_batch_size", 4)
        settings.update(cache=str(cache_path), output=str(output), resume=str(resume) if resume else None,
                        prepare=False, stop_after=stop_after, save_steps=save_steps, log_steps=log_steps,
                        buffer_tokens=buffer_tokens, dataset_revisions=cache.manifest["dataset_revisions"])
        write_run_config(output, run_config(model_name(recipe.name), "sae", argparse.Namespace(**settings),
                                           world_size=world, scheduled_steps=steps))
        (output / "recipe.json").write_text(json.dumps(recipe.checkpoint_config(), indent=2))
        (output / "cache_manifest.json").write_text(json.dumps(cache.manifest, indent=2))
        print(f"{recipe.name}: {steps:,} updates; global token batch {recipe.batch_size}; world size {world}", flush=True)

    def save(step, final=False):
        data_states = [None] * world
        if world > 1:
            dist.all_gather_object(data_states, batches.state)
        else:
            data_states[0] = batches.state
        if rank == 0:
            folder = output / ("final" if final else f"checkpoint-{step}")
            folder.mkdir(parents=True, exist_ok=True)
            weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
            save_file(weights, str(folder / "sae.safetensors"))
            (folder / "config.json").write_text(json.dumps(recipe.checkpoint_config(), indent=2))
            checkpoint = dict(step=step, model=weights, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                              data_states=data_states, recipe=recipe.checkpoint_config(), cache_manifest=cache.manifest,
                              world_size=world, buffer_tokens=buffer_tokens)
            temporary = folder / "training_state.pt.partial"
            torch.save(checkpoint, temporary)
            temporary.replace(folder / "training_state.pt")
        if world > 1:
            dist.barrier()

    end = min(steps, stop_after) if stop_after else steps
    if end <= start:
        raise ValueError("Requested stopping point is at or before the resumed step")
    for step in range(start, end):
        x = pending if pending is not None else next(batches)
        pending = None
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=recipe.bf16 and device.type == "cuda"):
            loss, recon, values, indices, mse, aux, balance = wrapped(x)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at update {step}")
        loss.backward()
        model.project_decoder_gradient()
        optimizer.step()
        scheduler.step()
        model.after_step(values, indices)
        if (step + 1) % log_steps == 0 or step + 1 == end:
            # Global-batch FVU (not a held-out evaluation).
            with torch.no_grad():
                moments = torch.cat([x.sum(0), x.square().sum(0), (recon.float()-x).square().sum().view(1),
                                     torch.stack([loss.detach(), mse, aux, balance])])
                if world > 1:
                    dist.all_reduce(moments)
                d = recipe.d_model
                variance = (moments[d:2*d] - moments[:d].square() / recipe.batch_size).sum()
                metrics = dict(step=step+1, tokens_seen=(step+1)*recipe.batch_size,
                    loss=float(moments[-4]/world), mse=float(moments[-3]/world), aux=float(moments[-2]/world),
                    balance=float(moments[-1]/world), fvu=float(moments[2*d]/variance.clamp_min(1e-12)),
                    dead_latents_pct=float((model.steps_since_fired > recipe.dead_steps).float().mean()*100),
                    lr=scheduler.get_last_lr()[0])
            if rank == 0:
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                print(json.dumps(metrics), flush=True)
        if (step + 1) % save_steps == 0 and step + 1 != end:
            save(step+1)
    save(end, final=end == steps)


def read_config(stage, path=None):
    default_model = "mSlideOn-SAE"
    source = Path(path) if path else Path(__file__).resolve().parents[1] / "training-configs" / default_model / "config.json"
    config = json.loads(source.read_text())
    if not isinstance(config, dict) or set(config) - {"model", "stage", "parameters", "resolved"}:
        raise ValueError("Expected a training config with model, stage, and parameters")
    if config.get("stage") != stage or config.get("model") != default_model:
        raise ValueError(f"Expected the {default_model} {stage} configuration")
    if not isinstance(config.get("parameters"), dict):
        raise ValueError("Configuration parameters must be an object")
    return config


def parse_config_args(parser, stage, argv=None, extra_keys=()):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    selected, _ = probe.parse_known_args(argv)
    parser.add_argument("--config", help="Training JSON; defaults to scripts/training-configs/mSlideOn-SAE/config.json")
    try:
        config = read_config(stage, selected.config)
        parameters = config["parameters"]
        actions = {action.dest: action for action in parser._actions if action.dest not in ("help", "config")}
        unknown = set(parameters) - set(actions) - set(extra_keys)
        if unknown:
            raise ValueError(f"Unknown configuration parameters: {', '.join(sorted(unknown))}")
        for key, value in parameters.items():
            action = actions.get(key)
            if action is None or value is None:
                continue
            if isinstance(action, argparse.BooleanOptionalAction):
                if type(value) is not bool:
                    raise ValueError(f"{key} must be a boolean")
            elif action.type:
                items = value if action.nargs in ("+", "*") else [value]
                if not isinstance(items, list):
                    raise ValueError(f"{key} must be a list")
                for item in items:
                    if action.type in (int, float) and (type(item) not in (int, float) or not math.isfinite(item)):
                        raise ValueError(f"{key} must be a finite number")
                    if action.type is int and type(item) is not int:
                        raise ValueError(f"{key} must be an integer")
                    if action.type is str and not isinstance(item, str):
                        raise ValueError(f"{key} must be a string")
        parser.set_defaults(**parameters)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    args = parser.parse_args(argv)
    args.model = config["model"]
    return args


def run_config(model, stage, args, **resolved):
    parameters = {key: value for key, value in vars(args).items() if key not in ("config", "model")}
    return dict(model=model, stage=stage, parameters=parameters, resolved=resolved)


def write_run_config(output, config):
    target = Path(output) / "run_config.json"
    temporary = target.with_suffix(".json.partial")
    temporary.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Pretrain mSlideOn-SAE (native PyTorch)")
    parser.add_argument("--cache", help="Reusable token cache directory")
    parser.add_argument("--output", help="Training output directory")
    parser.add_argument("--prepare", action=argparse.BooleanOptionalAction, default=False,
                        help="Prepare tokens only, using one process")
    parser.add_argument("--encode-batch-size", type=int)
    parser.add_argument("--dataset-revisions", help="JSON file mapping dataset repositories to exact revisions")
    parser.add_argument("--resume", help="Path to a training_state.pt from this trainer")
    parser.add_argument("--stop-after", type=int, help="Stop at this update without changing the full LR schedule")
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--log-steps", type=int)
    parser.add_argument("--buffer-tokens", type=int, help="Approximate buffer target; affects data order")
    parser.add_argument("--learning-rate", "--lr", dest="lr", type=float)
    parser.add_argument("--batch-size", type=int, help="Global token batch")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--query-length", type=int)
    parser.add_argument("--document-length", type=int)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction)
    args = parse_config_args(parser, "sae", argv, extra_keys=RECIPE_KEYS)
    try:
        recipe = recipe_from_parameters(args.model.removesuffix("-SAE").lower(), vars(args))
        if isinstance(args.dataset_revisions, str):
            args.dataset_revisions = json.loads(Path(args.dataset_revisions).read_text())
        if not isinstance(args.dataset_revisions, dict) or any(
            not isinstance(repo, str) or not isinstance(revision, str)
            for repo, revision in args.dataset_revisions.items()
        ):
            raise ValueError("dataset_revisions must map repository IDs to revision strings")
        if not args.cache:
            raise ValueError("--cache is required")
        if not args.prepare and not args.output:
            raise ValueError("--output is required for training")
        if min(args.save_steps, args.log_steps, args.buffer_tokens, args.encode_batch_size) < 1:
            raise ValueError("Step counts, buffer size, and encoding batch size must be positive")
        if args.stop_after is not None and args.stop_after < 1:
            raise ValueError("--stop-after must be positive")
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return args, recipe


def main(argv=None):
    args, recipe = parse_args(argv)
    if args.prepare:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Token preparation must use one process")
        prepare_tokens(recipe, args.cache, args.encode_batch_size, args.dataset_revisions)
        manifest = json.loads((Path(args.cache) / "manifest.json").read_text())
        args.dataset_revisions = manifest["dataset_revisions"]
        write_run_config(args.cache, run_config(args.model, "sae", args, world_size=1))
        return
    try:
        train(recipe, args.cache, args.output, resume=args.resume, stop_after=args.stop_after,
              save_steps=args.save_steps, log_steps=args.log_steps, buffer_tokens=args.buffer_tokens,
              run_settings={key: value for key, value in vars(args).items() if key not in ("config", "model")},
              dataset_revisions=args.dataset_revisions)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
