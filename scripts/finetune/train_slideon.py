"""Fine-tune SlideOn: data collation, retrieval losses, and Trainer setup."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from sentence_transformers import SparseEncoderTrainer, SparseEncoderTrainingArguments
from torch import nn
from torch.nn import functional as F
from transformers import TrainerCallback, set_seed

from slideon.loading import from_sae


def prune(values, budget):
    if budget == 0:
        return values
    if not 0 < budget <= values.shape[-1]:
        raise ValueError("Invalid pruning budget")
    selected, indices = values.topk(budget, dim=-1)
    return torch.zeros_like(values).scatter(-1, indices, selected)


class SlideOnLoss(nn.Module):
    def __init__(self, model, temperature=0.02, kd_weight=1.0, teacher_temperature=1.0,
                 balance_weight=0.01, query_flops=0.01, document_flops=0.01,
                 query_budgets=(128, 64, 32), document_budgets=(1024, 512),
                 flops_ramp_steps=3000):
        super().__init__()
        if temperature <= 0 or teacher_temperature <= 0:
            raise ValueError("Temperatures must be positive")
        self.model = model
        self.temperature, self.teacher_temperature = temperature, teacher_temperature
        self.kd_weight, self.balance_weight = kd_weight, balance_weight
        self.query_flops, self.document_flops = query_flops, document_flops
        self.query_budgets, self.document_budgets = query_budgets, document_budgets
        self.flops_ramp_steps = flops_ramp_steps
        self.step = 0

    def forward(self, sentence_features, labels=None):
        outputs = [self.model(dict(feature), top_k=0) for feature in sentence_features]
        q = outputs[0]["sentence_embedding"].float()
        docs = torch.cat([item["sentence_embedding"].float() for item in outputs[1:]])
        batch = q.shape[0]
        target = torch.arange(batch, device=q.device)
        own_docs = target[:, None] + torch.arange(len(outputs)-1, device=q.device)[None, :] * batch
        if self.kd_weight and (labels is None or labels.shape != own_docs.shape):
            raise ValueError("KD requires teacher_scores for positive then each negative")
        contrastive, kd = [], []
        for qb, db in itertools.product(self.query_budgets, self.document_budgets):
            scores = prune(q, qb) @ prune(docs, db).T / self.temperature
            contrastive.append(F.cross_entropy(scores, target))
            if self.kd_weight:
                kd.append(F.kl_div(F.log_softmax(scores.gather(1, own_docs), -1),
                          F.softmax(labels.float() / self.teacher_temperature, -1),
                          reduction="batchmean"))
        ramp = min(1.0, self.step / self.flops_ramp_steps) ** 2 if self.flops_ramp_steps else 1.0
        result = {
            "contrastive": torch.stack(contrastive).mean(),
            "load_balance": self.balance_weight * torch.stack([o["load_balance"] for o in outputs]).mean(),
            "query_flops": ramp * self.query_flops * q.mean(0).square().sum(),
            "document_flops": ramp * self.document_flops * docs.mean(0).square().sum(),
        }
        if kd:
            result["kd"] = self.kd_weight * torch.stack(kd).mean()
        return result


class RetrievalCollator:
    valid_label_columns = ["teacher_scores"]

    def __init__(self, module):
        self.module = module

    def __call__(self, rows):
        count = len(rows[0]["negatives"])
        if any(len(row["negatives"]) != count for row in rows):
            raise ValueError("Use a fixed number of negatives per example")
        result = {}
        columns = [[r["query"] for r in rows], [r["positive"] for r in rows]]
        columns += [[r["negatives"][i] for r in rows] for i in range(count)]
        for i, texts in enumerate(columns):
            features = self.module.preprocess(texts, task="query" if i == 0 else "document")
            result.update({f"text{i}_{key}": value for key, value in features.items()})
        if "teacher_scores" in rows[0]:
            result["label"] = torch.tensor([r["teacher_scores"] for r in rows])
        return result


class RampCallback(TrainerCallback):
    def __init__(self, loss):
        self.loss = loss

    def on_step_begin(self, args, state, control, **kwargs):
        self.loss.step = state.global_step


def read_config(stage, path=None):
    default_model = "SlideOn"
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
    parser.add_argument("--config", help="Training JSON; defaults to scripts/training-configs/SlideOn/config.json")
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
    parser = argparse.ArgumentParser(description="Fine-tune a token SAE using SparseEncoderTrainer")
    for key in ("sae", "sae_revision", "backbone_revision", "train", "output", "resume",
                "query_prompt", "document_prompt", "lr_scheduler_type", "optim"):
        parser.add_argument("--" + key.replace("_", "-"), type=str)
    for key in ("learning_rate", "epochs", "temperature", "teacher_temperature", "kd_weight",
                "balance_weight", "query_flops", "document_flops", "weight_decay", "warmup_ratio",
                "adam_beta1", "adam_beta2", "adam_epsilon", "max_grad_norm"):
        parser.add_argument("--" + key.replace("_", "-"), type=float)
    for key in ("max_steps", "global_batch_size", "batch_size", "gradient_accumulation",
                "query_length", "document_length", "token_k", "head_chunk_size", "flops_ramp_steps",
                "seed", "save_steps", "logging_steps", "save_total_limit"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    for key in ("query_budgets", "document_budgets"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, nargs="+")
    for key in ("bf16", "gradient_checkpointing"):
        parser.add_argument("--" + key.replace("_", "-"), action=argparse.BooleanOptionalAction)
    args = parse_config_args(parser, "retrieval", argv)
    try:
        for key, value in vars(args).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if not args.sae or args.query_prompt is None or args.document_prompt is None:
            raise ValueError("sae, query_prompt, and document_prompt must be configured")
        if not args.train or not args.output:
            raise ValueError("--train and --output are required")
        positive = ("batch_size", "query_length", "document_length", "token_k", "head_chunk_size",
                    "save_steps", "logging_steps", "save_total_limit", "learning_rate", "epochs",
                    "temperature", "teacher_temperature", "adam_epsilon")
        if any(getattr(args, key) is None or getattr(args, key) <= 0 for key in positive):
            raise ValueError("Batch sizes, lengths, step intervals, rates, and temperatures must be positive")
        if args.max_steps is None or args.max_steps == 0 or args.max_steps < -1:
            raise ValueError("max_steps must be -1 or a positive integer")
        for key in ("kd_weight", "balance_weight", "query_flops", "document_flops", "weight_decay",
                    "flops_ramp_steps", "seed", "max_grad_norm"):
            if getattr(args, key) is None or getattr(args, key) < 0:
                raise ValueError(f"{key} must be nonnegative")
        if not (0 <= args.warmup_ratio < 1 and 0 <= args.adam_beta1 < 1 and 0 <= args.adam_beta2 < 1):
            raise ValueError("Invalid warmup ratio or Adam betas")
        for budgets in (args.query_budgets, args.document_budgets):
            if not budgets or any(budget < 0 for budget in budgets):
                raise ValueError("Pruning budgets must be nonempty lists of nonnegative integers")
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world < 1:
            raise ValueError("WORLD_SIZE must be positive")
        per_update = args.batch_size * world
        if args.global_batch_size is None:
            args.global_batch_size = per_update * (args.gradient_accumulation or 1)
        if args.global_batch_size < 1 or args.global_batch_size % per_update:
            raise ValueError("global_batch_size must be divisible by batch_size * WORLD_SIZE")
        accumulation = args.global_batch_size // per_update
        if args.gradient_accumulation is not None and args.gradient_accumulation != accumulation:
            raise ValueError("gradient_accumulation does not match the requested global_batch_size")
        args.gradient_accumulation = accumulation
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    if torch.cuda.device_count() > 1 and "LOCAL_RANK" not in os.environ:
        raise ValueError("Use torchrun for multiple GPUs, or set CUDA_VISIBLE_DEVICES to one GPU")
    set_seed(args.seed)
    model = from_sae(args.sae, revision=args.sae_revision, backbone_revision=args.backbone_revision,
                     query_length=args.query_length, document_length=args.document_length)
    if args.gradient_checkpointing:
        model[0].model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    settings = model[0].settings
    settings.update(token_k=args.token_k, head_chunk_size=args.head_chunk_size,
                    query_top_k=0, document_top_k=0,
                    prompts={"query": args.query_prompt, "document": args.document_prompt})
    model.prompts = dict(settings["prompts"])
    dimensions = model.get_embedding_dimension()
    if args.token_k > dimensions or any(budget > dimensions for budget in args.query_budgets + args.document_budgets):
        raise ValueError("Token sparsity and pruning budgets must fit the model's latent dimension")
    dataset = load_dataset("json", data_files=args.train, split="train")
    required = {"query", "positive", "negatives"} | ({"teacher_scores"} if args.kd_weight else set())
    if not required.issubset(dataset.column_names):
        raise ValueError(f"Dataset must contain {sorted(required)}")
    if len(dataset) < args.batch_size * int(os.environ.get("WORLD_SIZE", "1")):
        raise ValueError("Dataset is smaller than one per-step batch; lower --batch-size")
    loss = SlideOnLoss(model, **{key: getattr(args, key) for key in (
        "temperature", "teacher_temperature", "kd_weight", "balance_weight", "query_flops",
        "document_flops", "query_budgets", "document_budgets", "flops_ramp_steps")})
    training_args = SparseEncoderTrainingArguments(
        output_dir=args.output, learning_rate=args.learning_rate, num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation, bf16=args.bf16,
        weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type, adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2, adam_epsilon=args.adam_epsilon, optim=args.optim,
        max_grad_norm=args.max_grad_norm, gradient_checkpointing=False,
        save_steps=args.save_steps, logging_steps=args.logging_steps, save_total_limit=args.save_total_limit,
        report_to="none", remove_unused_columns=False, dataloader_drop_last=True, seed=args.seed,
        ddp_find_unused_parameters=False)
    trainer = SparseEncoderTrainer(model=model, args=training_args, train_dataset=dataset,
        loss=loss, data_collator=RetrievalCollator(model[0]), callbacks=[RampCallback(loss)])
    if trainer.is_world_process_zero():
        Path(args.output).mkdir(parents=True, exist_ok=True)
        write_run_config(args.output, run_config(args.model, "retrieval", args,
            world_size=training_args.world_size, dataset_fingerprint=dataset._fingerprint,
            sae_commit=getattr(model[0].model.config, "_commit_hash", None)))
    try:
        trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_model(str(Path(args.output) / "final"))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
