# SlideOn and mSlideOn: Training and Evaluation

<p align="center">
  <img src="assets/slideon-mslideon.png" alt="SlideOn and mSlideOn: a golden retriever sliding through a library with a bag of glowing stars" width="512">
</p>

Training and evaluation scripts for **[SlideOn](https://huggingface.co/lightonai/SlideOn)**,
our English sparse retriever, and **[mSlideOn](https://huggingface.co/lightonai/mSlideOn)**,
our multilingual sparse retriever for natural language, long documents, and code.
This repository covers sparse autoencoder (SAE) pretraining, supervised retrieval
fine-tuning, and evaluation.

SlideOn starts from [DenseOn](https://huggingface.co/lightonai/DenseOn), while
mSlideOn starts from [mLateOn](https://huggingface.co/lightonai/mLateOn). SAE
pretraining followed by retrieval fine-tuning converts their token representations
into learned sparse vectors for inverted-index search. Matryoshka training allows
query and document sparsity to be adjusted at inference to control search and
storage costs.

For the method, training data, experiments, and results, see our
[SlideOn and mSlideOn blog post](https://huggingface.co/blog/lightonai/slideon-mslideon).
The final retrieval models and pretrained SAE initializations are available in our
[Hugging Face collection](https://huggingface.co/collections/lightonai/slideon-and-mslideon).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Training

Each model has its own editable training scripts under `scripts/`. Data
preparation, losses, and training are implemented directly in those files. `src/slideon/` contains the
model and checkpoint-loading code.

| Model | Training script | Configuration |
| --- | --- | --- |
| [SlideOn-SAE](https://huggingface.co/lightonai/SlideOn-SAE) | [SAE pretraining](scripts/sae-pretrain/train_slideon_sae.py) | [Config](scripts/training-configs/SlideOn-SAE/config.json) |
| [mSlideOn-SAE](https://huggingface.co/lightonai/mSlideOn-SAE) | [SAE pretraining](scripts/sae-pretrain/train_mslideon_sae.py) | [Config](scripts/training-configs/mSlideOn-SAE/config.json) |
| [SlideOn](https://huggingface.co/lightonai/SlideOn) | [Retrieval fine-tuning](scripts/finetune/train_slideon.py) | [Config](scripts/training-configs/SlideOn/config.json) |
| [mSlideOn](https://huggingface.co/lightonai/mSlideOn) | [Retrieval fine-tuning](scripts/finetune/train_mslideon.py) | [Config](scripts/training-configs/mSlideOn/config.json) |

Each script loads its corresponding config automatically. Edit the JSON or pass
`--config my-config.json`; command-line arguments override its values. Every run
saves the effective settings in `run_config.json`. Use `--help` for available flags.

### SAE pretraining

Prepare frozen token embeddings once, then train the SAE:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sae-pretrain/train_slideon_sae.py \
  --prepare --cache token-cache/slideon

torchrun --standalone --nproc_per_node=8 scripts/sae-pretrain/train_slideon_sae.py \
  --cache token-cache/slideon --output outputs/slideon-sae
```

For mSlideOn, use `scripts/sae-pretrain/train_mslideon_sae.py` with separate cache and output
directories. Its config selects mLateOn, multilingual data, and load balancing;
SlideOn uses DenseOn and AuxK.

Both configs use 32,768 latents, token top-k 32, a global token batch of 4,096,
and AdamW at 0.0003. Change these and the data selection in the config. Token
preparation records dataset revisions; `--dataset-revisions pins.json` accepts
a mapping of dataset IDs to exact commits. The FP16 cache uses two bytes per
token dimension, so check storage before preparing a full training corpus.

Training writes metrics to `metrics.jsonl` and the completed SAE to `final/`.
Resume with `--resume path/to/training_state.pt`, keeping the same config, token
cache, world size, and buffer size. `--stop-after N` allows an early stop without
changing the learning-rate schedule.

### Retrieval fine-tuning

Provide JSONL with a fixed number of negatives. Teacher scores are ordered
positive first, then negatives:

```json
{"query":"What is sparse retrieval?","positive":"Sparse retrieval uses inverted indexes.","negatives":["Bread is baked in an oven."],"teacher_scores":[3.0,0.0]}
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune/train_slideon.py \
  --train english.jsonl --output outputs/slideon

CUDA_VISIBLE_DEVICES=0 python scripts/finetune/train_mslideon.py \
  --train multilingual.jsonl --output outputs/mslideon
```

The scripts start from the pinned SAE releases. Pass `--sae outputs/slideon-sae/final`
to use your own pretrained SAE. Use `torchrun` for multiple GPUs and `--resume PATH`
for a Trainer checkpoint. The completed model is saved under `final/`.

The scripts expose contrastive and distillation losses, load balancing, FLOPS
penalties, and Matryoshka query/document budgets. For example:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune/train_slideon.py \
  --train english.jsonl --output outputs/slideon-experiment \
  --learning-rate 0.00002 --batch-size 8 --global-batch-size 128 \
  --query-budgets 32 64 --document-budgets 512 1024
```

`--batch-size` is per GPU; gradient accumulation is derived from the global batch.
Negatives and regularization statistics are computed per GPU, so accumulation
does not enlarge the negative pool. A positive `max_steps` takes precedence over
`epochs` (use `--max-steps -1` for epoch-based retrieval training).

The configs combine recorded checkpoint settings with explicit defaults for these
standalone scripts. Data mixing and filtering for retrieval fine-tuning remain
upstream responsibilities, and some historical dataset revisions are unavailable.
Full reproduction of the released metrics has not been established.

## Evaluation

```bash
pip install -e '.[eval]'
python scripts/eval/evaluate.py --model lightonai/SlideOn --trust-remote-code \
  --suite beir --query-top-k 32 --document-top-k 512 --output results/slideon

python scripts/eval/evaluate.py --model lightonai/mSlideOn --trust-remote-code \
  --suite beir miracl mldr code --output results/mslideon
```

The suites cover BEIR-15, [MIRACL hard negatives](https://huggingface.co/datasets/mteb/miracl-hard-negatives)
(dev), [MLDR](https://huggingface.co/datasets/mteb/MultiLongDocRetrieval) (test), and
12 code retrieval tasks. Use `--tasks SciFact` for a smaller run or `--subsets en`
for the English subsets of MIRACL and MLDR.

Evaluation uses exact dot-product search. Vectors are unpruned by default;
`--query-top-k` and `--document-top-k` control final pruning. Outputs include MTEB
results, `summary.json` with nDCG@10 on a 0–1 scale, and `efficiency.json` with
average nonzeros and retrieval FLOPS (expected active-latent overlap).

Use `--revision` to pin a checkpoint. Lower `--batch-size` for encoder memory or
`--query-batch` / `--document-chunk` for scoring memory. Corpus vectors stay in
host RAM. Use a separate output directory for each configuration.

## Inference

```python
from sentence_transformers import SparseEncoder

model = SparseEncoder("lightonai/SlideOn", trust_remote_code=True)
queries = model.encode_query(["What is sparse retrieval?"], top_k=32)
documents = model.encode_document(["Sparse retrieval uses inverted indexes."])
scores = model.similarity(queries, documents)
```

Use `lightonai/mSlideOn` for multilingual retrieval, or the `final/` directory
from your own run. Saved models include their backbone, sparse head, tokenizer,
and loading code.

## Citation

If you find this work useful for your research, please consider citing us:

```bibtex
@misc{moura2026slideon,
  author       = {Moura Junior, Paulo Roberto and Sourty, Raphaël},
  title        = {{SlideOn} \& {mSlideOn}: Sparsified late-interaction and dense retrievers},
  year         = {2026},
  howpublished = {Hugging Face Blog},
  url          = {https://huggingface.co/blog/lightonai/slideon-mslideon}
}
```
