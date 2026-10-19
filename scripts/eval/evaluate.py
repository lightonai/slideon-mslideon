"""MTEB retrieval through the checkpoint's ST adapter; no ANN or normalization."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy import sparse


BEIR = ("ArguAna", "CQADupstackRetrieval", "ClimateFEVER", "DBPedia", "FEVER", "FiQA2018",
        "HotpotQA", "MSMARCO", "NFCorpus", "NQ", "QuoraRetrieval", "SCIDOCS", "SciFact",
        "TRECCOVID", "Touche2020")
CODE = ("CodeSearchNetCCRetrieval", "COIRCodeSearchNetRetrieval", "CodeFeedbackST",
        "SyntheticText2SQL", "CodeFeedbackMT", "CodeEditSearchRetrieval", "StackOverflowQA",
        "CosQA", "AppsRetrieval", "CodeSearchNetRetrieval", "CodeTransOceanContest", "CodeTransOceanDL")
SUITES = dict(beir=BEIR, miracl=("MIRACLRetrievalHardNegatives",), mldr=("MultiLongDocRetrieval",), code=CODE)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def efficiency(queries, documents):
    """Expected active-latent overlap for independent uniform query/document draws."""
    if queries.shape[0] == 0 or documents.shape[0] == 0:
        raise ValueError("Cannot measure an empty collection")
    q_frequency = np.asarray(queries.getnnz(axis=0), dtype=np.float64) / queries.shape[0]
    d_frequency = np.asarray(documents.getnnz(axis=0), dtype=np.float64) / documents.shape[0]
    return dict(queries=queries.shape[0], documents=documents.shape[0],
                query_nnz=queries.nnz / queries.shape[0], document_nnz=documents.nnz / documents.shape[0],
                flops=float(q_frequency @ d_frequency))


@torch.inference_mode()
def exact_dot(queries, documents, top_k, device="cpu", query_batch=64, document_chunk=16384):
    """Exhaustive sparse-document x dense-query multiplication, with bounded score blocks."""
    if min(top_k, query_batch, document_chunk) < 1:
        raise ValueError("Search sizes must be positive")
    if queries.shape[1] != documents.shape[1] or documents.shape[0] == 0:
        raise ValueError("Invalid embedding shapes")
    device = torch.device(device)
    k = min(top_k, documents.shape[0])
    # Keep only top-k on CPU; never materialize the full query x document score matrix.
    best_scores = torch.full((queries.shape[0], k), -torch.inf)
    best_ids = torch.full((queries.shape[0], k), -1, dtype=torch.long)
    for start in range(0, documents.shape[0], document_chunk):
        block = documents[start:start+document_chunk]
        csr = torch.sparse_csr_tensor(
            torch.from_numpy(block.indptr.astype(np.int64)).to(device),
            torch.from_numpy(block.indices.astype(np.int64)).to(device),
            torch.from_numpy(block.data.astype(np.float32)).to(device), size=block.shape, check_invariants=True)
        for qstart in range(0, queries.shape[0], query_batch):
            qend = min(qstart+query_batch, queries.shape[0])
            dense = torch.from_numpy(queries[qstart:qend].toarray()).float().to(device)
            scores = torch.sparse.mm(csr, dense.T).T
            values, ids = scores.topk(min(k, block.shape[0]), dim=-1)
            merged_scores = torch.cat((best_scores[qstart:qend].to(device), values), dim=1)
            merged_ids = torch.cat((best_ids[qstart:qend].to(device), ids+start), dim=1)
            selected, positions = merged_scores.topk(k, dim=-1)
            best_scores[qstart:qend] = selected.cpu()
            best_ids[qstart:qend] = merged_ids.gather(1, positions).cpu()
    return best_scores.numpy(), best_ids.numpy()


class ExactRetrieval:
    def __init__(self, model, model_id, revision, output, batch_size=8, query_top_k=0,
                 document_top_k=0, query_batch=64, document_chunk=16384):
        from mteb.models.model_meta import ModelMeta, ScoringFunction
        self.model, self.output = model, Path(output)
        self.batch_size = batch_size
        self.query_top_k, self.document_top_k = query_top_k, document_top_k
        self.query_batch, self.document_chunk = query_batch, document_chunk
        self.mteb_model_meta = ModelMeta.create_empty().model_copy(update=dict(
            name=model_id, revision=revision, similarity_fn_name=ScoringFunction.DOT_PRODUCT, model_type=["sparse"]))
        self.efficiency_path = self.output / "efficiency.json"
        self.statistics = json.loads(self.efficiency_path.read_text()) if self.efficiency_path.exists() else {}
        self.corpus = None

    def encode_dataset(self, dataset, task_metadata, task):
        from mteb._create_dataloaders import create_dataloader
        from mteb.types import PromptType
        loader = create_dataloader(dataset, task_metadata=task_metadata,
            prompt_type=PromptType.query if task == "query" else PromptType.document, batch_size=self.batch_size)
        blocks = []
        for batch in loader:
            vectors = getattr(self.model, "encode_" + task)(
                batch["text"], batch_size=self.batch_size, convert_to_tensor=True,
                show_progress_bar=False, top_k=self.query_top_k if task == "query" else self.document_top_k)
            if vectors.layout != torch.strided:
                vectors = vectors.to_dense()
            array = vectors.detach().float().cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError("Non-finite model embeddings")
            blocks.append(sparse.csr_matrix(array))
        if not blocks:
            raise ValueError("Dataset is empty")
        return sparse.vstack(blocks, format="csr")

    def index(self, corpus, *, task_metadata, hf_split, hf_subset, encode_kwargs, num_proc=None):
        self.corpus = None  # release the previous corpus before encoding the next one
        self.document_ids = [str(value) for value in corpus["id"]]
        print(f"Encoding {task_metadata.name}/{hf_subset}/{hf_split}: {len(corpus):,} documents", flush=True)
        self.corpus = self.encode_dataset(corpus, task_metadata, "document")

    def search(self, queries, *, task_metadata, hf_split, hf_subset, top_k, encode_kwargs,
               top_ranked=None, num_proc=None):
        if top_ranked is not None:
            raise ValueError("This evaluator supports retrieval, not reranking")
        query_vectors = self.encode_dataset(queries, task_metadata, "query")
        key = f"{task_metadata.name}/{hf_subset}/{hf_split}"
        self.statistics[key] = efficiency(query_vectors, self.corpus)
        write_json(self.efficiency_path, self.statistics)
        print(f"Exact dot search: {key}; {self.statistics[key]}", flush=True)
        scores, indices = exact_dot(query_vectors, self.corpus, top_k, device=self.model.device,
                                   query_batch=self.query_batch, document_chunk=self.document_chunk)
        return {str(qid): {self.document_ids[int(i)]: float(score) for i, score in zip(ids, values)}
                for qid, ids, values in zip(queries["id"], indices, scores)}


def select_tasks(names, subsets=None, splits=None):
    import mteb
    tasks = list(mteb.get_tasks(tasks=names))
    if {t.metadata.name for t in tasks} != set(names):
        raise ValueError("MTEB did not resolve every requested task")
    for task in tasks:
        if task.metadata.type != "Retrieval":
            raise ValueError(f"Not a retrieval task: {task.metadata.name}")
        desired = splits or (list(task.metadata.eval_splits) if task.metadata.name in CODE else
                             ["dev"] if task.metadata.name in ("MSMARCO", "MIRACLRetrievalHardNegatives") else ["test"])
        if not set(desired).issubset(task.metadata.eval_splits):
            raise ValueError(f"Invalid splits for {task.metadata.name}: {desired}; available: {task.metadata.eval_splits}")
        task.metadata.eval_splits = list(desired)
        if subsets:
            if not set(subsets).issubset(task.hf_subsets):
                raise ValueError(f"Invalid subsets for {task.metadata.name}: {subsets}; available: {task.hf_subsets}")
            task.hf_subsets = list(subsets)
    return tasks


def summarize(results, tasks, suites):
    by_name = {task.metadata.name: task for task in tasks}
    scores = {}
    for result in results:
        name = result.task_name
        task = by_name[name]
        values = [row["ndcg_at_10"] for split, rows in result.scores.items() if split in task.metadata.eval_splits
                  for row in rows if row.get("hf_subset") in task.hf_subsets and "ndcg_at_10" in row]
        if not values:
            raise ValueError(f"No nDCG@10 results for {name}")
        scores[name] = float(np.mean(values))
    averages = {}
    for suite in suites:
        members = SUITES[suite]
        complete = all(name in scores for name in members)
        averages[suite] = dict(completed=sum(name in scores for name in members), total=len(members),
                              ndcg_at_10=float(np.mean([scores[name] for name in members])) if complete else None)
    return dict(metric="ndcg_at_10", scale="0-1", task_scores=scores, suites=averages,
                task_scope={name: dict(splits=list(task.metadata.eval_splits), subsets=list(task.hf_subsets))
                            for name, task in by_name.items()})


def main():
    parser = argparse.ArgumentParser(description="Evaluate an HF sparse checkpoint with exact dot search")
    parser.add_argument("--model", required=True, help="Hugging Face model ID")
    parser.add_argument("--revision", help="Optional HF commit/tag; resolved commit is recorded")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow the checkpoint's custom ST adapter")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--suite", choices=list(SUITES), nargs="+")
    group.add_argument("--tasks", nargs="+", help="Explicit MTEB retrieval task names")
    parser.add_argument("--subsets", nargs="+", help="For example: en (must exist in every selected task)")
    parser.add_argument("--splits", nargs="+", help="Override default test/dev splits")
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-top-k", type=int, default=0, help="Final vector budget; 0 is unpruned")
    parser.add_argument("--document-top-k", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8, help="Encoder batch size")
    parser.add_argument("--query-batch", type=int, default=64, help="Scoring query block size")
    parser.add_argument("--document-chunk", type=int, default=16384, help="Scoring document block size")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("Run evaluation with python, not torchrun")
    if min(args.batch_size, args.query_batch, args.document_chunk) < 1 or min(args.query_top_k, args.document_top_k) < 0:
        parser.error("Invalid batch size or pruning budget")
    import mteb
    from huggingface_hub import HfApi
    from mteb.cache import ResultCache
    from sentence_transformers import SparseEncoder
    suites = args.suite or ([] if args.tasks else ["beir"])
    names = args.tasks or list(dict.fromkeys(name for suite in suites for name in SUITES[suite]))
    tasks = select_tasks(names, args.subsets, args.splits)
    revision = HfApi().model_info(args.model, revision=args.revision).sha
    config = dict(model=args.model, revision=revision, mteb_version=mteb.__version__, score="dot",
                  query_top_k=args.query_top_k, document_top_k=args.document_top_k, batch_size=args.batch_size,
                  tasks={t.metadata.name: dict(splits=list(t.metadata.eval_splits), subsets=list(t.hf_subsets)) for t in tasks})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output contains a different evaluation configuration; use another directory")
    if not config_path.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory")
    write_json(config_path, config)
    model = SparseEncoder(args.model, revision=revision, trust_remote_code=args.trust_remote_code, device=args.device)
    model.eval()
    search = ExactRetrieval(model, args.model, revision, output, args.batch_size, args.query_top_k,
                            args.document_top_k, args.query_batch, args.document_chunk)
    cache = ResultCache(output / "mteb")
    results = []
    for task in tasks:
        result = mteb.evaluate(search, [task], cache=cache, co2_tracker=False,
                               encode_kwargs={"batch_size": args.batch_size}, raise_error=True)
        results.extend(result.task_results)
        write_json(output / "summary.json", summarize(results, tasks, suites))
    print(json.dumps(summarize(results, tasks, suites), indent=2))
