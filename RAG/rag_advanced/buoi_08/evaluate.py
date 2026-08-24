"""Offline evaluator for Buổi 08 Advanced RAG.

The evaluator measures retrieval/rerank output only. It never calls answer
generation. Real semantic/rerank modes require prepared local resources and only
run when the user explicitly invokes this script.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import advanced_rag


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS_PATH = BASE_DIR / "eval" / "questions.json"
REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_MODES = ("bm25", "semantic", "hybrid", "hybrid_rerank")

Retriever = Callable[[dict[str, Any], str, int, str], list[str]]


def recall_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Recall@K for binary relevance."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def mrr_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """MRR@K for binary relevance."""
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """nDCG@K with binary relevance."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in relevant:
            dcg += 1.0 / _log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / _log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_questions(
    questions: list[dict[str, Any]],
    *,
    modes: list[str],
    strategy: str,
    k: int,
    retriever: Retriever | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval modes over question labels; generation is never called."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k phải là integer dương")
    for mode in modes:
        if mode not in advanced_rag.SUPPORTED_ANSWER_MODES:
            raise ValueError(f"mode không hợp lệ: {mode}")

    retriever = retriever or retrieve_ranked_chunk_ids
    needs_review = any(question.get("needs_human_review") is True for question in questions)
    warnings = []
    if needs_review:
        warnings.append("Gold labels còn needs_human_review=true; không tuyên bố mode chiến thắng chính thức.")

    per_query: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, list[float]]] = {
        mode: {"recall_at_k": [], "mrr_at_k": [], "ndcg_at_k": [], "latency_ms": []}
        for mode in modes
    }

    for question in questions:
        query_result = {
            "query_id": question.get("query_id", ""),
            "question": question.get("question", ""),
            "relevant_chunk_ids": list(question.get("relevant_chunk_ids", [])),
            "scope": question.get("scope"),
            "needs_human_review": question.get("needs_human_review", False),
            "modes": {},
        }
        for mode in modes:
            start = time.perf_counter()
            try:
                ranked_ids = retriever(question, mode, k, strategy)
                latency_ms = (time.perf_counter() - start) * 1000
                metrics = {
                    "recall_at_k": recall_at_k(ranked_ids, query_result["relevant_chunk_ids"], k),
                    "mrr_at_k": mrr_at_k(ranked_ids, query_result["relevant_chunk_ids"], k),
                    "ndcg_at_k": ndcg_at_k(ranked_ids, query_result["relevant_chunk_ids"], k),
                    "latency_ms": latency_ms,
                    "ranked_chunk_ids": ranked_ids[:k],
                    "status": "ok",
                }
                for key in ("recall_at_k", "mrr_at_k", "ndcg_at_k", "latency_ms"):
                    aggregates[mode][key].append(metrics[key])
            except Exception as error:  # noqa: BLE001 - record per-query failure explicitly.
                latency_ms = (time.perf_counter() - start) * 1000
                metrics = {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}"[:500],
                    "latency_ms": latency_ms,
                    "ranked_chunk_ids": [],
                    "recall_at_k": 0.0,
                    "mrr_at_k": 0.0,
                    "ndcg_at_k": 0.0,
                }
                aggregates[mode]["latency_ms"].append(latency_ms)
            query_result["modes"][mode] = metrics
        per_query.append(query_result)

    metrics_by_mode: dict[str, dict[str, Any]] = {}
    for mode in modes:
        latencies = aggregates[mode]["latency_ms"]
        metrics_by_mode[mode] = {
            "recall_at_k": _mean(aggregates[mode]["recall_at_k"]),
            "mrr_at_k": _mean(aggregates[mode]["mrr_at_k"]),
            "ndcg_at_k": _mean(aggregates[mode]["ndcg_at_k"]),
            "latency_mean_ms": _mean(latencies),
            "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
            "query_count": len(questions),
            "failed_count": sum(1 for item in per_query if item["modes"][mode]["status"] == "failed"),
        }

    config_snapshot = _safe_config_snapshot()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "k": k,
        "modes": modes,
        "model_identity": {
            "embedding_model": config_snapshot.get("embedding_model"),
            "embedding_dim": config_snapshot.get("embedding_dim"),
            "reranker_model": config_snapshot.get("reranker_model"),
        },
        "config": config_snapshot,
        "warnings": warnings,
        "official_winner": None,
        "metrics": metrics_by_mode,
        "queries": per_query,
    }


def retrieve_ranked_chunk_ids(question: dict[str, Any], mode: str, k: int, strategy: str) -> list[str]:
    """Run a real retrieval mode and return ranked chunk IDs; no generation."""
    text = question.get("question", "")
    if mode == "bm25":
        import rag

        chunks, _stats = rag.load_chunks(rag.BUOI_05_CHUNKS_DIR, strategy=strategy)
        return [item["chunk_id"] for item in advanced_rag.bm25_search(text, chunks, k)]
    if mode == "semantic":
        return [item["chunk_id"] for item in advanced_rag.semantic_candidates(text, candidate_k=k, strategy=strategy)]
    if mode == "hybrid":
        return [item["chunk_id"] for item in advanced_rag.hybrid_retrieve(text, strategy=strategy)["candidates"][:k]]
    if mode == "hybrid_rerank":
        result = advanced_rag.hybrid_rerank_retrieve(text, strategy=strategy)
        if result["status"] != "reranked":
            raise RuntimeError(result.get("error", result["status"]))
        return [item["chunk_id"] for item in result["candidates"][:k]]
    raise ValueError(f"mode không hợp lệ: {mode}")


def mock_synthetic_retriever(question: dict[str, Any], mode: str, k: int, strategy: str) -> list[str]:
    """Deterministic offline retriever for evaluator command tests only."""
    ranked = question.get("mock_ranked_chunk_ids")
    if not ranked:
        relevant = list(question.get("relevant_chunk_ids", []))
        decoys = [f"mock-decoy-{mode}-{index}" for index in range(1, k + 2)]
        ranked = relevant + decoys
    return list(ranked)[:k]


def load_questions(path: Path | str) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("questions JSON phải là list")
    return data


def save_report(report: dict[str, Any], output_path: Path | str | None = None) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        timestamp = report["created_at"].replace(":", "").replace("+", "Z").split(".")[0]
        output_path = REPORTS_DIR / f"eval_{timestamp}.json"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Buổi 08 - Offline retrieval evaluator")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH), help="Path tới eval questions JSON")
    parser.add_argument("--strategy", default="hierarchical", help="fixed-size, semantic hoặc hierarchical")
    parser.add_argument("--k", type=int, default=5, help="K cho Recall/MRR/nDCG")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Comma-separated retrieval modes")
    parser.add_argument("--output", default=None, help="Path report JSON; mặc định reports/eval_<timestamp>.json")
    parser.add_argument("--mock-synthetic", action="store_true", help="Dùng deterministic mock retriever cho test offline, không chạy real retrieval")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    questions = load_questions(args.questions)
    retriever = mock_synthetic_retriever if args.mock_synthetic else None
    report = evaluate_questions(questions, modes=modes, strategy=args.strategy, k=args.k, retriever=retriever)
    path = save_report(report, args.output)
    print(f"Saved report: {path}")
    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    print("Metrics:")
    for mode, metrics in report["metrics"].items():
        print(
            f"- {mode}: recall@{args.k}={metrics['recall_at_k']:.4f}, "
            f"mrr@{args.k}={metrics['mrr_at_k']:.4f}, ndcg@{args.k}={metrics['ndcg_at_k']:.4f}, "
            f"latency_mean_ms={metrics['latency_mean_ms']:.3f}, failed={metrics['failed_count']}"
        )
    return 0


def _log2(value: int | float) -> float:
    import math

    return math.log2(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_config_snapshot() -> dict[str, Any]:
    try:
        config = advanced_rag.load_config(advanced_rag.ENV_EXAMPLE_PATH if not advanced_rag.ENV_PATH.exists() else advanced_rag.ENV_PATH)
    except Exception:
        return {}
    return {key: value for key, value in config.items() if key != "api_key"}


if __name__ == "__main__":
    raise SystemExit(main())
