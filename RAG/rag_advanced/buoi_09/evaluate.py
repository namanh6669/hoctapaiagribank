"""Offline/real retrieval evaluator for Buổi 09.

Evaluation is retrieval-only: it compares single/multi flat and parent modes, computes
child/parent metrics, and never calls answer generation. Real semantic/query-expansion
or reranker work only runs when the user explicitly invokes this script. Unit tests
use injected deterministic fakes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import hierarchical_rag as hr


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS_PATH = BASE_DIR / "eval" / "questions.json"
REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_MODES = ("single_flat", "multi_flat", "single_parent", "multi_parent")

ModeRetriever = Callable[[dict[str, Any], str, int, hr.HierarchyConfig], dict[str, Any]]


def recall_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    relevant = set(relevant_ids)
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_questions(
    questions: list[dict[str, Any]],
    *,
    modes: list[str] | None = None,
    k: int = 5,
    config: hr.HierarchyConfig | None = None,
    retriever: ModeRetriever | None = None,
    hierarchy_storage: Path | str = hr.HIERARCHY_STORAGE_DIR,
) -> dict[str, Any]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k phải là integer dương")
    modes = modes or list(DEFAULT_MODES)
    for mode in modes:
        if mode not in hr.ANSWER_MODES:
            raise ValueError(f"mode không hợp lệ: {mode}")
    config = config or hr.load_hierarchy_config()
    validate_questions(questions, hierarchy_storage=hierarchy_storage, config=config)
    retriever = retriever or retrieve_mode
    needs_review = any(question.get("needs_human_review") is True for question in questions)
    warnings = []
    if needs_review:
        warnings.append("Gold labels còn needs_human_review=true; không tuyên bố mode chiến thắng chính thức.")

    per_question: list[dict[str, Any]] = []
    aggregate_inputs: dict[str, dict[str, list[float]]] = {
        mode: {
            "child_recall_at_k": [],
            "parent_recall_at_k": [],
            "mrr_at_k": [],
            "ndcg_at_k": [],
            "latency_ms": [],
            "context_chars": [],
            "expansion_factor": [],
            "query_count": [],
            "child_union_count": [],
            "query_generation_call_count": [],
            "embedding_call_count": [],
            "unique_relevant_parent_count": [],
            "unique_relevant_source_count": [],
        }
        for mode in modes
    }
    failures: list[dict[str, Any]] = []

    for question in questions:
        item = {
            "question_id": question["question_id"],
            "question": question["question"],
            "question_type": question.get("question_type"),
            "relevant_child_ids": list(question.get("relevant_child_ids", [])),
            "relevant_parent_ids": list(question.get("relevant_parent_ids", [])),
            "needs_human_review": bool(question.get("needs_human_review", False)),
            "modes": {},
        }
        for mode in modes:
            started = time.perf_counter()
            try:
                result = retriever(question, mode, k, config)
                latency_ms = (time.perf_counter() - started) * 1000
                summary = summarize_mode_result(result, mode=mode, k=k)
                child_metrics_ids = summary["ranked_child_ids"]
                parent_metrics_ids = summary["ranked_parent_ids"]
                relevant_child_ids = item["relevant_child_ids"]
                relevant_parent_ids = item["relevant_parent_ids"]
                ranked_for_mrr = parent_metrics_ids if mode.endswith("parent") else child_metrics_ids
                relevant_for_mrr = relevant_parent_ids if mode.endswith("parent") else relevant_child_ids
                metrics = {
                    "status": summary["status"],
                    "ranked_child_ids": child_metrics_ids[:k],
                    "ranked_parent_ids": parent_metrics_ids[:k],
                    "child_recall_at_k": recall_at_k(child_metrics_ids, relevant_child_ids, k),
                    "parent_recall_at_k": recall_at_k(parent_metrics_ids, relevant_parent_ids, k),
                    "mrr_at_k": mrr_at_k(ranked_for_mrr, relevant_for_mrr, k),
                    "ndcg_at_k": ndcg_at_k(ranked_for_mrr, relevant_for_mrr, k),
                    "unique_relevant_parents_retrieved": len(set(parent_metrics_ids[:k]) & set(relevant_parent_ids)),
                    "unique_relevant_sources_retrieved": summary["unique_relevant_sources_retrieved"],
                    "unique_relevant_parent_count": len(set(parent_metrics_ids[:k]) & set(relevant_parent_ids)),
                    "unique_relevant_source_count": summary["unique_relevant_sources_retrieved"],
                    "query_count": summary["query_count"],
                    "child_union_count": summary["child_union_count"],
                    "context_chars": summary["context_chars"],
                    "expansion_factor": summary["expansion_factor"],
                    "query_generation_call_count": summary["query_generation_call_count"],
                    "embedding_call_count": summary["embedding_call_count"],
                    "latency_ms": latency_ms,
                    "warnings": summary["warnings"],
                }
                if summary["status"] not in {"ready", "answered", "partial", "multi_query_partial"}:
                    failures.append({"question_id": item["question_id"], "mode": mode, "status": summary["status"]})
                _append_aggregate(aggregate_inputs[mode], metrics)
            except Exception as error:  # noqa: BLE001 - evaluation reports failures explicitly.
                latency_ms = (time.perf_counter() - started) * 1000
                metrics = {
                    "status": "failed",
                    "error": safe_error(error),
                    "ranked_child_ids": [],
                    "ranked_parent_ids": [],
                    "child_recall_at_k": 0.0,
                    "parent_recall_at_k": 0.0,
                    "mrr_at_k": 0.0,
                    "ndcg_at_k": 0.0,
                    "latency_ms": latency_ms,
                }
                failures.append({"question_id": item["question_id"], "mode": mode, "status": "failed", "error": metrics["error"]})
                aggregate_inputs[mode]["latency_ms"].append(latency_ms)
            item["modes"][mode] = metrics
        per_question.append(item)

    metrics_by_mode = {mode: aggregate_metrics(values, question_count=len(questions)) for mode, values in aggregate_inputs.items()}
    hierarchy_status = hr.hierarchy_status(storage_dir=hierarchy_storage)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": "hierarchical",
        "k": k,
        "modes": modes,
        "config_identity": hr._config_identity(config),
        "model_identity": {
            "embedding_model": config.gemini_embedding_model,
            "embedding_dim": config.gemini_embedding_dim,
            "generation_model": config.gemini_generation_model,
            "reranker_model": config.reranker_model,
        },
        "corpus_identity": hierarchy_status.get("schema_version"),
        "hierarchy_identity": {
            "storage_path": hierarchy_status.get("storage_path"),
            "schema_version": hierarchy_status.get("schema_version"),
            "counts": hierarchy_status.get("counts", {}),
            "warning_counts": hierarchy_status.get("warning_counts", {}),
        },
        "human_review_warning": bool(needs_review),
        "warnings": warnings,
        "official_winner": None,
        "metrics": metrics_by_mode,
        "failures": failures,
        "questions": per_question,
    }
    return report


def summarize_mode_result(result: dict[str, Any], *, mode: str, k: int) -> dict[str, Any]:
    trace = result.get("trace", {})
    if "modes" in result:
        payload = result["modes"][mode]
        if mode.endswith("parent"):
            retrieval = payload.get("retrieval", {})
            rerank = payload.get("rerank", {})
            parents = rerank.get("parents") or retrieval.get("parents", [])
            children = retrieval.get("children", [])
            status = rerank.get("status") or retrieval.get("status", "unknown")
            trace = retrieval.get("trace", {})
            warnings = retrieval.get("warnings", []) + ([rerank.get("error")] if rerank.get("error") else [])
        else:
            parents = []
            children = payload.get("child_hits", [])
            status = payload.get("status", "unknown")
            warnings = payload.get("warnings", [])
            trace = payload.get("trace", {})
    else:
        parents = result.get("parent_candidates") or result.get("parents", [])
        children = result.get("child_hits") or result.get("children", [])
        status = result.get("status", "unknown")
        warnings = result.get("warnings", [])
    ranked_parent_ids = [parent["parent_id"] for parent in parents if parent.get("parent_id")]
    ranked_child_ids = _ranked_child_ids(children, parents)
    api_counts = trace.get("api_call_counts", {}) if isinstance(trace, dict) else {}
    retrieval_trace = trace.get("retrieval_trace", trace) if isinstance(trace, dict) else {}
    return {
        "status": status,
        "ranked_child_ids": ranked_child_ids,
        "ranked_parent_ids": ranked_parent_ids,
        "unique_relevant_sources_retrieved": len({item.get("source") for item in list(parents) + list(children) if item.get("source")}),
        "query_count": retrieval_trace.get("query_count", {}).get("executed", len(result.get("query_set", result.get("queries", [])))) if isinstance(retrieval_trace, dict) else 0,
        "child_union_count": retrieval_trace.get("union_child_count", len(children)) if isinstance(retrieval_trace, dict) else len(children),
        "context_chars": retrieval_trace.get("expanded_parent_chars", sum(len(str(parent.get("text", ""))) for parent in parents)) if isinstance(retrieval_trace, dict) else 0,
        "expansion_factor": retrieval_trace.get("context_expansion_factor", 0.0) if isinstance(retrieval_trace, dict) else 0.0,
        "query_generation_call_count": api_counts.get("query_expansion_generation_calls", retrieval_trace.get("gemini_expansion_call_count", 0) if isinstance(retrieval_trace, dict) else 0),
        "embedding_call_count": api_counts.get("semantic_embedding_calls", retrieval_trace.get("semantic_embedding_call_count", 0) if isinstance(retrieval_trace, dict) else 0),
        "warnings": [warning for warning in warnings if warning],
    }


def retrieve_mode(question: dict[str, Any], mode: str, k: int, config: hr.HierarchyConfig) -> dict[str, Any]:
    text = question["question"]
    if mode.endswith("parent"):
        retrieval = hr.parent_retrieve(text, mode=mode, config=config)
        if retrieval.get("status") not in {"ready", "partial", "multi_query_partial"}:
            return {"modes": {mode: {"retrieval": retrieval, "rerank": {"status": retrieval.get("status"), "parents": []}}}}
        reranked = hr.rerank_parent_candidates(text, retrieval.get("parents", []), config=config)
        return {"modes": {mode: {"retrieval": retrieval, "rerank": reranked}}}
    flat = hr._flat_mode_retrieve_and_rerank(
        text,
        mode=mode,
        config=config,
        query_generator_fn=None,
        hybrid_retriever_fn=None,
        child_rerank_scorer=None,
        strategy="hierarchical",
        input_path=None,
        storage_path=None,
    )
    return {"modes": {mode: flat}}


def mock_retriever(question: dict[str, Any], mode: str, k: int, config: hr.HierarchyConfig) -> dict[str, Any]:
    relevant_children = list(question.get("relevant_child_ids", []))
    relevant_parents = list(question.get("relevant_parent_ids", []))
    children = [
        {
            "child_id": child_id,
            "text": f"mock child {child_id}",
            "source": "mock.pdf",
            "page_start": 1,
            "page_end": 1,
            "multi_query_rank": index,
            "support_query_ids": ["Q0"],
            "per_query_ranks": {"Q0": index},
        }
        for index, child_id in enumerate(relevant_children + [f"mock-decoy-child-{mode}"], start=1)
    ]
    parents = [
        {
            "parent_id": parent_id,
            "text": f"mock parent {parent_id}",
            "source": "mock.pdf",
            "page_start": 1,
            "page_end": 2,
            "parent_rank": index,
            "parent_rerank_rank": index,
            "supporting_child_ids": relevant_children[:1],
        }
        for index, parent_id in enumerate(relevant_parents + [f"mock-decoy-parent-{mode}"], start=1)
    ]
    if mode.endswith("parent"):
        return {
            "modes": {
                mode: {
                    "retrieval": {
                        "status": "ready",
                        "children": children,
                        "parents": parents,
                        "warnings": [],
                        "trace": {"query_count": {"executed": 1}, "union_child_count": len(children), "expanded_parent_chars": sum(len(p["text"]) for p in parents), "context_expansion_factor": 2.0, "semantic_embedding_call_count": 1},
                    },
                    "rerank": {"status": "ready", "parents": parents, "reranked_count": len(parents)},
                }
            }
        }
    return {
        "modes": {
            mode: {
                "status": "ready",
                "child_hits": children,
                "accepted_evidence": children[:k],
                "warnings": [],
                "trace": {"query_count": {"executed": 1}, "union_child_count": len(children), "semantic_embedding_call_count": 1},
            }
        }
    }


def validate_questions(questions: list[dict[str, Any]], *, hierarchy_storage: Path | str, config: hr.HierarchyConfig) -> None:
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions JSON phải là list không rỗng")
    required = {"question_id", "question", "question_type", "relevant_child_ids", "relevant_parent_ids", "needs_human_review", "notes"}
    store = hr.load_hierarchy_store(storage_dir=hierarchy_storage, config=config)
    parents_by_id = set(store.get("parents_by_id", {})) if store.get("status") == "ready" else set()
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"question #{index} phải là object")
        missing = required - set(question)
        if missing:
            raise ValueError(f"question #{index} thiếu field: {sorted(missing)}")
        if question["question_type"] not in {"exact", "paraphrase", "multi_aspect", "hierarchy_context", "out_of_scope"}:
            raise ValueError(f"question_type không hợp lệ tại {question['question_id']}")
        if not isinstance(question["relevant_child_ids"], list) or not isinstance(question["relevant_parent_ids"], list):
            raise ValueError(f"relevant ids phải là list tại {question['question_id']}")
        stale = [parent_id for parent_id in question["relevant_parent_ids"] if parents_by_id and parent_id not in parents_by_id]
        if stale:
            raise ValueError(f"stale relevant_parent_ids tại {question['question_id']}: {stale}")


def load_questions(path: Path | str = DEFAULT_QUESTIONS_PATH) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("questions JSON phải là list")
    return data


def save_report(report: dict[str, Any], output_path: Path | str | None = None) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        timestamp = report["timestamp"].replace(":", "").replace("+", "Z").split(".")[0]
        output_path = REPORTS_DIR / f"eval_buoi09_{timestamp}.json"
    path = Path(output_path)
    _atomic_write_json(path, report)
    latest_path = path.parent / "latest_report.json"
    _atomic_write_json(latest_path, report)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Buổi 09 - Retrieval-only evaluator")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH), help="Path tới eval questions JSON")
    parser.add_argument("--k", type=int, default=5, help="K cho Recall/MRR/nDCG")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES), help="Comma-separated modes")
    parser.add_argument("--output", default=None, help="Path report JSON")
    parser.add_argument("--mock-synthetic", action="store_true", help="Dùng deterministic fake retriever, không gọi API/model/storage thật")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    questions = load_questions(args.questions)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    retriever = mock_retriever if args.mock_synthetic else None
    report = evaluate_questions(questions, modes=modes, k=args.k, retriever=retriever)
    path = save_report(report, args.output)
    print(f"Saved report: {path}")
    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    print("Metrics:")
    for mode, metrics in report["metrics"].items():
        print(
            f"- {mode}: child_recall@{args.k}={metrics['child_recall_at_k']:.4f}, "
            f"parent_recall@{args.k}={metrics['parent_recall_at_k']:.4f}, "
            f"mrr@{args.k}={metrics['mrr_at_k']:.4f}, ndcg@{args.k}={metrics['ndcg_at_k']:.4f}, "
            f"latency_mean_ms={metrics['latency_mean_ms']:.3f}, failed={metrics['failed_count']}"
        )
    return 0


def _ranked_child_ids(children: list[dict[str, Any]], parents: list[dict[str, Any]]) -> list[str]:
    ids = [child.get("child_id") or child.get("chunk_id") for child in children]
    if not ids and parents:
        for parent in parents:
            ids.extend(parent.get("supporting_child_ids", []))
    return [str(item) for item in ids if item]


def _append_aggregate(target: dict[str, list[float]], metrics: dict[str, Any]) -> None:
    for key in target:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key].append(float(value))


def aggregate_metrics(values: dict[str, list[float]], *, question_count: int) -> dict[str, Any]:
    latencies = values.get("latency_ms", [])
    return {
        "child_recall_at_k": _mean(values.get("child_recall_at_k", [])),
        "parent_recall_at_k": _mean(values.get("parent_recall_at_k", [])),
        "mrr_at_k": _mean(values.get("mrr_at_k", [])),
        "ndcg_at_k": _mean(values.get("ndcg_at_k", [])),
        "unique_relevant_parents_retrieved": _mean(values.get("unique_relevant_parent_count", [])),
        "unique_relevant_sources_retrieved": _mean(values.get("unique_relevant_source_count", [])),
        "query_count_mean": _mean(values.get("query_count", [])),
        "child_union_count_mean": _mean(values.get("child_union_count", [])),
        "context_chars_mean": _mean(values.get("context_chars", [])),
        "expansion_factor_mean": _mean(values.get("expansion_factor", [])),
        "query_generation_call_count_mean": _mean(values.get("query_generation_call_count", [])),
        "embedding_call_count_mean": _mean(values.get("embedding_call_count", [])),
        "latency_mean_ms": _mean(latencies),
        "latency_p50_ms": statistics.median(latencies) if latencies else 0.0,
        "failed_count": max(0, question_count - len(values.get("mrr_at_k", []))),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def safe_error(error: Exception) -> str:
    message = f"{type(error).__name__}: {error}"
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        message = message.replace(api_key, "[secret]")
    return message[:500]


if __name__ == "__main__":
    raise SystemExit(main())
