"""Deterministic parent-child hierarchy registry for Buổi 09.

This module builds a local article-level parent registry from hierarchical child
chunks. It is intentionally offline: no Gemini calls, no retrieval, no reranker
loading, and no Chroma collection creation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

import rag

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
ENV_EXAMPLE_PATH = BASE_DIR / ".env.example"
HIERARCHY_STORAGE_DIR = BASE_DIR / "storage" / "hierarchy"
CHILDREN_PATH = HIERARCHY_STORAGE_DIR / "children.json"
PARENTS_PATH = HIERARCHY_STORAGE_DIR / "parents.json"
MANIFEST_PATH = HIERARCHY_STORAGE_DIR / "manifest.json"
SCHEMA_VERSION = "buoi_09_hierarchy_v1"
DEFAULT_INPUT_PATH = rag.BUOI_05_CHUNKS_DIR
ALLOWED_RESOLUTION_METHODS = {"metadata", "heading_inferred", "carried_forward", "document_fallback"}
STRUCTURE_KEYS = ("chapter", "article", "clause", "point")
QUERY_GENERATION_UNAVAILABLE = "query_generation_unavailable"
QUERY_GENERATION_READY = "ready"
QUERY_FOCUS_VALUES = {"exact_legal_terms", "paraphrase", "missing_aspect"}
MAX_QUESTION_CHARS = 2_000
_QUERY_EXPANSION_CACHE: dict[str, dict[str, Any]] = {}

CHAPTER_HEADING_RE = re.compile(r"^\s*#{0,6}\s*Chương\s+([IVXLCDM0-9]+[A-Z]?)\b", re.IGNORECASE)
ARTICLE_HEADING_RE = re.compile(r"^\s*#{0,6}\s*(?:\*\*)?Điều\s+([0-9]+[a-zA-Z]?)\b", re.IGNORECASE)
CLAUSE_HEADING_RE = re.compile(r"^\s*(\d+)\.\s+")
POINT_HEADING_RE = re.compile(r"^\s*([a-zđ])\)\s+", re.IGNORECASE)
INLINE_ARTICLE_RE = re.compile(r"\bĐiều\s+([0-9]+[a-zA-Z]?)\b", re.IGNORECASE)
LEGAL_REFERENCE_RE = re.compile(
    r"\b(?:Điều|Khoản|Điểm)\s+[0-9a-zA-ZđĐ]+\b|\b(?:19|20)\d{2}\b|\b\d+\s*/\s*\d+\s*/\s*[A-ZĐ-]+\b",
    re.IGNORECASE,
)
PUNCT_NORMALIZE_RE = re.compile(r"[\s\.,;:!?\-_/()\[\]{}\"'“”‘’]+")
FINAL_NUMBER_RE = re.compile(r"(\d+)(?!.*\d)")


@dataclass(frozen=True)
class HierarchyConfig:
    gemini_embedding_model: str
    gemini_embedding_dim: int
    gemini_generation_model: str
    bm25_candidates: int
    semantic_candidates: int
    rrf_k: int
    rrf_bm25_weight: float
    rrf_semantic_weight: float
    rerank_candidates: int
    final_top_k: int
    reranker_model: str
    reranker_min_score: float
    reranker_device: str
    multi_query_count: int
    multi_query_max_chars: int
    multi_query_temperature: float
    multi_query_original_weight: float
    multi_query_variant_weight: float
    multi_query_rrf_k: int
    per_query_candidates: int
    parent_max_chars: int
    parent_score_child_limit: int
    parent_rrf_k: int
    parent_candidates: int
    final_parent_top_k: int
    total_context_max_chars: int


def load_hierarchy_config(env_path: Path | str | None = None) -> HierarchyConfig:
    """Load and validate Buổi 09 config from a path independent of cwd."""
    path = Path(env_path) if env_path is not None else (ENV_PATH if ENV_PATH.exists() else ENV_EXAMPLE_PATH)
    load_dotenv(path.resolve(), override=False)

    config = HierarchyConfig(
        gemini_embedding_model=_required_env_string("GEMINI_EMBEDDING_MODEL"),
        gemini_embedding_dim=_required_env_int("GEMINI_EMBEDDING_DIM", min_value=128, max_value=3072),
        gemini_generation_model=_required_env_string("GEMINI_GENERATION_MODEL"),
        bm25_candidates=_required_env_int("BM25_CANDIDATES", min_value=1, max_value=100),
        semantic_candidates=_required_env_int("SEMANTIC_CANDIDATES", min_value=1, max_value=100),
        rrf_k=_required_env_int("RRF_K", min_value=1, max_value=10_000),
        rrf_bm25_weight=_required_env_float("RRF_BM25_WEIGHT", min_value=0.0, max_value=None),
        rrf_semantic_weight=_required_env_float("RRF_SEMANTIC_WEIGHT", min_value=0.0, max_value=None),
        rerank_candidates=_required_env_int("RERANK_CANDIDATES", min_value=1, max_value=100),
        final_top_k=_required_env_int("FINAL_TOP_K", min_value=1, max_value=100),
        reranker_model=_required_env_string("RERANKER_MODEL"),
        reranker_min_score=_required_env_float("RERANK_MIN_SCORE", min_value=0.0, max_value=1.0),
        reranker_device=_required_env_string("RERANK_DEVICE"),
        multi_query_count=_required_env_int("MULTI_QUERY_COUNT", min_value=1, max_value=5),
        multi_query_max_chars=_required_env_int("MULTI_QUERY_MAX_CHARS", min_value=50, max_value=1000),
        multi_query_temperature=_required_env_float("MULTI_QUERY_TEMPERATURE", min_value=0.0, max_value=1.0),
        multi_query_original_weight=_required_env_float("MULTI_QUERY_ORIGINAL_WEIGHT", min_value=0.0, max_value=None),
        multi_query_variant_weight=_required_env_float("MULTI_QUERY_VARIANT_WEIGHT", min_value=0.0, max_value=None),
        multi_query_rrf_k=_required_env_int("MULTI_QUERY_RRF_K", min_value=1, max_value=10_000),
        per_query_candidates=_required_env_int("PER_QUERY_CANDIDATES", min_value=1, max_value=100),
        parent_max_chars=_required_env_int("PARENT_MAX_CHARS", min_value=1000, max_value=20_000),
        parent_score_child_limit=_required_env_int("PARENT_SCORE_CHILD_LIMIT", min_value=1, max_value=20),
        parent_rrf_k=_required_env_int("PARENT_RRF_K", min_value=1, max_value=10_000),
        parent_candidates=_required_env_int("PARENT_CANDIDATES", min_value=1, max_value=100),
        final_parent_top_k=_required_env_int("FINAL_PARENT_TOP_K", min_value=1, max_value=100),
        total_context_max_chars=_required_env_int("TOTAL_CONTEXT_MAX_CHARS", min_value=1, max_value=1_000_000),
    )

    if config.rrf_bm25_weight == 0.0 and config.rrf_semantic_weight == 0.0:
        raise ValueError("RRF_BM25_WEIGHT và RRF_SEMANTIC_WEIGHT không được đồng thời bằng 0")
    if config.multi_query_original_weight == 0.0 and config.multi_query_variant_weight == 0.0:
        raise ValueError("MULTI_QUERY_ORIGINAL_WEIGHT và MULTI_QUERY_VARIANT_WEIGHT không được đồng thời bằng 0")
    if config.final_parent_top_k > config.parent_candidates:
        raise ValueError("FINAL_PARENT_TOP_K phải <= PARENT_CANDIDATES")
    if config.total_context_max_chars < config.parent_max_chars:
        raise ValueError("TOTAL_CONTEXT_MAX_CHARS phải >= PARENT_MAX_CHARS")
    if config.final_top_k > config.rerank_candidates:
        raise ValueError("FINAL_TOP_K phải <= RERANK_CANDIDATES")
    if config.reranker_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("RERANK_DEVICE chỉ nhận auto, cpu hoặc cuda")
    return config


def hierarchy_audit(input_path: Path | str = DEFAULT_INPUT_PATH, *, config: HierarchyConfig | None = None) -> dict[str, Any]:
    """Read hierarchical chunks and return registry statistics without writing files."""
    config = config or load_hierarchy_config()
    raw_children, load_stats = load_hierarchical_children(input_path)
    resolved_children = resolve_children(raw_children)
    parents = build_parent_documents(resolved_children, config=config)
    return _summary(load_stats=load_stats, children=resolved_children, parents=parents, config=config)


def build_hierarchy_store(
    input_path: Path | str = DEFAULT_INPUT_PATH,
    *,
    storage_dir: Path | str = HIERARCHY_STORAGE_DIR,
    config: HierarchyConfig | None = None,
) -> dict[str, Any]:
    """Build hierarchy JSON store atomically after all validation succeeds."""
    config = config or load_hierarchy_config()
    raw_children, load_stats = load_hierarchical_children(input_path)
    resolved_children = resolve_children(raw_children)
    parents = build_parent_documents(resolved_children, config=config)
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    final_children = children_with_parent_ids(resolved_children, parents)
    children_payload = [child_to_json(child) for child in final_children]
    parents_payload = [parent_to_json(parent) for parent in parents]
    manifest = build_manifest(
        input_path=input_path,
        load_stats=load_stats,
        children=resolved_children,
        parents=parents,
        config=config,
    )

    _atomic_write_store(
        storage_dir,
        {
            "children.json": children_payload,
            "parents.json": parents_payload,
            "manifest.json": manifest,
        },
    )
    return {"children": children_payload, "parents": parents_payload, "manifest": manifest}


def hierarchy_status(*, storage_dir: Path | str = HIERARCHY_STORAGE_DIR) -> dict[str, Any]:
    """Read hierarchy store status without creating directories or modifying files."""
    storage_dir = Path(storage_dir)
    paths = {
        "children": storage_dir / "children.json",
        "parents": storage_dir / "parents.json",
        "manifest": storage_dir / "manifest.json",
    }
    status: dict[str, Any] = {
        "storage_path": str(storage_dir),
        "exists": all(path.exists() for path in paths.values()),
        "files": {},
        "schema_version": None,
        "counts": {},
        "warning_counts": {},
    }
    for name, path in paths.items():
        status["files"][name] = {
            "path": str(path),
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
            "mtime": path.stat().st_mtime if path.exists() else None,
        }
    if paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        status["schema_version"] = manifest.get("schema_version")
        status["counts"] = manifest.get("counts", {})
        status["warning_counts"] = manifest.get("warning_counts", {})
    return status


@dataclass(frozen=True)
class RawChild:
    chunk_id: str
    source: str
    page_start: int
    page_end: int
    text: str
    structure: dict[str, str | None]
    json_file: str
    record_no: int
    sequence: int | None


@dataclass(frozen=True)
class ResolvedChild:
    child_id: str
    parent_id: str
    source: str
    page_start: int
    page_end: int
    text: str
    structural_path: dict[str, str | None]
    resolution_method: str
    ambiguous: bool
    warnings: list[str]
    article_key: str
    parent_title: str
    sequence: int | None


@dataclass(frozen=True)
class ParentDocument:
    parent_id: str
    source: str
    page_start: int
    page_end: int
    article_key: str
    window_index: int
    child_ids: list[str]
    text: str
    char_count: int
    ambiguous_child_count: int
    warnings: list[str]


QueryGeneratorFn = Callable[[str, HierarchyConfig], Any]
HybridRetrieverFn = Callable[..., dict[str, Any]]
ChildRetrievalFn = Callable[..., dict[str, Any]]
ParentRerankScorer = Callable[[str, list[dict[str, Any]], dict[str, Any]], list[float]]
ChildRerankScorer = Callable[[str, list[dict[str, Any]], dict[str, Any]], list[float]]
AnswerGenerator = Callable[[str, dict[str, Any]], str]
PARENT_RETRIEVAL_MODES = {"single_parent", "multi_parent"}
ANSWER_MODES = {"single_flat", "multi_flat", "single_parent", "multi_parent"}
HIERARCHY_NOT_READY = "hierarchy_not_ready"


def expand_query(
    question: str,
    *,
    config: HierarchyConfig | None = None,
    query_generator_fn: QueryGeneratorFn | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Generate a validated Buổi 09 query set with Q0 preserved first."""
    config = config or load_hierarchy_config()
    started = time.perf_counter()
    try:
        original_question = _normalize_question(question)
    except ValueError as error:
        return _query_generation_error(question=str(question), config=config, error=error, latency_ms=_elapsed_ms(started))

    cache_key = _query_cache_key(original_question, config)
    if use_cache and cache_key in _QUERY_EXPANSION_CACHE:
        cached = copy.deepcopy(_QUERY_EXPANSION_CACHE[cache_key])
        cached["cache_hit"] = True
        cached["generation_latency_ms"] = 0.0
        return cached

    try:
        raw_generated = _call_query_generator(original_question, config=config, query_generator_fn=query_generator_fn)
        result = _validate_query_set(
            original_question=original_question,
            raw_generated=raw_generated,
            config=config,
            latency_ms=_elapsed_ms(started),
        )
    except Exception as error:  # noqa: BLE001 - public status must be explicit and bounded.
        return _query_generation_error(question=original_question, config=config, error=error, latency_ms=_elapsed_ms(started))

    if use_cache and result["status"] == QUERY_GENERATION_READY:
        _QUERY_EXPANSION_CACHE[cache_key] = copy.deepcopy(result)
    return result


def multi_child_retrieve(
    question: str,
    *,
    config: HierarchyConfig | None = None,
    query_generator_fn: QueryGeneratorFn | None = None,
    hybrid_retriever_fn: HybridRetrieverFn | None = None,
    strategy: str = "hierarchical",
    input_path: Path | str | None = None,
    storage_path: Path | str | None = None,
) -> dict[str, Any]:
    """Fan out hybrid child retrieval over Q0..Qn and fuse with cross-query RRF."""
    config = config or load_hierarchy_config()
    started_total = time.perf_counter()
    expansion = expand_query(question, config=config, query_generator_fn=query_generator_fn, use_cache=query_generator_fn is None)
    if expansion["status"] != QUERY_GENERATION_READY:
        return {
            "status": expansion["status"],
            "mode": "multi_child",
            "question": question,
            "queries": [],
            "children": [],
            "warnings": [expansion.get("error", "query_generation_unavailable")],
            "query_errors": {},
            "trace": _multi_child_empty_trace(expansion=expansion, total_started=started_total),
        }

    queries = _weighted_query_list(expansion["queries"], config=config)
    hybrid_retriever_fn = hybrid_retriever_fn or _default_hybrid_retriever
    per_query_results: dict[str, dict[str, Any]] = {}
    query_errors: dict[str, str] = {}
    executed_queries = 0
    semantic_embedding_call_count = 0
    retrieval_started_all = time.perf_counter()

    for query in queries:
        query_id = query["query_id"]
        started = time.perf_counter()
        try:
            retrieval = hybrid_retriever_fn(
                query["text"],
                strategy=strategy,
                input_path=input_path,
                storage_path=storage_path,
            )
            candidates = _per_query_candidates(retrieval, limit=config.per_query_candidates, query_id=query_id)
            executed_queries += 1
            semantic_embedding_call_count += int(retrieval.get("trace", {}).get("semantic_embedding_call_count", 1))
            per_query_results[query_id] = {
                "query": query,
                "candidates": candidates,
                "latency_ms": _elapsed_ms(started),
                "result_count": len(candidates),
                "hybrid_trace": retrieval.get("trace", {}),
            }
        except Exception as error:  # noqa: BLE001 - generated-query partial status is part of the contract.
            if query_id == "Q0":
                raise
            query_errors[query_id] = _safe_error_message(error)
            per_query_results[query_id] = {
                "query": query,
                "candidates": [],
                "latency_ms": _elapsed_ms(started),
                "result_count": 0,
                "hybrid_trace": {},
                "error": query_errors[query_id],
            }

    fusion_started = time.perf_counter()
    fused_children = fuse_multi_query_children(
        per_query_results,
        queries=queries,
        rrf_k=config.multi_query_rrf_k,
    )
    fusion_latency_ms = _elapsed_ms(fusion_started)
    status = _multi_child_status(queries=queries, query_errors=query_errors)
    warnings = _multi_child_warnings(status=status, query_errors=query_errors)
    trace = _multi_child_trace(
        expansion=expansion,
        queries=queries,
        per_query_results=per_query_results,
        query_errors=query_errors,
        children=fused_children,
        fusion_latency_ms=fusion_latency_ms,
        total_latency_ms=_elapsed_ms(started_total),
        retrieval_latency_ms=_elapsed_ms(retrieval_started_all),
        semantic_embedding_call_count=semantic_embedding_call_count,
        gemini_expansion_call_count=0 if query_generator_fn is not None or expansion.get("cache_hit") else 1,
    )
    return {
        "status": status,
        "mode": "multi_child",
        "question": expansion["original_question"],
        "strategy": strategy,
        "queries": queries,
        "children": fused_children,
        "warnings": warnings,
        "query_errors": query_errors,
        "trace": trace,
    }


def fuse_multi_query_children(
    per_query_results: dict[str, dict[str, Any]],
    *,
    queries: list[dict[str, Any]],
    rrf_k: int,
) -> list[dict[str, Any]]:
    """Union child hits by ID and compute weighted cross-query RRF."""
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k <= 0:
        raise ValueError("MULTI_QUERY_RRF_K phải là integer dương")
    query_order = {query["query_id"]: index for index, query in enumerate(queries)}
    query_weight = {query["query_id"]: _query_weight(query) for query in queries}
    fused_by_id: dict[str, dict[str, Any]] = {}

    for query in queries:
        query_id = query["query_id"]
        result = per_query_results.get(query_id, {})
        for candidate in result.get("candidates", []):
            child_id = _candidate_child_id(candidate)
            fused = fused_by_id.setdefault(child_id, _multi_child_base(candidate))
            _ensure_multi_child_metadata_consistent(fused, candidate, query_id=query_id)
            rank = _candidate_fused_rank(candidate)
            fused["support_query_ids"].append(query_id)
            fused["per_query_ranks"][query_id] = rank
            fused["per_query_trace"][query_id] = _candidate_per_query_trace(candidate)
            fused["multi_query_rrf_score"] += query_weight[query_id] / (rrf_k + rank)

    children: list[dict[str, Any]] = []
    for item in fused_by_id.values():
        item["support_query_ids"] = sorted(set(item["support_query_ids"]), key=lambda query_id: query_order[query_id])
        item["support_query_count"] = len(item["support_query_ids"])
        item["best_query_rank"] = min(item["per_query_ranks"].values())
        children.append(item)

    children.sort(key=lambda item: (-item["multi_query_rrf_score"], -item["support_query_count"], item["best_query_rank"], item["child_id"]))
    for rank, item in enumerate(children, start=1):
        item["multi_query_rank"] = rank
    return children


def parent_retrieve(
    question: str,
    *,
    mode: str = "multi_parent",
    config: HierarchyConfig | None = None,
    storage_dir: Path | str = HIERARCHY_STORAGE_DIR,
    query_generator_fn: QueryGeneratorFn | None = None,
    hybrid_retriever_fn: HybridRetrieverFn | None = None,
    child_retrieval_fn: ChildRetrievalFn | None = None,
    strategy: str = "hierarchical",
    input_path: Path | str | None = None,
    storage_path: Path | str | None = None,
) -> dict[str, Any]:
    """Retrieve child hits, map them to stored parents, aggregate, and budget context."""
    config = config or load_hierarchy_config()
    if mode not in PARENT_RETRIEVAL_MODES:
        raise ValueError(f"mode chỉ nhận {sorted(PARENT_RETRIEVAL_MODES)}")
    started = time.perf_counter()
    store = load_hierarchy_store(storage_dir=storage_dir, config=config, input_path=input_path)
    if store["status"] != QUERY_GENERATION_READY:
        return {
            "status": HIERARCHY_NOT_READY,
            "mode": mode,
            "question": question,
            "parents": [],
            "children": [],
            "warnings": store["warnings"],
            "trace": _parent_empty_trace(total_started=started),
        }

    child_started = time.perf_counter()
    if child_retrieval_fn is not None:
        child_result = child_retrieval_fn(question, mode=mode, config=config)
    elif mode == "single_parent":
        child_result = _single_child_retrieve(
            question,
            config=config,
            hybrid_retriever_fn=hybrid_retriever_fn,
            strategy=strategy,
            input_path=input_path,
            storage_path=storage_path,
        )
    else:
        child_result = multi_child_retrieve(
            question,
            config=config,
            query_generator_fn=query_generator_fn,
            hybrid_retriever_fn=hybrid_retriever_fn,
            strategy=strategy,
            input_path=input_path,
            storage_path=storage_path,
        )
    child_latency_ms = _elapsed_ms(child_started)
    if child_result.get("status") not in {QUERY_GENERATION_READY, "partial", "multi_query_partial"}:
        return {
            "status": child_result.get("status", "child_retrieval_failed"),
            "mode": mode,
            "question": question,
            "parents": [],
            "children": child_result.get("children", []),
            "warnings": child_result.get("warnings", []),
            "trace": _parent_empty_trace(total_started=started),
        }

    mapping_started = time.perf_counter()
    candidates, aggregation_trace = aggregate_parent_candidates(
        child_result.get("children", []),
        child_registry=store["children_by_id"],
        parents_by_id=store["parents_by_id"],
        config=config,
    )
    mapping_aggregation_latency_ms = _elapsed_ms(mapping_started)
    budget_started = time.perf_counter()
    selected, budget_trace, budget_warnings = apply_parent_context_budget(candidates, config=config)
    budget_latency_ms = _elapsed_ms(budget_started)
    warnings = list(child_result.get("warnings", [])) + budget_warnings + store.get("warnings", [])
    trace = _parent_trace(
        children=child_result.get("children", []),
        candidates=candidates,
        selected=selected,
        aggregation_trace=aggregation_trace,
        budget_trace=budget_trace,
        child_latency_ms=child_latency_ms,
        mapping_aggregation_latency_ms=mapping_aggregation_latency_ms,
        budget_latency_ms=budget_latency_ms,
        total_latency_ms=_elapsed_ms(started),
    )
    return {
        "status": child_result.get("status", QUERY_GENERATION_READY),
        "mode": mode,
        "question": child_result.get("question", question),
        "queries": child_result.get("queries", []),
        "children": child_result.get("children", []),
        "parents": selected,
        "warnings": warnings,
        "trace": trace,
    }


def aggregate_parent_candidates(
    child_hits: list[dict[str, Any]],
    *,
    child_registry: dict[str, dict[str, Any]],
    parents_by_id: dict[str, dict[str, Any]],
    config: HierarchyConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map fused child hits to parents and aggregate parent RRF scores."""
    grouped: dict[str, dict[str, Any]] = {}
    mapping_table: list[dict[str, Any]] = []
    child_text_by_hash: dict[str, tuple[str, str]] = {}
    for child in child_hits:
        child_id = _required_text_field(child, "child_id", source="child_hit")
        registry_child = child_registry.get(child_id)
        if registry_child is None:
            raise ValueError(f"Không tìm thấy child trong hierarchy registry: {child_id}")
        parent_id = _required_text_field(registry_child, "parent_id", source=child_id)
        parent = parents_by_id.get(parent_id)
        if parent is None:
            raise ValueError(f"Không tìm thấy parent trong hierarchy store: {parent_id} cho child {child_id}")
        text_hash = hashlib.sha256(child["text"].encode("utf-8")).hexdigest()
        previous = child_text_by_hash.get(text_hash)
        if previous and previous[1] != parent_id:
            raise ValueError(f"Duplicate child text across parents: {previous[0]} và {child_id}")
        child_text_by_hash[text_hash] = (child_id, parent_id)

        group = grouped.setdefault(
            parent_id,
            {
                "parent": parent,
                "children": [],
                "support_query_ids": set(),
                "structural_path": registry_child.get("structural_path", {}),
            },
        )
        group["children"].append(child)
        group["support_query_ids"].update(child.get("support_query_ids", []))
        mapping_table.append(
            {
                "child_id": child_id,
                "multi_query_rank": child.get("multi_query_rank"),
                "parent_id": parent_id,
                "support_query_ids": list(child.get("support_query_ids", [])),
            }
        )

    candidates: list[dict[str, Any]] = []
    score_components: dict[str, list[dict[str, Any]]] = {}
    child_count_by_parent: dict[str, int] = {}
    for parent_id, group in grouped.items():
        children = sorted(group["children"], key=lambda item: (item["multi_query_rank"], item["child_id"]))
        scoring = children[: config.parent_score_child_limit]
        score = sum(1.0 / (config.parent_rrf_k + child["multi_query_rank"]) for child in scoring)
        parent = group["parent"]
        supporting_ids = [child["child_id"] for child in children]
        scoring_ids = [child["child_id"] for child in scoring]
        support_query_ids = sorted(group["support_query_ids"], key=_query_id_sort_key)
        warnings = list(parent.get("warnings", []))
        ambiguous = bool(parent.get("ambiguous", False) or parent.get("ambiguous_child_count", 0))
        if ambiguous:
            warnings.append("ambiguous_parent_resolution")
        candidate = {
            "parent_id": parent_id,
            "source": _required_text_field(parent, "source", source=parent_id),
            "page_start": _required_int_field(parent, "page_start", source=parent_id),
            "page_end": _required_int_field(parent, "page_end", source=parent_id),
            "structural_path": parent.get("structural_path") or group.get("structural_path") or _structural_path_from_parent(parent),
            "text": _required_text_field(parent, "text", source=parent_id),
            "parent_rrf_score": score,
            "parent_rank": None,
            "anchor_child_id": children[0]["child_id"],
            "scoring_child_ids": scoring_ids,
            "supporting_child_ids": supporting_ids,
            "support_query_ids": support_query_ids,
            "best_child_rank": children[0]["multi_query_rank"],
            "ambiguous": ambiguous,
            "warnings": sorted(set(warnings)),
        }
        candidates.append(candidate)
        score_components[parent_id] = [{"child_id": child["child_id"], "multi_query_rank": child["multi_query_rank"], "component": 1.0 / (config.parent_rrf_k + child["multi_query_rank"])} for child in scoring]
        child_count_by_parent[parent_id] = len(children)

    candidates.sort(key=lambda item: (-item["parent_rrf_score"], -len(item["support_query_ids"]), item["best_child_rank"], item["parent_id"]))
    for rank, candidate in enumerate(candidates, start=1):
        candidate["parent_rank"] = rank
    limited = candidates[: config.parent_candidates]
    return limited, {
        "input_parent_count": len(candidates),
        "child_count_by_parent": child_count_by_parent,
        "child_to_parent_mapping": mapping_table,
        "parent_score_components": score_components,
        "parents_dropped_by_candidate_limit": [candidate["parent_id"] for candidate in candidates[config.parent_candidates :]],
    }


def apply_parent_context_budget(candidates: list[dict[str, Any]], *, config: HierarchyConfig) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    selected: list[dict[str, Any]] = []
    dropped: list[str] = []
    warnings: list[str] = []
    used_chars = 0
    seen: set[str] = set()
    for parent in candidates:
        parent_id = parent["parent_id"]
        if parent_id in seen:
            continue
        seen.add(parent_id)
        text_len = len(parent["text"])
        if not selected and text_len > config.total_context_max_chars:
            item = dict(parent)
            item["warnings"] = sorted(set(item.get("warnings", []) + ["first_parent_exceeds_total_context_budget"]))
            selected.append(item)
            used_chars += text_len
            warnings.append("first_parent_exceeds_total_context_budget")
            continue
        if used_chars + text_len <= config.total_context_max_chars:
            selected.append(parent)
            used_chars += text_len
        else:
            dropped.append(parent_id)
    return selected, {"used_context_chars": used_chars, "total_context_max_chars": config.total_context_max_chars, "parents_dropped_by_context_budget": dropped}, warnings


def load_hierarchy_store(*, storage_dir: Path | str = HIERARCHY_STORAGE_DIR, config: HierarchyConfig | None = None, input_path: Path | str | None = None) -> dict[str, Any]:
    storage_dir = Path(storage_dir)
    paths = {"children": storage_dir / "children.json", "parents": storage_dir / "parents.json", "manifest": storage_dir / "manifest.json"}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return {"status": HIERARCHY_NOT_READY, "warnings": [f"missing_hierarchy_store_files: {','.join(missing)}"]}
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        children = json.loads(paths["children"].read_text(encoding="utf-8"))
        parents = json.loads(paths["parents"].read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        return {"status": HIERARCHY_NOT_READY, "warnings": [f"read_hierarchy_store_failed: {_safe_error_message(error)}"]}
    manifest_input_path = manifest.get("input_path") if isinstance(manifest.get("input_path"), str) else None
    warnings = _hierarchy_store_readiness_warnings(manifest, config=config, input_path=input_path or manifest_input_path)
    if warnings:
        return {"status": HIERARCHY_NOT_READY, "warnings": warnings, "manifest": manifest}
    return {
        "status": QUERY_GENERATION_READY,
        "warnings": [],
        "manifest": manifest,
        "children_by_id": {child["child_id"]: child for child in children},
        "parents_by_id": {parent["parent_id"]: parent for parent in parents},
    }


def rerank_parent_candidates(
    original_question: str,
    parent_candidates: list[dict[str, Any]],
    *,
    config: HierarchyConfig,
    scorer: ParentRerankScorer | None = None,
) -> dict[str, Any]:
    """Rerank parent candidates with cross-encoder pairs (Q0, parent_text)."""
    candidates = sorted(parent_candidates, key=lambda item: item["parent_rank"])[: config.parent_candidates]
    if not candidates:
        return {"status": QUERY_GENERATION_READY, "parents": [], "reranked_count": 0, "latency_ms": 0.0}
    scorer = scorer or _default_parent_rerank_scorer
    started = time.perf_counter()
    try:
        raw_scores = scorer(original_question, candidates, _advanced_config_for_rerank(config))
    except Exception as error:  # noqa: BLE001
        return {"status": "reranker_unavailable", "parents": [], "reranked_count": 0, "latency_ms": _elapsed_ms(started), "error": _safe_error_message(error)}
    if len(raw_scores) != len(candidates):
        return {"status": "reranker_unavailable", "parents": [], "reranked_count": 0, "latency_ms": _elapsed_ms(started), "error": "parent_rerank_score_count_mismatch"}
    scored: list[dict[str, Any]] = []
    for parent, raw_score in zip(candidates, raw_scores):
        score = _finite_float(raw_score, name="parent_rerank_raw_score")
        item = dict(parent)
        item["parent_rerank_raw_score"] = score
        item["parent_rerank_score"] = _sigmoid(score)
        scored.append(item)
    scored.sort(key=lambda item: (-item["parent_rerank_score"], item["parent_rank"], item["parent_id"]))
    for rank, item in enumerate(scored, start=1):
        item["parent_rerank_rank"] = rank
        item["parent_rank_change"] = item["parent_rank"] - rank
    return {"status": QUERY_GENERATION_READY, "parents": scored[: config.final_parent_top_k], "reranked_count": len(scored), "latency_ms": _elapsed_ms(started)}


def answer_query_buoi09(
    question: str,
    *,
    mode: str = "multi_parent",
    config: HierarchyConfig | None = None,
    query_generator_fn: QueryGeneratorFn | None = None,
    hybrid_retriever_fn: HybridRetrieverFn | None = None,
    child_retrieval_fn: ChildRetrievalFn | None = None,
    parent_rerank_scorer: ParentRerankScorer | None = None,
    child_rerank_scorer: ChildRerankScorer | None = None,
    answer_generator: AnswerGenerator | None = None,
    storage_dir: Path | str = HIERARCHY_STORAGE_DIR,
    strategy: str = "hierarchical",
    input_path: Path | str | None = None,
    storage_path: Path | str | None = None,
) -> dict[str, Any]:
    """Route Buổi 09 modes through retrieval/rerank/gate/generation."""
    config = config or load_hierarchy_config()
    if mode not in ANSWER_MODES:
        raise ValueError(f"mode chỉ nhận {sorted(ANSWER_MODES)}")
    started = time.perf_counter()
    original_question = _normalize_question(question)
    api_counts = {"generation_calls": 0, "query_expansion_generation_calls": 0, "answer_generation_calls": 0, "semantic_embedding_calls": 0, "rerank_calls": 0}
    warnings: list[str] = []

    if mode.endswith("parent"):
        if child_retrieval_fn is not None:
            retrieval = child_retrieval_fn(original_question, mode=mode, config=config)
        else:
            retrieval = parent_retrieve(
                original_question,
                mode=mode,
                config=config,
                storage_dir=storage_dir,
                query_generator_fn=query_generator_fn,
                hybrid_retriever_fn=hybrid_retriever_fn,
                child_retrieval_fn=None,
                strategy=strategy,
                input_path=input_path,
                storage_path=storage_path,
            )
        api_counts.update(_counts_from_parent_retrieval(retrieval, query_generator_fn=query_generator_fn))
        if retrieval["status"] in {QUERY_GENERATION_UNAVAILABLE, HIERARCHY_NOT_READY}:
            return _final_answer_result(status=retrieval["status"], mode=mode, original_question=original_question, query_set=retrieval.get("queries", []), child_hits=retrieval.get("children", []), parent_candidates=retrieval.get("parents", []), accepted=[], answer="", citations=[], warnings=retrieval.get("warnings", []), error=retrieval.get("error"), api_counts=api_counts, started=started, config=config, retrieval_trace=retrieval.get("trace", {}))
        reranked = rerank_parent_candidates(original_question, retrieval["parents"], config=config, scorer=parent_rerank_scorer)
        api_counts["rerank_calls"] += 1 if retrieval.get("parents") else 0
        if reranked["status"] == "reranker_unavailable":
            return _final_answer_result(status="reranker_unavailable", mode=mode, original_question=original_question, query_set=retrieval.get("queries", []), child_hits=retrieval.get("children", []), parent_candidates=retrieval.get("parents", []), accepted=[], answer="", citations=[], warnings=warnings + [reranked.get("error", "reranker_unavailable")], error=reranked.get("error"), api_counts=api_counts, started=started, config=config, retrieval_trace=retrieval.get("trace", {}), rerank_trace=reranked)
        accepted = _accepted_parent_evidence(reranked["parents"], config=config)
        if not accepted:
            return _final_answer_result(status="insufficient_evidence", mode=mode, original_question=original_question, query_set=retrieval.get("queries", []), child_hits=retrieval.get("children", []), parent_candidates=reranked["parents"], accepted=[], answer="", citations=[], warnings=warnings, api_counts=api_counts, started=started, config=config, retrieval_trace=retrieval.get("trace", {}), rerank_trace=reranked)
        prompt = _build_parent_answer_prompt(original_question, accepted)
        generated = _call_answer_generator(prompt, config=config, answer_generator=answer_generator)
        api_counts["generation_calls"] += 0 if answer_generator is not None else 1
        api_counts["answer_generation_calls"] += 0 if answer_generator is not None else 1
        answer, citations, citation_warnings = _validate_parent_citations(generated, accepted)
        status = "answered" if not citation_warnings else "citation_validation_failed"
        return _final_answer_result(status=status, mode=mode, original_question=original_question, query_set=retrieval.get("queries", []), child_hits=retrieval.get("children", []), parent_candidates=reranked["parents"], accepted=accepted, answer=answer if status == "answered" else "", citations=citations, warnings=warnings + citation_warnings, api_counts=api_counts, started=started, config=config, retrieval_trace=retrieval.get("trace", {}), rerank_trace=reranked)

    flat = _flat_mode_retrieve_and_rerank(
        original_question,
        mode=mode,
        config=config,
        query_generator_fn=query_generator_fn,
        hybrid_retriever_fn=hybrid_retriever_fn,
        child_rerank_scorer=child_rerank_scorer,
        strategy=strategy,
        input_path=input_path,
        storage_path=storage_path,
    )
    api_counts.update(flat.get("api_counts", {}))
    if flat["status"] in {QUERY_GENERATION_UNAVAILABLE, "reranker_unavailable"}:
        return _final_answer_result(status=flat["status"], mode=mode, original_question=original_question, query_set=flat.get("query_set", []), child_hits=flat.get("child_hits", []), parent_candidates=[], accepted=[], answer="", citations=[], warnings=flat.get("warnings", []), error=flat.get("error"), api_counts=flat.get("api_counts", api_counts), started=started, config=config, retrieval_trace=flat.get("trace", {}))
    api_counts = flat.get("api_counts", api_counts)
    accepted_flat = flat["accepted_evidence"]
    if not accepted_flat:
        return _final_answer_result(status="insufficient_evidence", mode=mode, original_question=original_question, query_set=flat.get("query_set", []), child_hits=flat.get("child_hits", []), parent_candidates=[], accepted=[], answer="", citations=[], warnings=flat.get("warnings", []), api_counts=api_counts, started=started, config=config, retrieval_trace=flat.get("trace", {}))
    prompt = _build_flat_answer_prompt(original_question, accepted_flat)
    generated = _call_answer_generator(prompt, config=config, answer_generator=answer_generator)
    api_counts["generation_calls"] += 0 if answer_generator is not None else 1
    api_counts["answer_generation_calls"] += 0 if answer_generator is not None else 1
    answer, citations, citation_warnings = _validate_flat_citations(generated, accepted_flat)
    status = "answered" if not citation_warnings else "citation_validation_failed"
    return _final_answer_result(status=status, mode=mode, original_question=original_question, query_set=flat.get("query_set", []), child_hits=flat.get("child_hits", []), parent_candidates=[], accepted=accepted_flat, answer=answer if status == "answered" else "", citations=citations, warnings=flat.get("warnings", []) + citation_warnings, api_counts=api_counts, started=started, config=config, retrieval_trace=flat.get("trace", {}))


def compare_buoi09(
    question: str,
    *,
    config: HierarchyConfig | None = None,
    query_generator_fn: QueryGeneratorFn | None = None,
    hybrid_retriever_fn: HybridRetrieverFn | None = None,
    child_retrieval_fn: ChildRetrievalFn | None = None,
    parent_rerank_scorer: ParentRerankScorer | None = None,
    child_rerank_scorer: ChildRerankScorer | None = None,
    storage_dir: Path | str = HIERARCHY_STORAGE_DIR,
    strategy: str = "hierarchical",
    input_path: Path | str | None = None,
    storage_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run four Buổi 09 modes through retrieval/rerank only; no answer generation."""
    config = config or load_hierarchy_config()
    modes: dict[str, Any] = {}
    for mode in ("single_flat", "multi_flat", "single_parent", "multi_parent"):
        if mode.endswith("parent"):
            retrieval = parent_retrieve(question, mode=mode, config=config, storage_dir=storage_dir, query_generator_fn=query_generator_fn, hybrid_retriever_fn=hybrid_retriever_fn, child_retrieval_fn=child_retrieval_fn, strategy=strategy, input_path=input_path, storage_path=storage_path)
            reranked = {"status": retrieval.get("status"), "parents": []}
            if retrieval.get("status") in {QUERY_GENERATION_READY, "partial", "multi_query_partial"}:
                reranked = rerank_parent_candidates(question, retrieval.get("parents", []), config=config, scorer=parent_rerank_scorer)
            modes[mode] = {"retrieval": retrieval, "rerank": reranked}
        else:
            modes[mode] = _flat_mode_retrieve_and_rerank(question, mode=mode, config=config, query_generator_fn=query_generator_fn, hybrid_retriever_fn=hybrid_retriever_fn, child_rerank_scorer=child_rerank_scorer, strategy=strategy, input_path=input_path, storage_path=storage_path)
    return {"question": question, "modes": modes, "answer_generation_called": False}


def load_hierarchical_children(input_path: Path | str = DEFAULT_INPUT_PATH) -> tuple[list[RawChild], dict[str, Any]]:
    """Load only hierarchical strategy records and validate input with file/record context."""
    json_files = _json_files(input_path)
    records: list[RawChild] = []
    seen_ids: dict[str, tuple[str, int]] = {}
    stats: dict[str, Any] = {"files_read": 0, "total_records": 0, "selected_records": 0, "valid_children": 0, "input_files": []}

    for json_file in json_files:
        data = _read_records(json_file)
        stats["files_read"] += 1
        stats["total_records"] += len(data)
        stats["input_files"].append(_fingerprint_file(json_file))
        for record_no, record in enumerate(data, start=1):
            _ensure_object(record, json_file, record_no)
            strategy = _required_string(record, "strategy", json_file, record_no)
            if strategy != "hierarchical":
                continue
            stats["selected_records"] += 1
            child = _validate_raw_child(record, json_file, record_no)
            if child.chunk_id in seen_ids:
                first_file, first_record = seen_ids[child.chunk_id]
                raise ValueError(
                    f"duplicate chunk_id {child.chunk_id!r}; first={first_file} record #{first_record}; "
                    f"second={json_file} record #{record_no}"
                )
            seen_ids[child.chunk_id] = (str(json_file), record_no)
            records.append(child)

    by_source: dict[str, list[RawChild]] = {}
    for child in records:
        by_source.setdefault(child.source, []).append(child)
    ordered: list[RawChild] = []
    for source in sorted(by_source):
        ordered.extend(sorted(by_source[source], key=lambda item: (_sequence_sort_value(item.sequence), item.record_no)))
    stats["valid_children"] = len(ordered)
    stats["source_count"] = len(by_source)
    return ordered, stats


def resolve_children(children: list[RawChild]) -> list[ResolvedChild]:
    """Resolve each child to an article parent deterministically."""
    resolved: list[ResolvedChild] = []
    by_source: dict[str, list[RawChild]] = {}
    for child in children:
        by_source.setdefault(child.source, []).append(child)

    for source in sorted(by_source):
        current_chapter: str | None = None
        current_article: str | None = None
        current_title: str | None = None
        for child in by_source[source]:
            heading = _detect_headings(child.text)
            metadata_path = _clean_structure(child.structure)
            warnings: list[str] = []
            ambiguous = False
            method = "document_fallback"
            path = {key: None for key in STRUCTURE_KEYS}

            metadata_has_any = any(metadata_path.values())
            metadata_article = metadata_path.get("article")
            heading_article = heading.get("article")
            heading_chapter = heading.get("chapter")

            if metadata_has_any:
                method = "metadata"
                path.update(metadata_path)
                if heading_article and metadata_article and _label_key(heading_article) != _label_key(metadata_article):
                    ambiguous = True
                    warnings.append("metadata_heading_conflict")
                if heading_chapter and path.get("chapter") and _label_key(heading_chapter) != _label_key(path["chapter"]):
                    ambiguous = True
                    warnings.append("metadata_heading_conflict")
                if heading_article and not metadata_article:
                    path["article"] = heading_article
                    warnings.append("metadata_missing_article_heading_used")
                if heading_chapter and not path.get("chapter"):
                    path["chapter"] = heading_chapter
                    warnings.append("metadata_missing_chapter_heading_used")
            elif heading_article or heading_chapter:
                method = "heading_inferred"
                if heading_chapter:
                    path["chapter"] = heading_chapter
                if heading_article:
                    path["article"] = heading_article
                warnings.append("missing_structure_metadata")
                warnings.append("heading_detected_from_text")
            elif current_article:
                method = "carried_forward"
                path["chapter"] = current_chapter
                path["article"] = current_article
                warnings.append("missing_structure_metadata")
                warnings.append("carried_forward_article")
            else:
                method = "document_fallback"
                path["chapter"] = current_chapter
                path["article"] = None
                warnings.append("missing_structure_metadata")
                warnings.append("document_fallback")

            # Clause/point headings are local child annotations and must not create parents.
            if not path.get("clause") and heading.get("clause"):
                path["clause"] = heading["clause"]
            if not path.get("point") and heading.get("point"):
                path["point"] = heading["point"]

            article_candidates = _article_candidates(child.text)
            if heading_article is None and article_candidates:
                warnings.append("inline_article_reference_ignored")
            if heading_article and len(set(article_candidates)) > 1:
                ambiguous = True
                warnings.append("multiple_article_candidates")

            if path.get("chapter"):
                current_chapter = path["chapter"]
            if path.get("article"):
                current_article = path["article"]
                current_title = _heading_title(child.text, path["article"])

            article_key = _article_key(source, path.get("article"))
            parent_title = current_title if path.get("article") and current_title else _document_fallback_title(source)
            placeholder_parent_id = _parent_id(source=source, article_key=article_key, window_index=1)
            resolved.append(
                ResolvedChild(
                    child_id=child.chunk_id,
                    parent_id=placeholder_parent_id,
                    source=child.source,
                    page_start=child.page_start,
                    page_end=child.page_end,
                    text=child.text,
                    structural_path=path,
                    resolution_method=method,
                    ambiguous=ambiguous,
                    warnings=warnings,
                    article_key=article_key,
                    parent_title=parent_title,
                    sequence=child.sequence,
                )
            )
    return resolved


def build_parent_documents(children: list[ResolvedChild], *, config: HierarchyConfig) -> list[ParentDocument]:
    """Build article/document parent windows without cutting through child text."""
    grouped: dict[tuple[str, str], list[ResolvedChild]] = {}
    for child in children:
        grouped.setdefault((child.source, child.article_key), []).append(child)

    parents: list[ParentDocument] = []
    parent_id_by_child: dict[str, str] = {}
    for source, article_key in sorted(grouped):
        group = sorted(grouped[(source, article_key)], key=lambda item: (_sequence_sort_value(item.sequence), item.child_id))
        windows = _split_parent_windows(group, max_chars=config.parent_max_chars)
        for window_index, window_children in enumerate(windows, start=1):
            parent_id = _parent_id(source=source, article_key=article_key, window_index=window_index)
            text = "\n\n".join(child.text for child in window_children)
            warnings: list[str] = []
            if len(text) > config.parent_max_chars and len(window_children) == 1:
                warnings.append("oversized_single_child")
            ambiguous_count = sum(1 for child in window_children if child.ambiguous)
            if ambiguous_count:
                warnings.append("contains_ambiguous_children")
            for child in window_children:
                if child.child_id in parent_id_by_child:
                    raise ValueError(f"child assigned to multiple parent windows: {child.child_id}")
                parent_id_by_child[child.child_id] = parent_id
            parents.append(
                ParentDocument(
                    parent_id=parent_id,
                    source=source,
                    page_start=min(child.page_start for child in window_children),
                    page_end=max(child.page_end for child in window_children),
                    article_key=article_key,
                    window_index=window_index,
                    child_ids=[child.child_id for child in window_children],
                    text=text,
                    char_count=len(text),
                    ambiguous_child_count=ambiguous_count,
                    warnings=warnings,
                )
            )

    if len(parent_id_by_child) != len(children):
        missing = sorted({child.child_id for child in children} - set(parent_id_by_child))
        raise ValueError(f"children without parent window: {missing[:5]}")
    return parents


def children_with_parent_ids(children: list[ResolvedChild], parents: list[ParentDocument]) -> list[ResolvedChild]:
    """Return child records with final window parent IDs."""
    by_child: dict[str, str] = {}
    for parent in parents:
        for child_id in parent.child_ids:
            by_child[child_id] = parent.parent_id
    return [
        ResolvedChild(
            child_id=child.child_id,
            parent_id=by_child[child.child_id],
            source=child.source,
            page_start=child.page_start,
            page_end=child.page_end,
            text=child.text,
            structural_path=child.structural_path,
            resolution_method=child.resolution_method,
            ambiguous=child.ambiguous,
            warnings=child.warnings,
            article_key=child.article_key,
            parent_title=child.parent_title,
            sequence=child.sequence,
        )
        for child in children
    ]


def child_to_json(child: ResolvedChild) -> dict[str, Any]:
    return {
        "child_id": child.child_id,
        "parent_id": child.parent_id,
        "source": child.source,
        "page_start": child.page_start,
        "page_end": child.page_end,
        "text": child.text,
        "structural_path": dict(child.structural_path),
        "resolution_method": child.resolution_method,
        "ambiguous": child.ambiguous,
        "warnings": list(child.warnings),
    }


def parent_to_json(parent: ParentDocument) -> dict[str, Any]:
    return asdict(parent)


def build_manifest(
    *,
    input_path: Path | str,
    load_stats: dict[str, Any],
    children: list[ResolvedChild],
    parents: list[ParentDocument],
    config: HierarchyConfig,
) -> dict[str, Any]:
    final_children = children_with_parent_ids(children, parents)
    warning_counts = _warning_counts(final_children, parents)
    parent_sizes = sorted(parent.char_count for parent in parents)
    return {
        "schema_version": SCHEMA_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(Path(input_path).resolve()),
        "input_file_fingerprints": load_stats.get("input_files", []),
        "strategy": "hierarchical",
        "config_identity": _config_identity(config),
        "counts": {
            "files_read": load_stats.get("files_read", 0),
            "total_records": load_stats.get("total_records", 0),
            "children": len(final_children),
            "parents": len(parents),
            "sources": len({child.source for child in final_children}),
            "ambiguous_children": sum(1 for child in final_children if child.ambiguous),
            "document_fallback_children": sum(1 for child in final_children if child.resolution_method == "document_fallback"),
        },
        "warning_counts": warning_counts,
        "parent_size_distribution": _size_distribution(parent_sizes),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Buổi 09 hierarchy registry and query expansion tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    expand = subparsers.add_parser("expand-query", help="Generate controlled multi-query variants for one question")
    expand.add_argument("--question", required=True, help="Original Vietnamese legal question")
    expand.add_argument("--no-cache", action="store_true", help="Bypass in-process cache for this invocation")

    multi_child = subparsers.add_parser("multi-child", help="Fan out hybrid retrieval over expanded queries and fuse child hits")
    multi_child.add_argument("--question", required=True, help="Original Vietnamese legal question")
    multi_child.add_argument("--strategy", default="hierarchical", help="fixed-size, semantic hoặc hierarchical")
    multi_child.add_argument("--input", default=None, help="File/thư mục chunk JSON")
    multi_child.add_argument("--storage", default=None, help="Thư mục Chroma persistent của Buổi 09")

    parent_retrieve_parser = subparsers.add_parser("parent-retrieve", help="Retrieve child hits and return aggregated parent contexts")
    parent_retrieve_parser.add_argument("--mode", choices=sorted(PARENT_RETRIEVAL_MODES), default="multi_parent", help="single_parent hoặc multi_parent")
    parent_retrieve_parser.add_argument("--question", required=True, help="Original Vietnamese legal question")
    parent_retrieve_parser.add_argument("--strategy", default="hierarchical", help="fixed-size, semantic hoặc hierarchical")
    parent_retrieve_parser.add_argument("--input", default=None, help="File/thư mục chunk JSON")
    parent_retrieve_parser.add_argument("--storage", default=None, help="Thư mục Chroma persistent của Buổi 09")
    parent_retrieve_parser.add_argument("--hierarchy-storage", default=str(HIERARCHY_STORAGE_DIR), help="Thư mục hierarchy store của Buổi 09")

    query_parser = subparsers.add_parser("query", help="Run Buổi 09 answer pipeline for flat/parent modes")
    query_parser.add_argument("--mode", choices=sorted(ANSWER_MODES), default="multi_parent", help="single_flat, multi_flat, single_parent hoặc multi_parent")
    query_parser.add_argument("--question", required=True, help="Original Vietnamese legal question")
    query_parser.add_argument("--strategy", default="hierarchical", help="fixed-size, semantic hoặc hierarchical")
    query_parser.add_argument("--input", default=None, help="File/thư mục chunk JSON")
    query_parser.add_argument("--storage", default=None, help="Thư mục Chroma persistent của Buổi 09")
    query_parser.add_argument("--hierarchy-storage", default=str(HIERARCHY_STORAGE_DIR), help="Thư mục hierarchy store của Buổi 09")

    compare_parser = subparsers.add_parser("compare", help="Compare four Buổi 09 modes without answer generation")
    compare_parser.add_argument("--question", required=True, help="Original Vietnamese legal question")
    compare_parser.add_argument("--strategy", default="hierarchical", help="fixed-size, semantic hoặc hierarchical")
    compare_parser.add_argument("--input", default=None, help="File/thư mục chunk JSON")
    compare_parser.add_argument("--storage", default=None, help="Thư mục Chroma persistent của Buổi 09")
    compare_parser.add_argument("--hierarchy-storage", default=str(HIERARCHY_STORAGE_DIR), help="Thư mục hierarchy store của Buổi 09")

    audit = subparsers.add_parser("hierarchy-audit", help="Read chunks and print hierarchy stats without writing store")
    audit.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input chunk JSON file or directory")

    build = subparsers.add_parser("build-hierarchy", help="Build children/parents/manifest JSON atomically")
    build.add_argument("--input", default=str(DEFAULT_INPUT_PATH), help="Input chunk JSON file or directory")
    build.add_argument("--storage", default=str(HIERARCHY_STORAGE_DIR), help="Hierarchy storage directory")

    status = subparsers.add_parser("hierarchy-status", help="Read hierarchy store status without writing")
    status.add_argument("--storage", default=str(HIERARCHY_STORAGE_DIR), help="Hierarchy storage directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "expand-query":
            _print_expand_query(expand_query(args.question, use_cache=not args.no_cache))
            return 0
        if args.command == "multi-child":
            _print_multi_child(multi_child_retrieve(args.question, strategy=args.strategy, input_path=args.input, storage_path=args.storage))
            return 0
        if args.command == "parent-retrieve":
            _print_parent_retrieve(
                parent_retrieve(
                    args.question,
                    mode=args.mode,
                    storage_dir=args.hierarchy_storage,
                    strategy=args.strategy,
                    input_path=args.input,
                    storage_path=args.storage,
                )
            )
            return 0
        if args.command == "query":
            _print_answer_pipeline_result(
                answer_query_buoi09(
                    args.question,
                    mode=args.mode,
                    storage_dir=args.hierarchy_storage,
                    strategy=args.strategy,
                    input_path=args.input,
                    storage_path=args.storage,
                )
            )
            return 0
        if args.command == "compare":
            _print_compare_buoi09(
                compare_buoi09(
                    args.question,
                    storage_dir=args.hierarchy_storage,
                    strategy=args.strategy,
                    input_path=args.input,
                    storage_path=args.storage,
                )
            )
            return 0
        if args.command == "hierarchy-audit":
            result = hierarchy_audit(args.input)
            _print_audit(result)
            return 0
        if args.command == "build-hierarchy":
            result = build_hierarchy_store(args.input, storage_dir=args.storage)
            _print_build_result(result)
            return 0
        if args.command == "hierarchy-status":
            _print_status(hierarchy_status(storage_dir=args.storage))
            return 0
    except Exception as error:  # noqa: BLE001 - CLI diagnostics should stay concise.
        print(f"Lỗi: {type(error).__name__}: {error}")
        return 1
    parser.error(f"Command không hỗ trợ: {args.command}")
    return 2


def _summary(*, load_stats: dict[str, Any], children: list[ResolvedChild], parents: list[ParentDocument], config: HierarchyConfig) -> dict[str, Any]:
    final_children = children_with_parent_ids(children, parents)
    return {
        "schema_version": SCHEMA_VERSION,
        "counts": {
            "files_read": load_stats["files_read"],
            "total_records": load_stats["total_records"],
            "children": len(final_children),
            "parents": len(parents),
            "sources": len({child.source for child in final_children}),
            "ambiguous_children": sum(1 for child in final_children if child.ambiguous),
            "document_fallback_children": sum(1 for child in final_children if child.resolution_method == "document_fallback"),
        },
        "resolution_methods": _counter(child.resolution_method for child in final_children),
        "warning_counts": _warning_counts(final_children, parents),
        "warning_examples": _warning_examples(final_children, parents),
        "parent_size_distribution": _size_distribution(sorted(parent.char_count for parent in parents)),
        "config_identity": _config_identity(config),
    }


def _print_audit(result: dict[str, Any]) -> None:
    print("Hierarchy audit:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _print_build_result(result: dict[str, Any]) -> None:
    manifest = result["manifest"]
    print("Hierarchy build complete:")
    print(json.dumps({"counts": manifest["counts"], "warning_counts": manifest["warning_counts"], "parent_size_distribution": manifest["parent_size_distribution"]}, ensure_ascii=False, indent=2))


def _print_status(status: dict[str, Any]) -> None:
    print("Hierarchy status:")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def _print_expand_query(result: dict[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _print_multi_child(result: dict[str, Any]) -> None:
    print(f"status={result['status']} mode={result['mode']} strategy={result.get('strategy', 'hierarchical')}")
    print("queries:")
    for query in result.get("queries", []):
        print(f"- {query['query_id']} origin={query['origin']} weight={query['weight']} focus={query['focus']} text={query['text']}")
    if result.get("warnings"):
        print("warnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")
    print("child_id | per_query_ranks | support | mq_rrf_score | source | page")
    for child in result.get("children", []):
        print(
            f"{child['child_id']} | {child['per_query_ranks']} | {child['support_query_count']} | "
            f"{child['multi_query_rrf_score']:.8f} | {child['source']} | {_page_text(child['page_start'], child['page_end'])}"
        )
    print("trace:")
    print(json.dumps(result.get("trace", {}), ensure_ascii=False, indent=2))


def _print_parent_retrieve(result: dict[str, Any]) -> None:
    print(f"status={result['status']} mode={result['mode']}")
    if result.get("warnings"):
        print("warnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")
    for parent in result.get("parents", []):
        print(
            f"Parent #{parent['parent_rank']} score={parent['parent_rrf_score']:.8f} "
            f"parent_id={parent['parent_id']} source={parent['source']} page={_page_text(parent['page_start'], parent['page_end'])}"
        )
        for child_id in parent["supporting_child_ids"]:
            mapping = next((row for row in result["trace"].get("child_to_parent_mapping", []) if row["child_id"] == child_id), {})
            print(f"└── child {child_id}")
            print(f"    └── queries={mapping.get('support_query_ids', [])} rank={mapping.get('multi_query_rank')}")
    print("trace:")
    print(json.dumps(result.get("trace", {}), ensure_ascii=False, indent=2))


def _print_answer_pipeline_result(result: dict[str, Any]) -> None:
    print(f"status={result['status']} mode={result['mode']}")
    if result.get("error"):
        print(f"error={result['error']}")
    if result.get("warnings"):
        print("warnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")
    print("answer:")
    print(result.get("answer", ""))
    print("citations:")
    print(json.dumps(result.get("citations", []), ensure_ascii=False, indent=2))
    print("trace:")
    print(json.dumps(result.get("trace", {}), ensure_ascii=False, indent=2))


def _print_compare_buoi09(result: dict[str, Any]) -> None:
    print(f"Compare question={result['question']}")
    for mode, payload in result["modes"].items():
        if mode.endswith("parent"):
            retrieval = payload.get("retrieval", {})
            rerank = payload.get("rerank", {})
            print(f"- {mode}: retrieval={retrieval.get('status')} parents={len(retrieval.get('parents', []))} rerank={rerank.get('status')} reranked={rerank.get('reranked_count', 0)}")
        else:
            print(f"- {mode}: status={payload.get('status')} children={len(payload.get('child_hits', []))} accepted={len(payload.get('accepted_evidence', []))}")
    print("answer_generation_called=False")


def clear_query_expansion_cache() -> None:
    """Clear the in-process query expansion cache for deterministic tests."""
    _QUERY_EXPANSION_CACHE.clear()


def _advanced_config_for_rerank(config: HierarchyConfig) -> dict[str, Any]:
    try:
        import advanced_rag

        return advanced_rag.load_config(advanced_rag._default_env_path())
    except Exception:
        return {
            "reranker_model": config.reranker_model,
            "rerank_min_score": config.reranker_min_score,
            "rerank_device": config.reranker_device,
            "final_top_k": config.final_top_k,
            "rerank_candidates": config.rerank_candidates,
            "reranker_max_length": 512,
            "rerank_batch_size": 4,
        }


def _default_parent_rerank_scorer(original_question: str, parents: list[dict[str, Any]], advanced_config: dict[str, Any]) -> list[float]:
    import advanced_rag

    return advanced_rag._score_with_default_reranker(original_question, parents, advanced_config)


def _default_child_rerank_scorer(original_question: str, children: list[dict[str, Any]], advanced_config: dict[str, Any]) -> list[float]:
    import advanced_rag

    return advanced_rag._score_with_default_reranker(original_question, children, advanced_config)


def _flat_mode_retrieve_and_rerank(
    original_question: str,
    *,
    mode: str,
    config: HierarchyConfig,
    query_generator_fn: QueryGeneratorFn | None,
    hybrid_retriever_fn: HybridRetrieverFn | None,
    child_rerank_scorer: ChildRerankScorer | None,
    strategy: str,
    input_path: Path | str | None,
    storage_path: Path | str | None,
) -> dict[str, Any]:
    if mode == "single_flat":
        child_result = _single_child_retrieve(
            original_question,
            config=config,
            hybrid_retriever_fn=hybrid_retriever_fn,
            strategy=strategy,
            input_path=input_path,
            storage_path=storage_path,
        )
        query_set = child_result.get("queries", [])
        api_counts = {"semantic_embedding_calls": child_result.get("trace", {}).get("semantic_embedding_call_count", 0), "generation_calls": 0, "query_expansion_generation_calls": 0, "answer_generation_calls": 0, "rerank_calls": 0}
    else:
        child_result = multi_child_retrieve(
            original_question,
            config=config,
            query_generator_fn=query_generator_fn,
            hybrid_retriever_fn=hybrid_retriever_fn,
            strategy=strategy,
            input_path=input_path,
            storage_path=storage_path,
        )
        query_set = child_result.get("queries", [])
        api_counts = _counts_from_parent_retrieval(child_result, query_generator_fn=query_generator_fn)
    if child_result.get("status") == QUERY_GENERATION_UNAVAILABLE:
        return {"status": QUERY_GENERATION_UNAVAILABLE, "query_set": query_set, "child_hits": [], "accepted_evidence": [], "warnings": child_result.get("warnings", []), "trace": child_result.get("trace", {}), "api_counts": api_counts}

    children = child_result.get("children", [])
    reranked = _rerank_child_hits(original_question, children, config=config, scorer=child_rerank_scorer)
    api_counts["rerank_calls"] = 1 if children else 0
    if reranked["status"] == "reranker_unavailable":
        return {"status": "reranker_unavailable", "query_set": query_set, "child_hits": children, "accepted_evidence": [], "warnings": [reranked.get("error", "reranker_unavailable")], "error": reranked.get("error"), "trace": child_result.get("trace", {}), "api_counts": api_counts}
    accepted = _accepted_flat_evidence(reranked["children"], config=config)
    return {"status": child_result.get("status", QUERY_GENERATION_READY), "query_set": query_set, "child_hits": reranked["children"], "accepted_evidence": accepted, "warnings": child_result.get("warnings", []), "trace": child_result.get("trace", {}), "api_counts": api_counts}


def _rerank_child_hits(original_question: str, children: list[dict[str, Any]], *, config: HierarchyConfig, scorer: ChildRerankScorer | None) -> dict[str, Any]:
    candidates = sorted(children, key=lambda item: item.get("multi_query_rank", item.get("fused_rank", 10**9)))[: config.rerank_candidates]
    if not candidates:
        return {"status": QUERY_GENERATION_READY, "children": [], "reranked_count": 0}
    scorer = scorer or _default_child_rerank_scorer
    try:
        scores = scorer(original_question, candidates, _advanced_config_for_rerank(config))
    except Exception as error:  # noqa: BLE001
        return {"status": "reranker_unavailable", "children": [], "error": _safe_error_message(error)}
    if len(scores) != len(candidates):
        return {"status": "reranker_unavailable", "children": [], "error": "child_rerank_score_count_mismatch"}
    ranked = []
    for candidate, raw_score in zip(candidates, scores):
        item = dict(candidate)
        item["child_rerank_raw_score"] = _finite_float(raw_score, name="child_rerank_raw_score")
        item["child_rerank_score"] = _sigmoid(item["child_rerank_raw_score"])
        ranked.append(item)
    ranked.sort(key=lambda item: (-item["child_rerank_score"], item.get("multi_query_rank", item.get("fused_rank", 10**9)), item["child_id"]))
    for rank, item in enumerate(ranked, start=1):
        item["child_rerank_rank"] = rank
    return {"status": QUERY_GENERATION_READY, "children": ranked[: config.final_top_k], "reranked_count": len(ranked)}


def _accepted_parent_evidence(parents: list[dict[str, Any]], *, config: HierarchyConfig) -> list[dict[str, Any]]:
    accepted = []
    for index, parent in enumerate(parents, start=1):
        if float(parent.get("parent_rerank_score", -1.0)) >= config.reranker_min_score:
            item = dict(parent)
            item["evidence_id"] = f"P{len(accepted) + 1}"
            accepted.append(item)
    return accepted


def _accepted_flat_evidence(children: list[dict[str, Any]], *, config: HierarchyConfig) -> list[dict[str, Any]]:
    accepted = []
    for child in children:
        if float(child.get("child_rerank_score", -1.0)) >= config.reranker_min_score:
            item = dict(child)
            item["evidence_id"] = f"E{len(accepted) + 1}"
            accepted.append(item)
    return accepted


def _build_parent_answer_prompt(original_question: str, accepted: list[dict[str, Any]]) -> str:
    blocks = []
    for item in accepted:
        blocks.append(f"<<<CONTEXT {item['evidence_id']}>>>\n{item['text']}\n<<<END CONTEXT {item['evidence_id']}>>>")
    return (
        "Bạn là trợ lý RAG tiếng Việt. Chỉ dùng evidence được cung cấp, không tư vấn pháp lý ngoài context.\n"
        "Mỗi nhận định phải có citation [P1], [P2] tương ứng evidence. Không tự tạo nguồn/trang/Điều/Khoản/parent_id/child_id.\n"
        "Nếu evidence ambiguous hoặc mâu thuẫn, nói rõ giới hạn. Query variants không phải sự thật và không nằm trong prompt này.\n\n"
        f"Câu hỏi gốc:\n{original_question}\n\nEvidence accepted:\n" + "\n".join(blocks)
    )


def _build_flat_answer_prompt(original_question: str, accepted: list[dict[str, Any]]) -> str:
    blocks = []
    for item in accepted:
        blocks.append(f"<<<CONTEXT {item['evidence_id']}>>>\n{item['text']}\n<<<END CONTEXT {item['evidence_id']}>>>")
    return (
        "Bạn là trợ lý RAG tiếng Việt. Chỉ dùng evidence được cung cấp. Mỗi nhận định phải có citation [E1], [E2].\n"
        "Không tự tạo nguồn/trang/chunk_id.\n\n"
        f"Câu hỏi gốc:\n{original_question}\n\nEvidence accepted:\n" + "\n".join(blocks)
    )


def _call_answer_generator(prompt: str, *, config: HierarchyConfig, answer_generator: AnswerGenerator | None) -> str:
    if answer_generator is not None:
        return answer_generator(prompt, {"generation_model": config.gemini_generation_model})
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Thiếu GEMINI_API_KEY để sinh answer")
    try:
        from google import genai
    except ImportError as error:
        raise RuntimeError("Thiếu package google-genai") from error
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=config.gemini_generation_model, contents=prompt)
    return getattr(response, "text", "")


def _validate_parent_citations(answer: str, accepted: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], list[str]]:
    allowed = {item["evidence_id"]: item for item in accepted}
    labels = re.findall(r"\[(P\d+)\]", answer or "")
    warnings: list[str] = []
    if not labels:
        warnings.append("missing_parent_citation")
    invalid = sorted(set(label for label in labels if label not in allowed))
    if invalid:
        warnings.append("invalid_parent_citation_label: " + ",".join(invalid))
    citations = [_parent_citation(allowed[label]) for label in dict.fromkeys(labels) if label in allowed]
    return (answer or "").strip(), citations, warnings


def _validate_flat_citations(answer: str, accepted: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], list[str]]:
    allowed = {item["evidence_id"]: item for item in accepted}
    labels = re.findall(r"\[(E\d+)\]", answer or "")
    warnings: list[str] = []
    if not labels:
        warnings.append("missing_flat_citation")
    invalid = sorted(set(label for label in labels if label not in allowed))
    if invalid:
        warnings.append("invalid_flat_citation_label: " + ",".join(invalid))
    citations = [_flat_citation(allowed[label]) for label in dict.fromkeys(labels) if label in allowed]
    return (answer or "").strip(), citations, warnings


def _parent_citation(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item["evidence_id"],
        "parent_id": item["parent_id"],
        "anchor_child_id": item["anchor_child_id"],
        "supporting_child_ids": list(item["supporting_child_ids"]),
        "source": item["source"],
        "page_start": item["page_start"],
        "page_end": item["page_end"],
        "structural_path": item.get("structural_path", {}),
        "parent_rerank_score": item.get("parent_rerank_score"),
        "ambiguous": item.get("ambiguous", False),
        "warnings": list(item.get("warnings", [])),
    }


def _flat_citation(item: dict[str, Any]) -> dict[str, Any]:
    return {"evidence_id": item["evidence_id"], "child_id": item["child_id"], "source": item["source"], "page_start": item["page_start"], "page_end": item["page_end"], "child_rerank_score": item.get("child_rerank_score")}


def _final_answer_result(
    *,
    status: str,
    mode: str,
    original_question: str,
    query_set: list[dict[str, Any]],
    child_hits: list[dict[str, Any]],
    parent_candidates: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    answer: str,
    citations: list[dict[str, Any]],
    warnings: list[str],
    api_counts: dict[str, int],
    started: float,
    config: HierarchyConfig,
    retrieval_trace: dict[str, Any],
    rerank_trace: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "mode": mode,
        "original_question": original_question,
        "query_set": query_set,
        "child_hits": child_hits,
        "parent_candidates": parent_candidates,
        "accepted_evidence": accepted,
        "answer": answer,
        "citations": citations,
        "warnings": warnings,
        "error": error,
        "trace": {
            "stage_latencies_ms": {"total": _elapsed_ms(started), "retrieval_total": retrieval_trace.get("latency_ms", {}).get("total", 0.0), "rerank": (rerank_trace or {}).get("latency_ms", 0.0)},
            "api_call_counts": api_counts,
            "model_identity": {"generation_model": config.gemini_generation_model, "reranker_model": config.reranker_model, "embedding_model": config.gemini_embedding_model, "embedding_dim": config.gemini_embedding_dim},
            "config_identity": _config_identity(config),
            "retrieval_trace": retrieval_trace,
            "rerank_trace": rerank_trace or {},
        },
    }


def _counts_from_parent_retrieval(retrieval: dict[str, Any], *, query_generator_fn: QueryGeneratorFn | None) -> dict[str, int]:
    trace = retrieval.get("trace", {})
    expansion_calls = int(trace.get("gemini_expansion_call_count", 0))
    if query_generator_fn is not None:
        expansion_calls = 0
    return {"generation_calls": expansion_calls, "query_expansion_generation_calls": expansion_calls, "answer_generation_calls": 0, "semantic_embedding_calls": int(trace.get("semantic_embedding_call_count", 0)), "rerank_calls": 0}


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} phải là số")
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} phải hữu hạn")
    return value


def _sigmoid(value: float) -> float:
    value = _finite_float(value, name="sigmoid_input")
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _single_child_retrieve(
    question: str,
    *,
    config: HierarchyConfig,
    hybrid_retriever_fn: HybridRetrieverFn | None,
    strategy: str,
    input_path: Path | str | None,
    storage_path: Path | str | None,
) -> dict[str, Any]:
    retriever = hybrid_retriever_fn or _default_hybrid_retriever
    started = time.perf_counter()
    retrieval = retriever(question, strategy=strategy, input_path=input_path, storage_path=storage_path)
    candidates = _per_query_candidates(retrieval, limit=config.per_query_candidates, query_id="Q0")
    children = []
    for rank, candidate in enumerate(candidates, start=1):
        child = _multi_child_base(candidate)
        child["support_query_ids"] = ["Q0"]
        child["per_query_ranks"] = {"Q0": _candidate_fused_rank(candidate)}
        child["best_query_rank"] = _candidate_fused_rank(candidate)
        child["per_query_trace"] = {"Q0": _candidate_per_query_trace(candidate)}
        child["support_query_count"] = 1
        child["multi_query_rrf_score"] = config.multi_query_original_weight / (config.multi_query_rrf_k + _candidate_fused_rank(candidate))
        child["multi_query_rank"] = rank
        children.append(child)
    return {
        "status": QUERY_GENERATION_READY,
        "mode": "single_child",
        "question": question,
        "queries": [{"query_id": "Q0", "text": question, "origin": "original", "focus": "original_intent", "weight": config.multi_query_original_weight}],
        "children": children,
        "warnings": [],
        "trace": {
            "query_count": {"requested": 1, "valid": 1, "executed": 1, "failed": 0},
            "query_retrieval_latency_ms": {"Q0": _elapsed_ms(started)},
            "result_count_by_query": {"Q0": len(children)},
            "union_child_count": len(children),
            "semantic_embedding_call_count": int(retrieval.get("trace", {}).get("semantic_embedding_call_count", 1)),
        },
    }


def _default_hybrid_retriever(question: str, *, strategy: str, input_path: Path | str | None, storage_path: Path | str | None) -> dict[str, Any]:
    import advanced_rag

    return advanced_rag.hybrid_retrieve(
        question,
        strategy=strategy,
        input_path=input_path,
        storage_path=storage_path,
    )


def _hierarchy_store_readiness_warnings(manifest: dict[str, Any], *, config: HierarchyConfig | None, input_path: Path | str | None) -> list[str]:
    warnings: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        warnings.append("stale_hierarchy_schema_version")
    fingerprints = manifest.get("input_file_fingerprints")
    if not fingerprints:
        warnings.append("missing_hierarchy_input_fingerprint")
    if input_path is not None and fingerprints:
        try:
            expected = [_fingerprint_file(path) for path in _json_files(input_path)]
            expected_by_path = {item["path"]: item["sha256"] for item in expected}
            actual_by_path = {item.get("path"): item.get("sha256") for item in fingerprints if isinstance(item, dict)}
            if expected_by_path != actual_by_path:
                warnings.append("stale_hierarchy_input_fingerprint")
        except Exception as error:  # noqa: BLE001
            warnings.append(f"hierarchy_input_fingerprint_check_failed: {_safe_error_message(error)}")
    if config is not None:
        expected = _config_identity(config)
        actual = manifest.get("config_identity", {})
        for key in ("parent_max_chars", "parent_rrf_k", "parent_score_child_limit", "parent_candidates", "final_parent_top_k", "total_context_max_chars"):
            if actual.get(key) != expected.get(key):
                warnings.append(f"stale_hierarchy_config: {key}")
    return warnings


def _structural_path_from_parent(parent: dict[str, Any]) -> dict[str, str | None]:
    article_key = str(parent.get("article_key", ""))
    article = article_key.split("::", 1)[1] if article_key.startswith("article::") else None
    return {"chapter": None, "article": article, "clause": None, "point": None}


def _query_id_sort_key(query_id: str) -> tuple[int, str]:
    match = re.search(r"\d+", query_id)
    return (int(match.group(0)) if match else 10**9, query_id)


def _parent_empty_trace(*, total_started: float) -> dict[str, Any]:
    return {
        "input_child_hit_count": 0,
        "unique_parent_count": 0,
        "child_count_by_parent": {},
        "child_to_parent_mapping": [],
        "parent_score_components": {},
        "parents_dropped_by_candidate_limit": [],
        "parents_dropped_by_context_budget": [],
        "child_chars": 0,
        "expanded_parent_chars": 0,
        "context_expansion_factor": 0.0,
        "ambiguous_parent_count": 0,
        "warning_count": 0,
        "latency_ms": {"mapping_aggregation": 0.0, "context_budget": 0.0, "total": _elapsed_ms(total_started)},
    }


def _parent_trace(
    *,
    children: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    aggregation_trace: dict[str, Any],
    budget_trace: dict[str, Any],
    child_latency_ms: float,
    mapping_aggregation_latency_ms: float,
    budget_latency_ms: float,
    total_latency_ms: float,
) -> dict[str, Any]:
    child_chars = sum(len(child.get("text", "")) for child in children)
    parent_chars = sum(len(parent.get("text", "")) for parent in selected)
    warning_count = sum(len(parent.get("warnings", [])) for parent in selected)
    return {
        "input_child_hit_count": len(children),
        "unique_parent_count": aggregation_trace.get("input_parent_count", len(candidates)),
        "selected_parent_count": len(selected),
        "child_count_by_parent": aggregation_trace.get("child_count_by_parent", {}),
        "child_to_parent_mapping": aggregation_trace.get("child_to_parent_mapping", []),
        "parent_score_components": aggregation_trace.get("parent_score_components", {}),
        "parents_dropped_by_candidate_limit": aggregation_trace.get("parents_dropped_by_candidate_limit", []),
        "parents_dropped_by_context_budget": budget_trace.get("parents_dropped_by_context_budget", []),
        "child_chars": child_chars,
        "expanded_parent_chars": parent_chars,
        "context_expansion_factor": round(parent_chars / child_chars, 6) if child_chars else 0.0,
        "ambiguous_parent_count": sum(1 for parent in selected if parent.get("ambiguous")),
        "warning_count": warning_count,
        "latency_ms": {
            "child_retrieval": child_latency_ms,
            "mapping_aggregation": mapping_aggregation_latency_ms,
            "context_budget": budget_latency_ms,
            "total": total_latency_ms,
        },
    }


def _weighted_query_list(queries: list[dict[str, Any]], *, config: HierarchyConfig) -> list[dict[str, Any]]:
    weighted = []
    for item in queries:
        query = dict(item)
        query["weight"] = config.multi_query_original_weight if query["query_id"] == "Q0" else config.multi_query_variant_weight
        weighted.append(query)
    return weighted


def _query_weight(query: dict[str, Any]) -> float:
    weight = query.get("weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError(f"query {query.get('query_id')} thiếu weight hợp lệ")
    weight = float(weight)
    if math.isnan(weight) or math.isinf(weight) or weight <= 0:
        raise ValueError(f"query {query.get('query_id')} weight phải hữu hạn > 0")
    return weight


def _per_query_candidates(retrieval: dict[str, Any], *, limit: int, query_id: str) -> list[dict[str, Any]]:
    candidates = retrieval.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"{query_id}: hybrid retriever phải trả field candidates là list")
    ranked = sorted(candidates, key=lambda item: (_candidate_fused_rank(item), _candidate_child_id(item)))
    return ranked[:limit]


def _candidate_child_id(candidate: dict[str, Any]) -> str:
    value = candidate.get("child_id", candidate.get("chunk_id"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("candidate child_id/chunk_id phải là string không rỗng")
    return value.strip()


def _candidate_fused_rank(candidate: dict[str, Any]) -> int:
    value = candidate.get("fused_rank", candidate.get("inner_rrf_rank"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"candidate {_candidate_child_id(candidate)} thiếu fused_rank hợp lệ")
    return value


def _multi_child_base(candidate: dict[str, Any]) -> dict[str, Any]:
    child_id = _candidate_child_id(candidate)
    return {
        "child_id": child_id,
        "text": _required_text_field(candidate, "text", source=child_id),
        "source": _required_text_field(candidate, "source", source=child_id),
        "page_start": _required_int_field(candidate, "page_start", source=child_id),
        "page_end": _required_int_field(candidate, "page_end", source=child_id),
        "multi_query_rrf_score": 0.0,
        "multi_query_rank": None,
        "support_query_count": 0,
        "support_query_ids": [],
        "per_query_ranks": {},
        "best_query_rank": None,
        "per_query_trace": {},
    }


def _ensure_multi_child_metadata_consistent(fused: dict[str, Any], candidate: dict[str, Any], *, query_id: str) -> None:
    for key in ("text", "source", "page_start", "page_end"):
        if fused[key] != candidate.get(key):
            raise ValueError(f"Metadata mismatch cho child_id={fused['child_id']!r} field={key} tại {query_id}")


def _candidate_per_query_trace(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "bm25_rank": candidate.get("bm25_rank"),
        "semantic_rank": candidate.get("semantic_rank"),
        "inner_rrf_rank": _candidate_fused_rank(candidate),
        "fused_rank": _candidate_fused_rank(candidate),
        "matched_by": list(candidate.get("matched_by", [])),
    }


def _required_text_field(candidate: dict[str, Any], key: str, *, source: str) -> str:
    value = candidate.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}.{key} phải là string không rỗng")
    return value.strip()


def _required_int_field(candidate: dict[str, Any], key: str, *, source: str) -> int:
    value = candidate.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source}.{key} phải là integer")
    return value


def _multi_child_status(*, queries: list[dict[str, Any]], query_errors: dict[str, str]) -> str:
    if not query_errors:
        return QUERY_GENERATION_READY
    generated_query_ids = {query["query_id"] for query in queries if query["query_id"] != "Q0"}
    failed_generated = generated_query_ids & set(query_errors)
    if generated_query_ids and failed_generated == generated_query_ids:
        return "multi_query_partial"
    return "partial"


def _multi_child_warnings(*, status: str, query_errors: dict[str, str]) -> list[str]:
    warnings = []
    if status in {"partial", "multi_query_partial"}:
        warnings.append("generated_query_retrieval_failed")
        warnings.extend(f"{query_id}: {error}" for query_id, error in sorted(query_errors.items()))
    return warnings


def _multi_child_empty_trace(*, expansion: dict[str, Any], total_started: float) -> dict[str, Any]:
    return {
        "query_count": {"requested": 0, "valid": 0, "executed": 0, "failed": 0},
        "query_generation_latency_ms": expansion.get("generation_latency_ms", 0.0),
        "query_retrieval_latency_ms": {},
        "result_count_by_query": {},
        "union_child_count": 0,
        "overlap_distribution": {},
        "fusion_latency_ms": 0.0,
        "total_latency_ms": _elapsed_ms(total_started),
        "gemini_expansion_call_count": 0,
        "semantic_embedding_call_count": 0,
    }


def _multi_child_trace(
    *,
    expansion: dict[str, Any],
    queries: list[dict[str, Any]],
    per_query_results: dict[str, dict[str, Any]],
    query_errors: dict[str, str],
    children: list[dict[str, Any]],
    fusion_latency_ms: float,
    total_latency_ms: float,
    retrieval_latency_ms: float,
    semantic_embedding_call_count: int,
    gemini_expansion_call_count: int,
) -> dict[str, Any]:
    return {
        "query_count": {
            "requested": len(queries),
            "valid": len(queries),
            "executed": sum(1 for query_id, result in per_query_results.items() if query_id not in query_errors and result.get("hybrid_trace") is not None),
            "failed": len(query_errors),
        },
        "query_generation_latency_ms": expansion.get("generation_latency_ms", 0.0),
        "query_retrieval_latency_ms": {query_id: result.get("latency_ms", 0.0) for query_id, result in sorted(per_query_results.items())},
        "result_count_by_query": {query_id: result.get("result_count", 0) for query_id, result in sorted(per_query_results.items())},
        "union_child_count": len(children),
        "overlap_distribution": _overlap_distribution(children),
        "fusion_latency_ms": fusion_latency_ms,
        "retrieval_latency_ms": retrieval_latency_ms,
        "total_latency_ms": total_latency_ms,
        "gemini_expansion_call_count": gemini_expansion_call_count,
        "semantic_embedding_call_count": semantic_embedding_call_count,
    }


def _overlap_distribution(children: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for child in children:
        key = str(child["support_query_count"])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _call_query_generator(original_question: str, *, config: HierarchyConfig, query_generator_fn: QueryGeneratorFn | None) -> Any:
    if query_generator_fn is not None:
        return query_generator_fn(original_question, config)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Thiếu GEMINI_API_KEY; expand-query chỉ gọi Gemini khi người dùng chủ động chạy command này")
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise RuntimeError("Thiếu package google-genai; hãy cài requirements.txt của Buổi 09") from error

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=config.gemini_generation_model,
        contents=_query_expansion_prompt(original_question, config=config),
        config=types.GenerateContentConfig(
            temperature=config.multi_query_temperature,
            max_output_tokens=512,
            response_mime_type="application/json",
            response_schema=_query_expansion_response_schema(),
        ),
    )
    return getattr(response, "text", "")


def _query_expansion_prompt(original_question: str, *, config: HierarchyConfig) -> str:
    return (
        "Bạn là bộ tạo query expansion cho retrieval tiếng Việt pháp lý. "
        "Chỉ tạo các cách tra cứu, tuyệt đối không trả lời câu hỏi.\n\n"
        "Nhiệm vụ:\n"
        f"- Tạo 1 đến {config.multi_query_count} query generated cho câu hỏi gốc.\n"
        "- Bao phủ: thuật ngữ pháp lý chính xác, cách diễn đạt tương đương, và một khía cạnh còn thiếu nếu câu hỏi có nhiều ý.\n"
        "- Không thêm sự kiện, kết luận pháp lý, nguồn ngoài, citation, answer hoặc metadata retrieval.\n"
        "- Nếu câu hỏi có Điều/Khoản/Điểm/số hiệu/năm, ít nhất một query phải giữ nguyên reference đó.\n"
        "- Không phát minh số Điều/Khoản/Điểm không có trong câu hỏi.\n"
        "- Mỗi query phải ngắn gọn, dùng để tìm kiếm tài liệu, không quá giới hạn ký tự.\n\n"
        "Trả về JSON đúng schema, chỉ gồm generated variants, dạng:\n"
        '{"queries":[{"text":"...","focus":"exact_legal_terms|paraphrase|missing_aspect"}]}\n\n'
        f"Câu hỏi gốc: {original_question}"
    )


def _query_expansion_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "focus": {"type": "string", "enum": sorted(QUERY_FOCUS_VALUES)},
                    },
                    "required": ["text", "focus"],
                },
            }
        },
        "required": ["queries"],
    }


def _validate_query_set(*, original_question: str, raw_generated: Any, config: HierarchyConfig, latency_ms: float) -> dict[str, Any]:
    payload = _parse_generated_payload(raw_generated)
    raw_queries = payload["queries"]
    warnings: list[str] = []
    dropped_duplicate_count = 0
    dropped_invalid_count = 0
    if len(raw_queries) > config.multi_query_count:
        warnings.append("generated_query_count_truncated")
        raw_queries = raw_queries[: config.multi_query_count]

    original_key = _query_dedupe_key(original_question)
    seen = {original_key}
    generated: list[dict[str, str]] = []
    for index, item in enumerate(raw_queries, start=1):
        try:
            text = _normalize_generated_query(item["text"], config=config)
        except ValueError as error:
            warnings.append(f"dropped_invalid_query_{index}: {error}")
            dropped_invalid_count += 1
            continue
        focus = item["focus"]
        invalid_reason = _generated_query_invalid_reason(original_question, text)
        if invalid_reason:
            warnings.append(f"dropped_invalid_{invalid_reason}")
            dropped_invalid_count += 1
            continue
        key = _query_dedupe_key(text)
        if key in seen:
            dropped_duplicate_count += 1
            warnings.append("dropped_duplicate_query")
            continue
        seen.add(key)
        generated.append({"text": text, "focus": focus})

    if not generated:
        detail = "; ".join(warnings[:3])
        message = "Không còn generated query hợp lệ sau validation"
        raise ValueError(f"{message}: {detail}" if detail else message)
    if _legal_reference_keys(original_question) and not _any_generated_preserves_legal_reference(original_question, generated):
        raise ValueError("Không có generated query nào giữ reference pháp lý quan trọng từ câu hỏi gốc")

    queries = [{"query_id": "Q0", "text": original_question, "origin": "original", "focus": "original_intent"}]
    for index, item in enumerate(generated, start=1):
        queries.append({"query_id": f"Q{index}", "text": item["text"], "origin": "generated", "focus": item["focus"]})
    return {
        "original_question": original_question,
        "queries": queries,
        "model": config.gemini_generation_model,
        "generation_latency_ms": latency_ms,
        "status": QUERY_GENERATION_READY,
        "cache_hit": False,
        "dropped_duplicate_count": dropped_duplicate_count,
        "dropped_invalid_count": dropped_invalid_count,
        "warnings": sorted(set(warnings)),
    }


def _parse_generated_payload(raw_generated: Any) -> dict[str, Any]:
    if isinstance(raw_generated, str):
        try:
            payload = json.loads(raw_generated)
        except json.JSONDecodeError as error:
            raise ValueError(f"Generator không trả JSON hợp lệ: dòng {error.lineno}, cột {error.colno}: {error.msg}") from error
    elif isinstance(raw_generated, dict):
        payload = raw_generated
    else:
        raise ValueError("Generator phải trả dict hoặc JSON string")
    if set(payload) != {"queries"}:
        raise ValueError("Generator JSON chỉ được chứa field top-level 'queries'")
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("Generator JSON field 'queries' phải là list không rỗng")
    for index, item in enumerate(queries, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"queries[{index}] phải là object")
        if set(item) != {"text", "focus"}:
            raise ValueError(f"queries[{index}] chỉ được chứa field 'text' và 'focus'")
        if not isinstance(item["text"], str):
            raise ValueError(f"queries[{index}].text phải là string")
        if item["focus"] not in QUERY_FOCUS_VALUES:
            raise ValueError(f"queries[{index}].focus không hợp lệ")
    return {"queries": queries}


def _normalize_question(question: str) -> str:
    if not isinstance(question, str):
        raise ValueError("question phải là string")
    text = unicodedata.normalize("NFC", question).strip()
    if not text:
        raise ValueError("question không được rỗng")
    if len(text) > MAX_QUESTION_CHARS:
        raise ValueError(f"question không được vượt {MAX_QUESTION_CHARS} ký tự")
    return text


def _normalize_generated_query(text: str, *, config: HierarchyConfig) -> str:
    normalized = unicodedata.normalize("NFC", text).strip()
    if not normalized:
        raise ValueError("generated query rỗng")
    if len(normalized) > config.multi_query_max_chars:
        raise ValueError(f"generated query vượt {config.multi_query_max_chars} ký tự")
    return normalized


def _generated_query_invalid_reason(original_question: str, generated_text: str) -> str | None:
    try:
        invented = _invented_legal_refs(original_question, generated_text)
    except ValueError:
        return "legal_reference"
    return "legal_reference" if invented else None


def _invented_legal_refs(original_question: str, generated_text: str) -> list[str]:
    invented: list[str] = []
    for label in ("Điều", "Khoản", "Điểm"):
        original_values = _numbered_legal_values(original_question, label)
        generated_values = _numbered_legal_values(generated_text, label)
        for value in sorted(generated_values - original_values):
            invented.append(f"{label} {value}")
    original_years = set(re.findall(r"\b(?:19|20)\d{2}\b", original_question))
    generated_years = set(re.findall(r"\b(?:19|20)\d{2}\b", generated_text))
    for value in sorted(generated_years - original_years):
        invented.append(value)
    return invented


def _numbered_legal_values(text: str, label: str) -> set[str]:
    pattern = re.compile(rf"\b{label}\s+([0-9a-zA-ZđĐ]+)\b", re.IGNORECASE)
    return {_label_key(match.group(1)) for match in pattern.finditer(text)}


def _legal_reference_keys(text: str) -> set[str]:
    return {_query_dedupe_key(match.group(0)) for match in LEGAL_REFERENCE_RE.finditer(text)}


def _any_generated_preserves_legal_reference(original_question: str, generated: list[dict[str, str]]) -> bool:
    original_refs = _legal_reference_keys(original_question)
    if not original_refs:
        return True
    for item in generated:
        if original_refs & _legal_reference_keys(item["text"]):
            return True
    return False


def _query_dedupe_key(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold().strip()
    normalized = PUNCT_NORMALIZE_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def _query_cache_key(original_question: str, config: HierarchyConfig) -> str:
    identity = {
        "question": original_question,
        "model": config.gemini_generation_model,
        "multi_query_count": config.multi_query_count,
        "multi_query_max_chars": config.multi_query_max_chars,
        "multi_query_temperature": config.multi_query_temperature,
    }
    data = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _query_generation_error(*, question: str, config: HierarchyConfig, error: Exception, latency_ms: float) -> dict[str, Any]:
    original_question = unicodedata.normalize("NFC", question).strip() if isinstance(question, str) else ""
    return {
        "original_question": original_question,
        "queries": [],
        "model": config.gemini_generation_model,
        "generation_latency_ms": latency_ms,
        "status": QUERY_GENERATION_UNAVAILABLE,
        "cache_hit": False,
        "error": _safe_error_message(error),
    }


def _safe_error_message(error: Exception) -> str:
    message = f"{type(error).__name__}: {error}"
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        message = message.replace(api_key, "[REDACTED_API_KEY]")
    message = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[REDACTED_API_KEY]", message)
    return message[:500]


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _page_text(page_start: int, page_end: int) -> str:
    if page_start == page_end:
        return f"tr. {page_start}"
    return f"tr. {page_start}-{page_end}"


def _split_parent_windows(children: list[ResolvedChild], *, max_chars: int) -> list[list[ResolvedChild]]:
    windows: list[list[ResolvedChild]] = []
    current: list[ResolvedChild] = []
    current_chars = 0
    for child in children:
        child_len = len(child.text)
        join_len = 2 if current else 0
        projected = current_chars + join_len + child_len
        if current and projected > max_chars:
            windows.append(current)
            current = [child]
            current_chars = child_len
        else:
            current.append(child)
            current_chars = projected
    if current:
        windows.append(current)
    return windows


def _validate_raw_child(record: dict[str, Any], json_file: Path, record_no: int) -> RawChild:
    for field in rag.REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"{json_file} record #{record_no}: thiếu field {field!r}")
    chunk_id = _required_string(record, "chunk_id", json_file, record_no)
    source = _required_string(record, "source", json_file, record_no)
    text = _required_string(record, "text", json_file, record_no)
    page_start = _required_page(record, "page_start", json_file, record_no)
    page_end = _required_page(record, "page_end", json_file, record_no)
    if page_start > page_end:
        raise ValueError(f"{json_file} record #{record_no}: page_start phải <= page_end")
    structure = _validate_structure(record.get("structure"), json_file, record_no)
    return RawChild(
        chunk_id=chunk_id,
        source=source,
        page_start=page_start,
        page_end=page_end,
        text=text,
        structure=structure,
        json_file=str(json_file),
        record_no=record_no,
        sequence=_final_sequence(chunk_id),
    )


def _validate_structure(value: Any, json_file: Path, record_no: int) -> dict[str, str | None]:
    if value is None:
        return {key: None for key in STRUCTURE_KEYS}
    if not isinstance(value, dict):
        raise ValueError(f"{json_file} record #{record_no}: structure phải là object hoặc null")
    clean: dict[str, str | None] = {key: None for key in STRUCTURE_KEYS}
    for key in STRUCTURE_KEYS:
        raw = value.get(key)
        if raw is None or raw == "":
            continue
        if isinstance(raw, bool) or not isinstance(raw, (str, int)):
            raise ValueError(f"{json_file} record #{record_no}: structure.{key} phải là string/integer hoặc null")
        text = str(raw).strip()
        if not text:
            continue
        clean[key] = text
    return clean


def _detect_headings(text: str) -> dict[str, str | None]:
    first_lines = [line for line in text.splitlines()[:3] if line.strip()]
    first = first_lines[0] if first_lines else text[:300]
    result: dict[str, str | None] = {key: None for key in STRUCTURE_KEYS}
    chapter_match = CHAPTER_HEADING_RE.search(first)
    if chapter_match:
        result["chapter"] = chapter_match.group(1).upper()
    article_match = ARTICLE_HEADING_RE.search(first)
    if article_match:
        result["article"] = article_match.group(1)
    clause_match = CLAUSE_HEADING_RE.search(first)
    if clause_match:
        result["clause"] = clause_match.group(1)
    point_match = POINT_HEADING_RE.search(first)
    if point_match:
        result["point"] = point_match.group(1).lower()
    return result


def _article_candidates(text: str) -> list[str]:
    candidates = []
    for match in INLINE_ARTICLE_RE.finditer(text):
        candidates.append(match.group(1))
    return candidates


def _heading_title(text: str, article: str | None) -> str | None:
    if not article:
        return None
    first = next((line.strip() for line in text.splitlines() if line.strip()), "") or " ".join(text.split())[:160]
    if not first:
        return None
    return re.sub(r"\s+", " ", first).strip(" #*")[:240]


def _clean_structure(structure: dict[str, str | None]) -> dict[str, str | None]:
    return {key: structure.get(key) for key in STRUCTURE_KEYS}


def _article_key(source: str, article: str | None) -> str:
    if article:
        return f"article::{_label_key(article)}"
    return f"document::{_slug(source)}"


def _document_fallback_title(source: str) -> str:
    return f"Document fallback: {source}"


def _parent_id(*, source: str, article_key: str, window_index: int) -> str:
    digest = hashlib.sha256(f"{source}|{article_key}|{window_index}".encode("utf-8")).hexdigest()[:16]
    return f"parent::{digest}::w{window_index:04d}"


def _fingerprint_file(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _config_identity(config: HierarchyConfig) -> dict[str, Any]:
    return {
        "parent_max_chars": config.parent_max_chars,
        "multi_query_count": config.multi_query_count,
        "multi_query_rrf_k": config.multi_query_rrf_k,
        "parent_rrf_k": config.parent_rrf_k,
        "parent_score_child_limit": config.parent_score_child_limit,
        "parent_candidates": config.parent_candidates,
        "final_parent_top_k": config.final_parent_top_k,
        "total_context_max_chars": config.total_context_max_chars,
        "embedding_model": config.gemini_embedding_model,
        "embedding_dim": config.gemini_embedding_dim,
        "reranker_model": config.reranker_model,
    }


def _atomic_write_store(storage_dir: Path, files: dict[str, Any]) -> None:
    storage_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for filename in sorted(files):
            target = storage_dir / filename
            tmp_path = _write_json_temp(target, files[filename])
            staged.append((tmp_path, target))
        for tmp_path, target in staged:
            tmp_path.replace(target)
    except Exception:
        for tmp_path, _target in staged:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        raise


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _write_json_temp(path, payload)
    try:
        tmp_path.replace(path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_json_temp(path: Path, payload: Any) -> Path:
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        return tmp_path
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _json_files(input_path: Path | str) -> list[Path]:
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy input: {path}")
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise ValueError(f"Input file phải có đuôi .json: {path}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input không phải file hoặc thư mục: {path}")
    files = sorted(path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Không có file .json trong thư mục: {path}")
    return files


def _read_records(json_file: Path) -> list[Any]:
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{json_file}: JSON lỗi tại dòng {error.lineno}, cột {error.colno}: {error.msg}") from error
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("chunks"), list):
        return data["chunks"]
    raise ValueError(f"{json_file}: JSON phải là list hoặc object có field 'chunks' là list")


def _ensure_object(record: Any, json_file: Path, record_no: int) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{json_file} record #{record_no}: record phải là object")


def _required_string(record: dict[str, Any], field: str, json_file: Path, record_no: int) -> str:
    value = record[field]
    if not isinstance(value, str):
        raise ValueError(f"{json_file} record #{record_no}: field {field!r} phải là string")
    value = value.strip()
    if not value:
        raise ValueError(f"{json_file} record #{record_no}: field {field!r} không được rỗng")
    return value


def _required_page(record: dict[str, Any], field: str, json_file: Path, record_no: int) -> int:
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{json_file} record #{record_no}: field {field!r} phải là integer")
    if value < 1:
        raise ValueError(f"{json_file} record #{record_no}: field {field!r} phải >= 1")
    return value


def _required_env_string(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Thiếu cấu hình {name}")
    return value


def _required_env_int(name: str, *, min_value: int, max_value: int) -> int:
    value = os.getenv(name, "").strip()
    try:
        number = int(value)
    except ValueError as error:
        raise ValueError(f"{name} phải là integer") from error
    if number < min_value or number > max_value:
        raise ValueError(f"{name} phải nằm trong khoảng {min_value}..{max_value}")
    return number


def _required_env_float(name: str, *, min_value: float, max_value: float | None) -> float:
    value = os.getenv(name, "").strip()
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"{name} phải là float") from error
    if math.isnan(number) or math.isinf(number):
        raise ValueError(f"{name} phải hữu hạn")
    if number < min_value or (max_value is not None and number > max_value):
        if max_value is None:
            raise ValueError(f"{name} phải >= {min_value}")
        raise ValueError(f"{name} phải nằm trong khoảng {min_value}..{max_value}")
    return number


def _final_sequence(chunk_id: str) -> int | None:
    match = FINAL_NUMBER_RE.search(chunk_id)
    return int(match.group(1)) if match else None


def _sequence_sort_value(sequence: int | None) -> tuple[int, int]:
    return (0, sequence) if sequence is not None else (1, 0)


def _label_key(value: str | None) -> str:
    return re.sub(r"\s+", "", str(value or "").casefold())


def _slug(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "-", value).strip("-").lower() or "document"


def _counter(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


def _warning_counts(children: list[ResolvedChild], parents: list[ParentDocument]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for child in children:
        for warning in child.warnings:
            counts[warning] = counts.get(warning, 0) + 1
    for parent in parents:
        for warning in parent.warnings:
            counts[warning] = counts.get(warning, 0) + 1
    return dict(sorted(counts.items()))


def _warning_examples(children: list[ResolvedChild], parents: list[ParentDocument]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for child in children:
        if child.warnings or child.ambiguous:
            examples.append({"type": "child", "id": child.child_id, "warnings": child.warnings, "ambiguous": child.ambiguous, "preview": " ".join(child.text.split())[:180]})
        if len(examples) >= 5:
            return examples
    for parent in parents:
        if parent.warnings:
            examples.append({"type": "parent", "id": parent.parent_id, "warnings": parent.warnings, "char_count": parent.char_count})
        if len(examples) >= 5:
            return examples
    return examples


def _size_distribution(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {"min": values[0], "median": _median(values), "p95": _nearest_rank(values, 95), "max": values[-1]}


def _median(values: list[int]) -> float:
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return (values[mid - 1] + values[mid]) / 2


def _nearest_rank(values: list[int], percentile: int) -> int:
    index = max(0, min(len(values) - 1, math.ceil(percentile / 100 * len(values)) - 1))
    return values[index]


if __name__ == "__main__":
    raise SystemExit(main())
