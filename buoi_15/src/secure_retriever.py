"""secure_retriever.py — Buổi 15.

Bộ tìm kiếm **an toàn** (RBAC-aware) cho hệ thống knowledge graph.

Bao gồm 3 retrieval methods, T�T CẢ đều lọc theo quyền trước khi đưa
ứng viên sang các bước tiếp theo:

  • SecureBM25Retriever      — pre-filter DataFrame, build BM25 trên subset
  • SecureDenseRetriever     — full-corpus encode + post-filter
  • SecureGraphRetriever     — Cypher với ``WHERE any(...)`` tích hợp

Sau đó:
  • SecureHybridRetriever    — RRF trên candidates đã lọc
  • SecureRerankPipeline     — rerank candidates đã lọc (có fallback)

Quy tắc an toàn (rất cứng):
  1. RBAC filter chạy TRƯỚC mọi tính toán điểm.
  2. Không có candidate nào bị cấm lọt vào RRF hoặc reranker.
  3. Output schema chuẩn hoá, có thêm ``allowed_roles`` + ``security_label``.

Đọc cấu hình Neo4j từ ``buoi_15/.env`` (xem ``src.config.get_neo4j_config``).
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# ── Project setup ─────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    VALID_ROLES,
    ROLE_LIST,
    assert_valid_role,
    get_neo4j_config,
)

# Ép HF_HOME trong project để không leak ra ngoài.
os.environ.setdefault(
    "HF_HOME",
    str(PROJECT_ROOT / "cache" / "huggingface"),
)

DEFAULT_CORPUS = PROJECT_ROOT / "data" / "processed" / "chunks_secure.csv"
DEFAULT_DENSE_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_K_RRF = 60
LAB_SESSION = "buoi_15"

# Token regex (giữ mã văn bản có dấu `/` `-` `.` dính vào token)
_TOKEN_RE = re.compile(
    r"[À-ỹA-Za-z0-9]+(?:[/\-\.][À-ỹA-Za-z0-9]+)*",
    re.UNICODE,
)

# Map role names "legacy" (từ đề bài) → canonical VALID_ROLES
_LEGACY_ROLE_MAP: dict[str, str] = {
    "HR": "HR_Manager",
    "Risk_Manager": "Risk_Officer",
    "Staff": "Employee",
    "Admin": "Admin",
    "HR_Manager": "HR_Manager",
    "Risk_Officer": "Risk_Officer",
    "Employee": "Employee",
    "Guest": "Guest",
}

# Schema kết quả (xem tài liệu ở đầu file)
RESULT_COLUMNS: list[str] = [
    "final_rank",
    "chunk_id",
    "document_id",
    "title",
    "text",
    "citation",
    "allowed_roles",
    "security_label",
    "bm25_rank",
    "dense_rank",
    "graph_rank",
    "rrf_score",
    "rerank_score",
    "retrieval_method",
    "is_fallback",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. RBAC filter
# ─────────────────────────────────────────────────────────────────────────────


def normalize_roles(user_roles: Iterable[str]) -> list[str]:
    """Map mọi biến thể role → canonical ``VALID_ROLES`` (drop unknown)."""
    out: list[str] = []
    for r in user_roles:
        canon = _LEGACY_ROLE_MAP.get(str(r).strip())
        if canon and canon in VALID_ROLES:
            if canon not in out:
                out.append(canon)
    return out


def _row_allowed_roles(row: pd.Series) -> list[str]:
    """Parse cột ``allowed_roles`` (chuỗi CSV) → list[str] canonical."""
    raw = row.get("allowed_roles", "")
    if isinstance(raw, list):
        items = raw
    else:
        items = [x.strip() for x in str(raw).split(",")]
    out = []
    for x in items:
        if not x:
            continue
        # Chuẩn hoá nếu là legacy
        canon = _LEGACY_ROLE_MAP.get(x, x)
        if canon in VALID_ROLES and canon not in out:
            out.append(canon)
    return out


class RBACFilter:
    """Bộ lọc trung tâm — dùng chung cho mọi retrieval method.

    Một chunk được coi là "được phép" khi CẢ HAI điều kiện đều đúng:
      1. Chunk-level: user có role trong ``allowed_roles`` của chunk.
      2. VanBan-level: user có role trong ``vanban_allowed_roles`` của document
         chứa chunk (mặc định = INTERSECTION của tất cả allowed_roles trong
         document đó — đảm bảo người xem VanBan cũng thấy được MỌI chunk bên
         trong).
    """

    def __init__(self, df: pd.DataFrame, user_roles: list[str]):
        if not user_roles:
            raise ValueError(
                "user_roles không được rỗng — phải chỉ định ít nhất 1 role."
            )
        for r in user_roles:
            assert_valid_role(r)  # raise nếu typo

        self.user_roles = list(user_roles)
        self.user_set = set(user_roles)

        # Pre-parse allowed_roles + VanBan-level cho mỗi chunk.
        self._allowed_by_chunk: dict[str, list[str]] = {}
        self._label_by_chunk: dict[str, str] = {}
        self._vanban_by_chunk: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            cid = str(row["chunk_id"])
            self._allowed_by_chunk[cid] = _row_allowed_roles(row)
            self._label_by_chunk[cid] = str(row.get("security_label", "")).strip()
            if "vanban_allowed_roles" in df.columns:
                vb_raw = row.get("vanban_allowed_roles", "")
                if isinstance(vb_raw, list):
                    self._vanban_by_chunk[cid] = [r for r in vb_raw if r in VALID_ROLES]
                else:
                    self._vanban_by_chunk[cid] = [
                        r.strip() for r in str(vb_raw).split(",") if r.strip()
                    ]
            else:
                self._vanban_by_chunk[cid] = list(self._allowed_by_chunk[cid])

    def is_allowed(self, chunk_id: str) -> bool:
        """True nếu user có ÍT NHẤT 1 role nằm trong CẢ allowed_roles của chunk
        VÀ vanban_allowed_roles của VanBan chứa chunk đó."""
        cid = str(chunk_id)
        chunk_roles = self._allowed_by_chunk.get(cid, [])
        if not any(r in self.user_set for r in chunk_roles):
            return False
        vanban_roles = self._vanban_by_chunk.get(cid, [])
        return any(r in self.user_set for r in vanban_roles)

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trả về DataFrame chỉ chứa các chunk user được phép xem."""
        mask = df["chunk_id"].astype(str).map(self.is_allowed)
        return df[mask].reset_index(drop=True)

    def annotate(self, result: dict) -> dict:
        """Gắn ``allowed_roles`` + ``security_label`` + VanBan roles."""
        cid = str(result.get("chunk_id", ""))
        result["allowed_roles"] = list(self._allowed_by_chunk.get(cid, []))
        result["security_label"] = self._label_by_chunk.get(cid, "")
        result["vanban_allowed_roles"] = list(self._vanban_by_chunk.get(cid, []))
        return result

    def annotate_many(self, results: list[dict]) -> list[dict]:
        return [self.annotate(r) for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Corpus loader
# ─────────────────────────────────────────────────────────────────────────────


def load_secure_corpus(csv_path: Path | None = None) -> pd.DataFrame:
    """Đọc ``chunks_secure.csv`` và chuẩn hoá schema."""
    csv_path = Path(csv_path) if csv_path else DEFAULT_CORPUS
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Corpus không tồn tại: {csv_path}. "
            f"Chạy `scripts/assign_security_tags.py` trước."
        )
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if "allowed_roles" not in df.columns:
        raise ValueError(
            f"{csv_path} thiếu cột 'allowed_roles' — "
            f"chạy `scripts/assign_security_tags.py` trước."
        )
    if df["chunk_id"].duplicated().any():
        dup = df[df["chunk_id"].duplicated(keep=False)]["chunk_id"].head()
        raise ValueError(f"chunk_id phải duy nhất. Ví dụ trùng: {dup.tolist()}")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Citation + result formatting
# ─────────────────────────────────────────────────────────────────────────────


def _shorten(s: str, max_len: int = 140) -> str:
    s = s.strip()
    return s if len(s) <= max_len else s[: max_len - 1].rstrip() + "…"


def make_citation(row: pd.Series | dict) -> str:
    if isinstance(row, pd.Series):
        get = row.get
    else:
        get = lambda k: row.get(k, "")  # noqa: E731
    title = str(get("title") or "").strip()
    so = str(get("so_ky_hieu") or "").strip()
    doc_id = str(get("document_id") or "").strip()
    article = str(get("article") or "").strip()
    cid = str(get("chunk_id") or "").strip()
    head = title
    if so and so not in head:
        head = f"{head} ({so})" if head else so
    if not head:
        head = doc_id or "(không rõ)"
    head = _shorten(head, 140)
    parts = [head]
    if article:
        parts.append(f"Điều {article}")
    parts.append(cid)
    return "[" + " | ".join(parts) + "]"


# ─────────────────────────────────────────────────────────────────────────────
# 4. BM25 (lexical) — PRE-FILTER
# ─────────────────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = unicodedata.normalize("NFC", text).lower()
    return [t for t in _TOKEN_RE.findall(text) if t]


class SecureBM25Retriever:
    """BM25 + RBAC pre-filter: chỉ build index trên chunks user được phép xem."""

    method = "BM25"

    def __init__(self, df: pd.DataFrame, filter: RBACFilter):
        self.filter = filter
        self.df_full = df.reset_index(drop=True)
        self.df_allowed = filter.filter_dataframe(self.df_full)
        if self.df_allowed.empty:
            raise ValueError(
                f"User roles {filter.user_roles} không có quyền với chunk nào "
                f"trong corpus ({len(self.df_full)} chunks)."
            )

        texts = self.df_allowed["text"].fillna("").astype(str).tolist()
        self._tokens = [_tokenize(t) for t in texts]

        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi(self._tokens)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        idx_sorted = np.argsort(-scores, kind="stable")
        idx_sorted = [int(i) for i in idx_sorted if scores[int(i)] > 0][:top_k]

        out: list[dict] = []
        for rank, i in enumerate(idx_sorted, 1):
            row = self.df_allowed.iloc[i]
            r = {
                "rank": rank,
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "title": str(row.get("title", "")),
                "text": str(row.get("text", "")),
                "retrieval_score": float(scores[i]),
                "retrieval_method": self.method,
                "citation": make_citation(row),
            }
            self.filter.annotate(r)  # gắn allowed_roles + security_label
            out.append(r)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dense (semantic) — POST-FILTER
# ─────────────────────────────────────────────────────────────────────────────


class SecureDenseRetriever:
    """Dense embedding + RBAC post-filter (vẫn cache trên full corpus để
    tái sử dụng giữa các role khác nhau)."""

    method = "DENSE"

    def __init__(
        self,
        df: pd.DataFrame,
        filter: RBACFilter,
        model_name: str = DEFAULT_DENSE_MODEL,
        cache_root: Path | None = None,
        show_progress: bool = False,
    ):
        self.filter = filter
        self.df = df.reset_index(drop=True)
        self.model_name = model_name

        # Cache trong project
        cache_root = Path(cache_root) if cache_root else PROJECT_ROOT / "cache"
        safe = model_name.replace("/", "__").replace("\\", "__")
        cache_dir = cache_root / "dense_index" / safe

        import torch  # noqa: F401  (needed before sentence_transformers)

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512

        # Tải cache nếu h�p lệ
        emb_path = cache_dir / "embeddings.npy"
        meta_path = cache_dir / "meta.json"
        if emb_path.exists() and meta_path.exists():
            try:
                import json
                import hashlib

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                emb = np.load(emb_path)
                ids = self.df["chunk_id"].astype(str).tolist()
                sig = hashlib.md5("|".join(ids).encode("utf-8")).hexdigest()
                if (
                    meta.get("model") == model_name
                    and meta.get("n_chunks") == len(self.df)
                    and meta.get("signature") == sig
                    and emb.shape[0] == len(self.df)
                ):
                    self.embeddings = emb.astype("float32")
                    print(
                        f"[dense] cache hit  : {cache_dir}  "
                        f"(n={self.embeddings.shape[0]}, dim={self.embeddings.shape[1]})"
                    )
                    return
            except Exception:
                pass  # fallback sang encode lại

        # Build embeddings trên full corpus
        print(
            f"[dense] building   : {len(self.df)} chunks · model={model_name} · device={device}"
        )
        texts = self.df["text"].fillna("").astype(str).tolist()
        e5_texts = ["passage: " + t if t else "passage: " for t in texts]

        import time

        t0 = time.time()
        embs = self.model.encode(
            e5_texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print(f"[dense] encoded in {time.time() - t0:.1f}s, shape={embs.shape}")

        self.embeddings = embs.astype("float32")
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_dir / "embeddings.npy", self.embeddings)
        import json as _json

        (_json.dumps if False else (lambda d, p: Path(p).write_text(
            _json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8"
        )))(
            {
                "model": model_name,
                "n_chunks": int(embs.shape[0]),
                "dim": int(embs.shape[1]),
                "dtype": "float32",
                "signature": __import__("hashlib").md5(
                    "|".join(self.df["chunk_id"].astype(str).tolist()).encode("utf-8")
                ).hexdigest(),
                "device": device,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            cache_dir / "meta.json",
        )
        print(f"[dense] cached to {cache_dir}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []
        q_emb = self.model.encode(
            ["query: " + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype("float32")

        scores = self.embeddings @ q_emb
        idx_sorted = np.argsort(-scores, kind="stable")

        # Post-filter: lấy dư để chắc chắn còn đủ top_k sau khi lọc.
        out: list[dict] = []
        for i in idx_sorted:
            cid = str(self.df.iloc[int(i)]["chunk_id"])
            if not self.filter.is_allowed(cid):
                continue
            row = self.df.iloc[int(i)]
            r = {
                "rank": len(out) + 1,
                "chunk_id": cid,
                "document_id": str(row["document_id"]),
                "title": str(row.get("title", "")),
                "text": str(row.get("text", "")),
                "retrieval_score": float(scores[int(i)]),
                "retrieval_method": self.method,
                "citation": make_citation(row),
            }
            self.filter.annotate(r)
            out.append(r)
            if len(out) >= top_k:
                break
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Graph retrieval — Cypher với WHERE tích hợp RBAC
# ─────────────────────────────────────────────────────────────────────────────


class SecureGraphRetriever:
    """Graph retrieval bằng Cypher, lọc quyền trong chính câu query.

    Search strategy: keyword ``CONTAINS`` trên ``title`` (VanBan) + ``text``
    (DieuKhoan). Không phụ thuộc fulltext index.
    """

    method = "GRAPH"

    def __init__(self, df: pd.DataFrame, filter: RBACFilter):
        self.filter = filter
        # Map chunk_id → row để tra cứu title/text khi graph không có text
        self._rows_by_id = {
            str(r["chunk_id"]): r for _, r in df.iterrows()
        }

        cfg = get_neo4j_config()
        if not (cfg["uri"] and cfg["password"]):
            raise RuntimeError(
                "Thiếu NEO4J_URI hoặc NEO4J_PASSWORD trong .env — không thể "
                "dùng Graph retrieval."
            )
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(
            cfg["uri"], auth=(cfg["user"], cfg["password"])
        )
        self._database = cfg["database"]

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------ search

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Tìm DieuKhoan (chunk) có text chứa query, RBAC lọc ngay trong Cypher.

        Vì ``DieuKhoan.text`` có thể không có trong DB local (chỉ buổi 15
        MERGE mới thêm allowed_roles/lab_session), ta fallback sang title
        của VanBan cha khi text trống.
        """
        if not query or not query.strip():
            return []
        tokens = [t for t in re.findall(r"[À-ỹ\w]+", query) if len(t) >= 2][:5]
        if not tokens:
            return []

        # MATCH VanBan � DieuKhoan qua document_id (độc lập với CONTAINS rel).
        # RBAC enforced ở CẢ VanBan và DieuKhoan — phòng trường hợp cấp
        # doc rộng hơn cấp chunk, vẫn không để chunk cấm lọt.
        cypher = (
            "MATCH (vb:VanBan {lab_session:$lab}), (dk:DieuKhoan {lab_session:$lab}) "
            "WHERE vb.id = dk.document_id "
            "  AND any(role IN vb.allowed_roles WHERE role IN $user_roles) "
            "  AND any(role IN dk.allowed_roles WHERE role IN $user_roles) "
            "  AND ("
            "    reduce(acc = true, t IN $tokens | "
            "      acc AND ("
            "        toLower(coalesce(dk.text, '')) CONTAINS toLower(t) "
            "        OR toLower(coalesce(vb.title, '')) CONTAINS toLower(t)"
            "      )"
            "    )"
            "  ) "
            "RETURN DISTINCT dk.id AS chunk_id, "
            "       dk.document_id AS document_id, "
            "       dk.allowed_roles AS allowed_roles, "
            "       dk.security_label AS security_label, "
            "       CASE "
            "         WHEN vb.title IS NOT NULL THEN vb.title "
            "         ELSE vb.id "
            "       END AS title, "
            "       left(coalesce(dk.text, vb.title, vb.id, ''), 300) AS preview "
            "LIMIT $limit"
        )
        with self._driver.session(database=self._database) as session:
            rows = list(
                session.run(
                    cypher,
                    lab=LAB_SESSION,
                    user_roles=self.filter.user_roles,
                    tokens=tokens,
                    limit=top_k * 5,  # lấy dư để filter trùng
                )
            )

        # De-duplicate by chunk_id, giữ bản đầu.
        seen: set[str] = set()
        unique = []
        for r in rows:
            cid = str(r["chunk_id"])
            if cid in seen:
                continue
            seen.add(cid)
            unique.append(r)

        def _score(row) -> int:
            blob = f"{row.get('title') or ''} {row.get('preview') or ''}".lower()
            return sum(1 for t in tokens if t.lower() in blob)

        unique.sort(key=lambda r: (-_score(r), r["chunk_id"]))

        out: list[dict] = []
        for rank, row in enumerate(unique[:top_k], 1):
            cid = str(row["chunk_id"])
            csv_row = self._rows_by_id.get(cid)
            text = (
                str(csv_row["text"])
                if csv_row is not None and str(csv_row.get("text", "")).strip()
                else str(row["preview"] or "")
            )
            title = (
                str(csv_row.get("title", ""))
                if csv_row is not None and str(csv_row.get("title", "")).strip()
                else str(row["title"] or "")
            )
            doc_id = (
                str(csv_row["document_id"])
                if csv_row is not None
                else str(row["document_id"])
            )
            r = {
                "rank": rank,
                "chunk_id": cid,
                "document_id": doc_id,
                "title": title,
                "text": text,
                "retrieval_score": float(_score(row)),
                "retrieval_method": self.method,
                "citation": make_citation(
                    csv_row if csv_row is not None else {"chunk_id": cid, "document_id": doc_id, "title": title}
                ),
            }
            self.filter.annotate(r)
            out.append(r)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. Hybrid (RRF) trên candidates đã lọc
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SecureHybridHit:
    final_rank: int
    chunk_id: str
    document_id: str
    bm25_rank: int | None
    dense_rank: int | None
    graph_rank: int | None
    rrf_score: float
    text: str
    citation: str
    allowed_roles: list[str]
    security_label: str


class SecureHybridRetriever:
    """RRF fusion trên BM25 + Dense (+ optional Graph)."""

    def __init__(
        self,
        df: pd.DataFrame,
        filter: RBACFilter,
        bm25: SecureBM25Retriever,
        dense: SecureDenseRetriever,
        graph: SecureGraphRetriever | None = None,
        k_rrf: int = DEFAULT_K_RRF,
    ):
        self.df = df.reset_index(drop=True)
        self.filter = filter
        self.bm25 = bm25
        self.dense = dense
        self.graph = graph
        self.k_rrf = k_rrf
        self._rows_by_id = {
            str(r["chunk_id"]): r for _, r in self.df.iterrows()
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict]:
        if not query or not query.strip():
            return []
        if candidate_k < top_k:
            raise ValueError("candidate_k phải >= top_k")

        # 1) Lấy candidates từ từng retriever (ĐÃ LỌC ở cấp retriever rồi).
        bm = self.bm25.search(query, top_k=candidate_k)
        dn = self.dense.search(query, top_k=candidate_k) if self.dense else []
        gr = self.graph.search(query, top_k=candidate_k) if self.graph else []

        bm_rank = {r["chunk_id"]: r["rank"] for r in bm}
        dn_rank = {r["chunk_id"]: r["rank"] for r in dn}
        gr_rank = {r["chunk_id"]: r["rank"] for r in gr}

        all_ids = set(bm_rank) | set(dn_rank) | set(gr_rank)
        if not all_ids:
            return []

        # 2) Tính RRF
        scored: list[tuple[str, float, int | None, int | None, int | None]] = []
        for cid in all_ids:
            rrf = 0.0
            if (b := bm_rank.get(cid)) is not None:
                rrf += 1.0 / (self.k_rrf + b)
            if (d := dn_rank.get(cid)) is not None:
                rrf += 1.0 / (self.k_rrf + d)
            if (g := gr_rank.get(cid)) is not None:
                rrf += 1.0 / (self.k_rrf + g)
            scored.append((cid, rrf, bm_rank.get(cid), dn_rank.get(cid), gr_rank.get(cid)))

        # 3) Sort RRF desc, tie-break chunk_id
        scored.sort(key=lambda x: (-x[1], x[0]))

        # 4) Double-check: tuyệt đối không để chunk bị cấm lọt.
        out: list[dict] = []
        for final_rank, (cid, rrf, bmr, dnr, ggr) in enumerate(scored, 1):
            if not self.filter.is_allowed(cid):
                # Không bao giờ xảy ra (đã lọc ở retriever), nhưng đề phòng.
                continue
            row = self._rows_by_id[cid]
            r = {
                "final_rank": final_rank,
                "chunk_id": cid,
                "document_id": str(row["document_id"]),
                "title": str(row.get("title", "")),
                "text": str(row.get("text", "")),
                "bm25_rank": bmr,
                "dense_rank": dnr,
                "graph_rank": ggr,
                "rrf_score": float(rrf),
                "retrieval_method": "HYBRID",
                "citation": make_citation(row),
            }
            self.filter.annotate(r)
            out.append(r)
            if len(out) >= top_k:
                break
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 8. Reranker (optional) — chỉ rerank trên candidates đã lọc
# ─────────────────────────────────────────────────────────────────────────────


class _IdentityReranker:
    """Không neural reranker — giữ nguyên thứ tự hybrid."""

    name = "FALLBACK identity"
    is_fallback = True

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        sorted_cands = sorted(
            candidates,
            key=lambda x: -(x.get("rrf_score") or x.get("hybrid_score") or 0.0),
        )[:top_k]
        out: list[dict] = []
        for rank, c in enumerate(sorted_cands, 1):
            hybrid_score = c.get("rrf_score") or c.get("hybrid_score") or 0.0
            out.append(
                {
                    "final_rank": rank,
                    "hybrid_rank": c.get("hybrid_rank"),
                    "rrf_score": float(hybrid_score),
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "title": c.get("title", ""),
                    "text": c.get("text", ""),
                    "citation": c.get("citation", ""),
                    "allowed_roles": c.get("allowed_roles", []),
                    "security_label": c.get("security_label", ""),
                    "rerank_score": float(hybrid_score),
                    "retrieval_method": "RERANK",
                    "is_fallback": True,
                }
            )
        return out


class SecureRerankPipeline:
    """Hybrid (RRF) → Rerank. Cả hai bước đều trên candidates đã qua RBAC filter."""

    def __init__(
        self,
        hybrid: SecureHybridRetriever,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        force_fallback: bool = False,
    ):
        self.hybrid = hybrid
        self.filter = hybrid.filter
        if force_fallback:
            print("[rerank] FORCE FALLBACK (--fallback).")
            self.reranker: object = _IdentityReranker()
            return

        try:
            import torch
            from sentence_transformers import CrossEncoder

            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            self.reranker = CrossEncoder(rerank_model, max_length=512, device=device)
            self.reranker.name = f"CrossEncoder:{rerank_model}"
            self.reranker.is_fallback = False  # type: ignore[attr-defined]
            print(f"[rerank] loaded CrossEncoder on device={device}")
        except Exception as e:
            print(f"[rerank] ! Cross-Encoder không khả dụng: {e}")
            print("[rerank] ! Chuyển sang FALLBACK identity.")
            self.reranker = _IdentityReranker()

    def search(
        self, query: str, top_k: int = 5, candidate_k: int = 20
    ) -> tuple[list[dict], list[dict], dict]:
        if not query or not query.strip():
            return [], [], {
                "method": getattr(self.reranker, "name", "?"),
                "is_fallback": getattr(self.reranker, "is_fallback", True),
            }

        # 1) Lấy candidates đã lọc từ Hybrid
        hybrid_candidates = self.hybrid.search(
            query, top_k=candidate_k, candidate_k=candidate_k
        )

        # Chuẩn hoá cho reranker (giữ vị trí hybrid để in BEFORE table)
        normalized = []
        for c in hybrid_candidates:
            normalized.append(
                {
                    "hybrid_rank": c.get("final_rank"),
                    "rrf_score": c.get("rrf_score"),
                    "bm25_rank": c.get("bm25_rank"),
                    "dense_rank": c.get("dense_rank"),
                    "graph_rank": c.get("graph_rank"),
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "title": c.get("title", ""),
                    "text": c.get("text", ""),
                    "citation": c.get("citation", ""),
                    "allowed_roles": c.get("allowed_roles", []),
                    "security_label": c.get("security_label", ""),
                }
            )

        # 2) Rerank — v�n giữ đảm bảo không có chunk bị cấm
        candidates_for_rerank = [
            c for c in normalized if self.filter.is_allowed(c["chunk_id"])
        ]
        if hasattr(self.reranker, "predict"):  # CrossEncoder
            reranked = self._crossencoder_rerank(query, candidates_for_rerank, top_k)
        else:
            reranked = self.reranker.rerank(query, candidates_for_rerank, top_k)

        meta = {
            "method": getattr(self.reranker, "name", "?"),
            "is_fallback": getattr(self.reranker, "is_fallback", True),
        }
        return normalized, reranked, meta

    def _crossencoder_rerank(self, query: str, candidates: list[dict], top_k: int):
        pairs = [(query, (c.get("text") or "")[:5000]) for c in candidates]
        scores = self.reranker.predict(pairs, convert_to_numpy=True)
        for c, s in zip(candidates, [float(x) for x in scores]):
            c["rerank_score"] = s
        sorted_cands = sorted(candidates, key=lambda x: -x["rerank_score"])[:top_k]
        out = []
        for rank, c in enumerate(sorted_cands, 1):
            out.append(
                {
                    "final_rank": rank,
                    "hybrid_rank": c.get("hybrid_rank"),
                    "rrf_score": c.get("rrf_score"),
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "title": c.get("title", ""),
                    "text": c.get("text", ""),
                    "citation": c.get("citation", ""),
                    "allowed_roles": c.get("allowed_roles", []),
                    "security_label": c.get("security_label", ""),
                    "rerank_score": c["rerank_score"],
                    "retrieval_method": "RERANK",
                    "is_fallback": False,
                }
            )
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 9. SecureRetriever — facade
# ─────────────────────────────────────────────────────────────────────────────


class SecureRetriever:
    """Facade — khởi tạo tất cả sub-retrievers và expose API đơn giản."""

    def __init__(
        self,
        csv_path: Path | None = None,
        user_roles: Iterable[str] | None = None,
        use_dense: bool = True,
        use_graph: bool = True,
        use_rerank: bool = True,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        force_fallback: bool = False,
    ):
        self.df = load_secure_corpus(csv_path)
        self.user_roles = normalize_roles(user_roles or [])
        if not self.user_roles:
            raise ValueError(
                "user_roles không được rỗng và phải chứa role hợp lệ."
            )
        self.filter = RBACFilter(self.df, self.user_roles)

        # Sub-retrievers
        self.bm25 = SecureBM25Retriever(self.df, self.filter)
        self.dense = SecureDenseRetriever(self.df, self.filter) if use_dense else None
        self.graph: SecureGraphRetriever | None = None
        if use_graph:
            try:
                self.graph = SecureGraphRetriever(self.df, self.filter)
            except Exception as e:
                print(f"[graph] ! Không khởi tạo được Graph retriever: {e}")
                print("[graph] ! Tiếp tục không có Graph retrieval.")
                self.graph = None

        self.hybrid = SecureHybridRetriever(
            self.df,
            self.filter,
            bm25=self.bm25,
            dense=self.dense,  # type: ignore[arg-type]
            graph=self.graph,
        )
        self.rerank: SecureRerankPipeline | None = None
        if use_rerank:
            self.rerank = SecureRerankPipeline(
                self.hybrid,
                rerank_model=rerank_model,
                force_fallback=force_fallback,
            )

    def close(self) -> None:
        if self.graph is not None:
            self.graph.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Public API ──────────────────────────────────────────────────────────

    def search_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        """BM25 lexical + RBAC pre-filter."""
        return self.bm25.search(query, top_k=top_k)

    def search_dense(self, query: str, top_k: int = 5) -> list[dict]:
        """Dense semantic + RBAC post-filter."""
        if self.dense is None:
            return []
        return self.dense.search(query, top_k=top_k)

    def search_graph(self, query: str, top_k: int = 5) -> list[dict]:
        """Neo4j Cypher + RBAC WHERE clause."""
        if self.graph is None:
            return []
        return self.graph.search(query, top_k=top_k)

    def search_hybrid(
        self, query: str, top_k: int = 5, candidate_k: int = 20
    ) -> list[dict]:
        """RRF trên BM25 + Dense (+ optional Graph). RBAC ở mỗi retriever."""
        return self.hybrid.search(query, top_k=top_k, candidate_k=candidate_k)

    def search_with_rerank(
        self, query: str, top_k: int = 5, candidate_k: int = 20
    ) -> tuple[list[dict], list[dict], dict]:
        """Hybrid → Rerank. RBAC ở mọi bước."""
        if self.rerank is None:
            raise RuntimeError("Rerank pipeline chưa được khởi tạo.")
        return self.rerank.search(query, top_k=top_k, candidate_k=candidate_k)


__all__ = [
    "RBACFilter",
    "normalize_roles",
    "load_secure_corpus",
    "make_citation",
    "SecureBM25Retriever",
    "SecureDenseRetriever",
    "SecureGraphRetriever",
    "SecureHybridRetriever",
    "SecureRerankPipeline",
    "SecureRetriever",
    "RESULT_COLUMNS",
    "DEFAULT_DENSE_MODEL",
    "DEFAULT_RERANK_MODEL",
]
