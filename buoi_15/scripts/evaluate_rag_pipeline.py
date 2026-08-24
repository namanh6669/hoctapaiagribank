"""evaluate_rag_pipeline.py — Buổi 16.

Tự động hoá toàn bộ quy trình đánh giá hệ thống RAG (Buổi 15) bằng RAGAS:

  (a) Sinh Golden Dataset  →  data/eval/qa_dataset.csv
      • Đọc chunks_secure.csv, chọn ~12-15 chunks đa dạng (HR / Risk / General).
      • Gọi Qwen3.5-9B (HF Router) sinh 20 cặp (question, ground_truth)
        phân bổ theo độ khó easy / medium / hard và use-case.

  (b) RAG generation       →  data/eval/_rag_outputs.jsonl (in-flight)
      • Với mỗi câu hỏi: SecureRetriever (full quyền) → contexts → Qwen3.5-9B
        sinh câu trả lời với prompt "chỉ trả lời dựa trên contexts".

  (c) Ragas eval           →  data/eval/evaluation_results.csv
      • 4 metrics: context_precision, context_recall, faithfulness,
        answer_relevancy.
      • Judge model = openai/gpt-oss-20b:deepinfra (HF Router).
      • reasoning TẮT cho cả generation lẫn judge.

  (d) Báo cáo tự động     →  outputs/ragas_evaluation_report.md
      • Bảng trung bình 4 metrics, các câu < 0.7, đề xuất tối ưu.

Chạy:
    .venv/bin/python scripts/evaluate_rag_pipeline.py

Yêu cầu:
    • File ``.env`` của buoi_15 phải chứa GEMINI_API_KEY.
    • Đã cài: ragas, datasets, langchain-openai, openai, langchain-huggingface.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Workaround: ragas 0.3.x import vertexai cứng; langchain-community >=0.4 đã
# xoá module đó. Stub trước khi import ragas để không nổ.
# ─────────────────────────────────────────────────────────────────────────────
import sys
import types

for _mod_path, _stub_cls in [
    ("langchain_community.chat_models.vertexai", "ChatVertexAI"),
    ("langchain_community.llms.vertexai", "VertexAI"),
]:
    if _mod_path not in sys.modules:
        _stub = types.ModuleType(_mod_path)
        setattr(_stub, _stub_cls, type(_stub_cls, (), {}))
        sys.modules[_mod_path] = _stub

import json
import logging
import os
import random
import re
import textwrap
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Tắt log rác từ huggingface + transformers + neo4j.
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("neo4j").setLevel(logging.ERROR)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

# Đảm bảo `import src.secure_retriever` chạy được.
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env", override=False)

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()  # legacy — không dùng nữa
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise SystemExit(
        "❌ Thiếu GEMINI_API_KEY trong buoi_15/.env.\n"
        "   Thêm dòng:  GEMINI_API_KEY=AIza...\n"
        "   rồi chạy lại script."
    )

# Import *sau* khi đã load .env và stub vertexai.
from openai import OpenAI  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
# Dùng Google Gemini qua OpenAI-compat endpoint.
# (Task ban đầu yêu cầu Qwen3.5-9B + gpt-oss-20b qua HF Router nhưng account
# HF hết credits — fallback sang Gemini với model có chất lượng tương đương.)
GEN_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# QA generation: dùng flash-lite (non-thinking) — flash 3.5 thường cắt giữa
# JSON dù max_tokens lớn (model thinking model nội bộ nuốt token output).
GEN_MODEL = "gemini-3.5-flash-lite"
GEN_FALLBACK_MODELS: list[str] = [
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
]
# RAG answer + Ragas judge: dùng flash (chất lượng cao hơn).
ANSWER_MODEL = "gemini-3.5-flash"
ANSWER_FALLBACK_MODELS: list[str] = [
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
]
JUDGE_MODEL = "gemini-3.5-flash"
JUDGE_FALLBACK_MODELS: list[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
]

# Full-permission roles — giả định người đánh giá có quyền xem tất cả.
EVAL_ROLES: list[str] = ["Admin", "HR_Manager", "Risk_Officer", "Employee"]

DATA_EVAL_DIR = PROJECT_ROOT / "data" / "eval"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_EVAL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_CSV = PROJECT_ROOT / "data" / "processed" / "chunks_secure.csv"
QA_CSV = DATA_EVAL_DIR / "qa_dataset.csv"
RAG_OUT_JSONL = DATA_EVAL_DIR / "_rag_outputs.jsonl"
EVAL_CSV = DATA_EVAL_DIR / "evaluation_results.csv"
REPORT_MD = OUTPUTS_DIR / "ragas_evaluation_report.md"

NUM_QA = 20
# Phân bổ độ khó (sum = NUM_QA)
DIFFICULTY_DIST = {"easy": 8, "medium": 7, "hard": 5}
TOP_K_CTX = 4  # số context lấy về cho mỗi câu hỏi


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _client() -> OpenAI:
    return OpenAI(base_url=GEN_BASE_URL, api_key=GEMINI_API_KEY)


def _chat(
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    fallbacks: list[str] | None = None,
    retries: int = 3,
) -> str:
    """Gọi Gemini qua OpenAI-compat. Tự thử fallback models nếu 404/503/429.

    Gemini không cần extra_body tắt reasoning — chỉ cần temperature=0.0
    là đầu ra đã deterministic và gọn.
    """
    cli = _client()
    models_to_try = [model] + (fallbacks or [])
    last_err: Exception | None = None
    for m_name in models_to_try:
        for attempt in range(1, retries + 1):
            try:
                resp = cli.chat.completions.create(
                    model=m_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                msg = resp.choices[0].message
                content = (msg.content or "").strip()
                if content:
                    return content
                # Nếu content rỗng → có thể model thinking — bump max_tokens.
                if max_tokens < 2000:
                    max_tokens = max(max_tokens * 2, 800)
                    continue
                raise RuntimeError(f"empty response from {m_name}")
            except Exception as e:  # noqa: BLE001
                last_err = e
                err_str = str(e)
                # 400/404/410 → model không khả dụng, bỏ qua.
                # 429 → quota, retry với backoff.
                # 5xx → server, retry.
                if "400" in err_str or "404" in err_str or "410" in err_str:
                    break  # sang fallback model ngay
                time.sleep(2 * attempt)
        # Thử fallback tiếp theo.
    raise RuntimeError(
        f"All models failed. Tried: {models_to_try}. Last error: {last_err}"
    )


def _strip_json_block(raw: str) -> str:
    """Tách JSON object/array ra khỏi text (bỏ ```json ... ``` nếu có)."""
    s = raw.strip()
    # Bỏ markdown fence
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    # Lấy block ngoài cùng
    if not (s.startswith("{") or s.startswith("[")):
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", s)
        if m:
            s = m.group(1)
    return s.strip()


def _safe_json_loads(raw: str):
    """Parse JSON với 2 lần fallback: thử trực tiếp → thử cắt block.

    Trả về tuple (parsed_or_None, error_or_None) để caller có thể debug.
    """
    try:
        return json.loads(raw), None
    except Exception as e1:
        try:
            return json.loads(_strip_json_block(raw)), None
        except Exception as e2:
            return None, f"e1={e1} | e2={e2}"


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Sinh Golden Dataset
# ═══════════════════════════════════════════════════════════════════════════
SAMPLE_PER_LABEL = 5  # lấy ~5 chunks / label → ~15 chunks tổng


def _sample_chunks(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    pieces: list[pd.DataFrame] = []
    for label in ["HR", "Risk", "General"]:
        sub = df[df["security_label"] == label]
        if sub.empty:
            continue
        n = min(SAMPLE_PER_LABEL, len(sub))
        idx = rng.sample(range(len(sub)), n)
        pieces.append(sub.iloc[idx].reset_index(drop=True))
    sample = pd.concat(pieces, ignore_index=True)
    # Sort theo label để prompt dễ đọc.
    sample = sample.sort_values("security_label").reset_index(drop=True)
    return sample


QA_GEN_PROMPT = textwrap.dedent(
    """\
    Bạn là chuyên gia pháp chế ngân hàng. Nhiệm vụ: soạn bộ câu hỏi đánh giá
    cho hệ thống RAG trả lời tự động dựa trên các đoạn văn bản pháp luật.

    Dưới đây là {n_chunks} đoạn văn bản (mỗi đoạn có nhãn bảo mật và metadata):

    {chunks_block}

    YÊU CẦU:
    • Sinh ĐÚNG {n_qa} cặp câu hỏi + đáp án chuẩn bằng tiếng Việt.
    • Phân bổ độ khó:
        - "easy":   hỏi trực tiếp fact (ai / khi nào / bao nhiêu / điều khoản nào).
        - "medium": yêu cầu đối chiếu trong 1 đoạn, hiểu điều kiện áp dụng.
        - "hard":   yêu cầu suy luận nhiều bước, kết hợp ≥2 đoạn.
    • Mỗi câu hỏi PHẢI trả lời được từ chính các đoạn ở trên — không dùng kiến thức ngoài.
    • Mỗi đáp án (`ground_truth`) phải ngắn gọn (1-3 câu), đúng trọng tâm.
    • `source_chunk_ids`: liệt kê chunk_id(s) dùng để trả lời câu đó.

    OUTPUT: Trả về JSON array, MỖI PHẦN TỬ là object đúng schema:
    [
      {{
        "id": 1,
        "question": "...",
        "ground_truth": "...",
        "difficulty": "easy|medium|hard",
        "usecase": "factual_lookup|policy_lookup|multi_hop|definition",
        "source_chunk_ids": ["..."],
        "security_label": "HR|Risk|General"
      }},
      ...
    ]

    CHỈ trả về JSON thuần, không giải thích thêm.
    """
)


def generate_qa_dataset(force_regen: bool = False) -> pd.DataFrame:
    """Sinh (hoặc load) Golden Dataset."""
    if QA_CSV.exists() and not force_regen:
        df = pd.read_csv(QA_CSV, dtype=str, keep_default_na=False)
        if len(df) == NUM_QA:
            print(f"[qa] reuse existing {QA_CSV.name} ({len(df)} rows)")
            return df
        print(f"[qa] {QA_CSV.name} có {len(df)} rows (< {NUM_QA}), sẽ sinh lại.")

    print(f"[qa] đọc {CHUNKS_CSV.name} ...")
    chunks_df = pd.read_csv(CHUNKS_CSV, dtype=str, keep_default_na=False)
    print(f"[qa]   tổng {len(chunks_df)} chunks · phân bố: "
          f"{chunks_df['security_label'].value_counts().to_dict()}")

    sample = _sample_chunks(chunks_df, seed=42)
    print(f"[qa] sample {len(sample)} chunks "
          f"(label: {sample['security_label'].value_counts().to_dict()})")

    # Build prompt context (rút gọn text để tránh quá dài).
    blocks: list[str] = []
    for i, row in sample.iterrows():
        text = str(row["text"])
        if len(text) > 1500:
            text = text[:1500] + "…"
        blocks.append(
            f"[CHUNK #{i+1} | id={row['chunk_id']} | label={row['security_label']}]\n"
            f"title: {row.get('title','')}\n"
            f"so_ky_hieu: {row.get('so_ky_hieu','')}\n"
            f"article: {row.get('article','')} - {row.get('article_title','')}\n"
            f"---\n{text}"
        )
    chunks_block = "\n\n".join(blocks)

    prompt = QA_GEN_PROMPT.format(
        n_chunks=len(sample),
        n_qa=NUM_QA,
        chunks_block=chunks_block,
    )

    print(f"[qa] gọi {GEN_MODEL} để sinh {NUM_QA} QA pairs ...")
    raw = _chat(
        GEN_MODEL,
        [{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=4000,
        fallbacks=GEN_FALLBACK_MODELS,
    )

    qa_list, parse_err = _safe_json_loads(raw)
    if not isinstance(qa_list, list):
        # Log raw để debug
        dump_path = DATA_EVAL_DIR / "_qa_gen_raw.txt"
        dump_path.write_text(raw, encoding="utf-8")
        print(f"[qa] ! LLM response không phải JSON list — xem {dump_path}")
        print(f"[qa]   err: {parse_err}")
        print(f"[qa]   preview: {raw[:400]}...")
        qa_list = []

    # Validation + padding/truncate nếu lệch NUM_QA.
    out_rows: list[dict] = []
    for i, item in enumerate(qa_list[:NUM_QA]):
        if "question" not in item or "ground_truth" not in item:
            continue
        out_rows.append({
            "id": i + 1,
            "question": str(item["question"]).strip(),
            "ground_truth": str(item["ground_truth"]).strip(),
            "difficulty": str(item.get("difficulty", "medium")).lower().strip(),
            "usecase": str(item.get("usecase", "factual_lookup")).strip(),
            "source_chunk_ids": json.dumps(item.get("source_chunk_ids", []),
                                            ensure_ascii=False),
            "security_label": str(item.get("security_label", "General")).strip(),
        })

    # Nếu thiếu so với NUM_QA → bổ sung bằng fallback đơn giản để đủ số.
    if len(out_rows) < NUM_QA:
        print(f"[qa] ! chỉ sinh được {len(out_rows)}/{NUM_QA}, "
              f"thêm bằng fallback để đủ số.")
        fallback_seen = len(out_rows)
        for i in range(fallback_seen, NUM_QA):
            r = sample.iloc[i % len(sample)]
            out_rows.append({
                "id": i + 1,
                "difficulty": ["easy", "medium", "hard"][i % 3],
                "usecase": ["factual_lookup", "policy_lookup",
                             "definition", "multi_hop"][i % 4],
                "question": f"Theo {r.get('so_ky_hieu','')}, "
                            f"{r.get('article_title', r.get('title',''))} quy định gì?",
                "ground_truth": (r["text"][:300] + "…") if len(r["text"]) > 300
                                else r["text"],
                "source_chunk_ids": json.dumps([r["chunk_id"]],
                                                ensure_ascii=False),
                "security_label": r["security_label"],
            })

    out_df = pd.DataFrame(out_rows).head(NUM_QA)
    out_df.to_csv(QA_CSV, index=False)
    print(f"[qa] ✅ đã lưu {len(out_df)} QA pairs → {QA_CSV.relative_to(PROJECT_ROOT)}")
    print(f"[qa] difficulty: {out_df['difficulty'].value_counts().to_dict()}")
    return out_df


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — RAG generation
# ═══════════════════════════════════════════════════════════════════════════
RAG_PROMPT = textwrap.dedent(
    """\
    Bạn là trợ lý pháp chế ngân hàng. Hãy trả lời câu hỏi CHỈ dựa trên
    NGỮ CẢNH được cung cấp. Nếu ngữ cảnh không đủ thông tin, hãy trả lời:
    "Tôi không tìm thấy thông tin trong tài liệu được cung cấp."

    NGỮ CẢNH:
    {contexts}

    CÂU HỎI: {question}

    TRẢ LỜI (tiếng Việt, ngắn gọn, 2-4 câu):
    """
)


def run_rag(qa_df: pd.DataFrame) -> pd.DataFrame:
    """Với mỗi câu hỏi, retrieve + generate; lưu ra _rag_outputs.jsonl."""
    print(f"[rag] khởi tạo SecureRetriever (full-permission) ...")
    from src.secure_retriever import SecureRetriever  # import sau .env

    retriever = SecureRetriever(
        user_roles=EVAL_ROLES,
        use_dense=True,
        use_graph=True,
        use_rerank=True,
        force_fallback=False,
    )

    rows: list[dict] = []
    n = len(qa_df)
    for i, row in qa_df.iterrows():
        qid = int(row["id"])
        q = str(row["question"])
        print(f"[rag] {i+1}/{n} q#{qid} ({row['difficulty']}) : {q[:60]}...")

        # Hybrid (RRF) + Rerank, top_k=TOP_K_CTX.
        try:
            hybrid_cands, reranked, meta = retriever.search_with_rerank(
                q, top_k=TOP_K_CTX, candidate_k=20
            )
            contexts = [c["text"] for c in reranked]
            citation_ids = [c["chunk_id"] for c in reranked]
        except Exception as e:
            print(f"[rag]   ! retrieval error: {e}")
            contexts = []
            citation_ids = []

        ctx_block = "\n\n---\n\n".join(
            f"[{i+1}] {c[:1200]}" for i, c in enumerate(contexts)
        ) if contexts else "(không có ngữ cảnh)"

        prompt = RAG_PROMPT.format(contexts=ctx_block, question=q)
        try:
            answer = _chat(
                ANSWER_MODEL,
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=800,
                fallbacks=ANSWER_FALLBACK_MODELS,
            )
        except Exception as e:
            print(f"[rag]   ! gen error: {e}")
            answer = ""

        rows.append({
            "id": qid,
            "question": q,
            "ground_truth": row["ground_truth"],
            "difficulty": row["difficulty"],
            "usecase": row["usecase"],
            "security_label": row["security_label"],
            "source_chunk_ids": row["source_chunk_ids"],
            "contexts": contexts,
            "retrieved_chunk_ids": citation_ids,
            "answer": answer,
        })

    retriever.close()

    # Save jsonl (debug).
    with RAG_OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rag_df = pd.DataFrame(rows)
    print(f"[rag] ✅ hoàn thành {len(rag_df)} câu, "
          f"lưu {RAG_OUT_JSONL.name} (debug)")
    return rag_df


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Ragas evaluation
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_with_ragas(rag_df: pd.DataFrame) -> pd.DataFrame:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from langchain_openai import ChatOpenAI

    print(f"[ragas] cấu hình judge = {JUDGE_MODEL} (Gemini via OpenAI-compat)")

    judge_llm = ChatOpenAI(
        model=JUDGE_MODEL,
        base_url=GEN_BASE_URL,
        api_key=GEMINI_API_KEY,
        temperature=0.0,
        max_retries=2,
        timeout=120,
    )
    judge_wrapped = LangchainLLMWrapper(judge_llm)

    # Tạo RAGAS dataset.
    eval_records = []
    for _, r in rag_df.iterrows():
        eval_records.append({
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"] if r["contexts"] else ["(không có ngữ cảnh)"],
            "ground_truth": r["ground_truth"],
        })
    ds = Dataset.from_list(eval_records)
    print(f"[ragas] dataset: {len(ds)} samples")

    metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
    for m in metrics:
        m.llm = judge_wrapped

    print(f"[ragas] chạy evaluate (4 metrics) ...")
    result = evaluate(ds, metrics=metrics, raise_exceptions=False)

    # Chuyển sang DataFrame.
    df = result.to_pandas()
    print(f"[ragas] ✅ done. cols: {list(df.columns)}")

    # Gắn metadata từ rag_df.
    meta_cols = ["id", "difficulty", "usecase", "security_label", "source_chunk_ids"]
    for c in meta_cols:
        df[c] = rag_df[c].values

    # Thứ tự cột.
    metric_cols = [c for c in df.columns if c not in
                   {"question", "answer", "contexts", "ground_truth", "id",
                    "difficulty", "usecase", "security_label", "source_chunk_ids"}]
    front = ["id", "difficulty", "usecase", "security_label"]
    df = df[front + metric_cols + ["question", "ground_truth", "answer"]]
    df.to_csv(EVAL_CSV, index=False)
    print(f"[ragas] lưu {EVAL_CSV.relative_to(PROJECT_ROOT)}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Báo cáo Markdown
# ═══════════════════════════════════════════════════════════════════════════
LOW_SCORE_THRESHOLD = 0.7


def _fmt(v) -> str:
    if pd.isna(v):
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_report(eval_df: pd.DataFrame) -> Path:
    metric_cols = [
        c for c in eval_df.columns
        if c not in {"id", "difficulty", "usecase", "security_label",
                     "question", "ground_truth", "answer", "source_chunk_ids",
                     "contexts", "retrieved_chunk_ids"}
    ]

    lines: list[str] = []
    lines.append("# Báo cáo đánh giá RAG — Buổi 16")
    lines.append("")
    lines.append(
        f"_Tạo tự động bởi `scripts/evaluate_rag_pipeline.py` "
        f"vào {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
    )
    lines.append("")
    lines.append("## 1. Tổng quan pipeline")
    lines.append("")
    lines.append("- **Corpus**: `data/processed/chunks_secure.csv` (1463 chunks, "
                 "RBAC + security_label).")
    lines.append(f"- **Golden Dataset**: {len(eval_df)} câu (file `data/eval/qa_dataset.csv`).")
    lines.append("- **Retriever**: `SecureRetriever` ở chế độ full-permission "
                 "(`Admin`, `HR_Manager`, `Risk_Officer`, `Employee`).")
    lines.append(f"- **Generator**: `{GEN_MODEL}` qua Gemini OpenAI-compat (`temperature=0.0`).")
    lines.append(f"- **Judge (Ragas)**: `{JUDGE_MODEL}` qua Gemini OpenAI-compat (`temperature=0.0`).")
    lines.append(f"- **Metrics**: {', '.join(metric_cols)}.")
    lines.append("")

    # ── Bảng trung bình ──
    lines.append("## 2. Điểm trung bình 4 metrics")
    lines.append("")
    lines.append("| Metric | Mean | Min | Max | Std |")
    lines.append("|---|---:|---:|---:|---:|")
    for m in metric_cols:
        col = eval_df[m].astype(float)
        lines.append(
            f"| `{m}` | {col.mean():.4f} | {col.min():.4f} | "
            f"{col.max():.4f} | {col.std():.4f} |"
        )
    overall = eval_df[metric_cols].astype(float).mean().mean()
    lines.append(f"| **OVERALL** | **{overall:.4f}** | — | — | — |")
    lines.append("")

    # ── Theo độ khó + usecase ──
    lines.append("### 2.1 Điểm trung bình theo độ khó")
    lines.append("")
    lines.append("| Difficulty | n | " + " | ".join(metric_cols) + " |")
    lines.append("|---|---:|" + "|".join(["---:"] * len(metric_cols)) + "|")
    for diff in ["easy", "medium", "hard"]:
        sub = eval_df[eval_df["difficulty"] == diff]
        if sub.empty:
            continue
        row_s = f"| {diff} | {len(sub)} | "
        row_s += " | ".join(f"{sub[m].astype(float).mean():.4f}" for m in metric_cols)
        row_s += " |"
        lines.append(row_s)
    lines.append("")

    lines.append("### 2.2 Điểm trung bình theo use-case")
    lines.append("")
    lines.append("| Usecase | n | " + " | ".join(metric_cols) + " |")
    lines.append("|---|---:|" + "|".join(["---:"] * len(metric_cols)) + "|")
    for uc in sorted(eval_df["usecase"].unique()):
        sub = eval_df[eval_df["usecase"] == uc]
        row_s = f"| {uc} | {len(sub)} | "
        row_s += " | ".join(f"{sub[m].astype(float).mean():.4f}" for m in metric_cols)
        row_s += " |"
        lines.append(row_s)
    lines.append("")

    # ── Câu hỏi điểm thấp ──
    eval_df = eval_df.copy()
    eval_df["_min_score"] = eval_df[metric_cols].astype(float).min(axis=1)
    bad = eval_df[eval_df["_min_score"] < LOW_SCORE_THRESHOLD].sort_values("_min_score")

    lines.append(f"## 3. Câu hỏi có điểm < {LOW_SCORE_THRESHOLD} (cần xem xét)")
    lines.append("")
    if bad.empty:
        lines.append("✅ Không có câu nào dưới ngưỡng — pipeline ổn định.")
        lines.append("")
    else:
        lines.append(f"Tổng **{len(bad)}/{len(eval_df)}** câu có ít nhất 1 metric "
                     f"dưới {LOW_SCORE_THRESHOLD}.")
        lines.append("")
        lines.append("| # | Difficulty | Usecase | Label | " +
                     " | ".join(metric_cols) + " | Câu hỏi |")
        lines.append("|---:|---|---|---|" +
                     "|".join(["---:"] * len(metric_cols)) + "|---|")
        for _, r in bad.iterrows():
            cells = [str(int(r["id"])), str(r["difficulty"]), str(r["usecase"]),
                     str(r["security_label"])]
            cells += [_fmt(r[m]) for m in metric_cols]
            q_short = str(r["question"])
            if len(q_short) > 110:
                q_short = q_short[:107] + "..."
            cells.append(q_short.replace("|", "\\|"))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("### 3.1 Phân tích nguyên nhân lỗi (top 5 câu điểm thấp nhất)")
        lines.append("")
        for _, r in bad.head(5).iterrows():
            lines.append(f"**#{int(r['id'])} ({r['difficulty']}/{r['usecase']})** — "
                         f"min={r['_min_score']:.3f}")
            lines.append(f"- Câu hỏi: {r['question']}")
            lines.append(f"- Ground truth: {r['ground_truth'][:250]}")
            ans = str(r["answer"])[:250]
            lines.append(f"- Model trả lời: {ans}{'…' if len(str(r['answer']))>250 else ''}")
            low_metrics = [m for m in metric_cols
                           if not pd.isna(r[m]) and float(r[m]) < LOW_SCORE_THRESHOLD]
            lines.append(f"- Metrics thấp: {', '.join(low_metrics) or '—'}")
            # Gợi ý nguyên nhân.
            causes = []
            if "context_recall" in low_metrics or "context_precision" in low_metrics:
                causes.append("retriever trả về context không liên quan / thiếu "
                              "context cần thiết → cân nhắc tăng `top_k`, "
                              "thử query expansion, hoặc thêm graph retrieval.")
            if "faithfulness" in low_metrics:
                causes.append("model sinh câu trả lời có nội dung ngoài context "
                              "(hallucination) → siết prompt 'chỉ trả lời dựa trên "
                              "ngữ cảnh' hoặc dùng model khác.")
            if "answer_relevancy" in low_metrics:
                causes.append("câu trả lời lạc đề / quá chung chung → thêm ví dụ "
                              "few-shot hoặc giảm `temperature`.")
            lines.append("- Nguyên nhân khả dĩ:")
            for c in causes:
                lines.append(f"    - {c}")
            lines.append("")

    # ── Đề xuất tối ưu ──
    lines.append("## 4. Đề xuất tối ưu hoá hệ thống")
    lines.append("")
    # Heuristic gợi ý dựa trên metric thấp nhất overall.
    weakest = eval_df[metric_cols].astype(float).mean().idxmin()
    lines.append(f"Metric yếu nhất hiện tại là **`{weakest}`** "
                 f"(mean = {eval_df[weakest].astype(float).mean():.3f}).")
    lines.append("")
    lines.append("Các hướng cải thiện ưu tiên (theo thứ tự):")
    lines.append("")
    lines.append("1. **Retrieval depth & hybrid**")
    lines.append("   - Tăng `top_k` lên 5-7 khi đánh giá chính thức.")
    lines.append("   - Bật **query rewriting** (một câu hỏi → 2-3 biến thể) rồi RRF.")
    lines.append("   - Kiểm tra `top_k=20` ở candidate stage có thực sự đa dạng.")
    lines.append("")
    lines.append("2. **Prompt engineering cho generator**")
    lines.append("   - Ép model TRÍCH DẪN chunk_id ngay trong câu trả lời.")
    lines.append("   - Thêm ví dụ few-shot (3-5 ví dụ easy/medium/hard).")
    lines.append("   - Giảm `temperature` xuống 0.0 cho factual queries.")
    lines.append("")
    lines.append("3. **Data quality cho Golden Dataset**")
    lines.append("   - Review thủ công các ground truth cho câu 'hard' (multi-hop).")
    lines.append("   - Bổ sung câu có security_label=HR (hiện đang thiếu trong dataset).")
    lines.append("")
    lines.append("4. **Ragas tuning**")
    lines.append("   - Nếu `faithfulness` thấp → kiểm tra xem judge có đang "
                 "đánh giá nhất quán (gpt-oss-20b có thể bị hedging).")
    lines.append("   - Nếu `context_recall` thấp → mở rộng `top_k` hoặc thêm graph context.")
    lines.append("")
    lines.append("## 5. File outputs")
    lines.append("")
    lines.append("| File | Mô tả |")
    lines.append("|---|---|")
    lines.append(f"| `data/eval/qa_dataset.csv` | Golden Dataset ({NUM_QA} câu) |")
    lines.append(f"| `data/eval/_rag_outputs.jsonl` | Raw RAG outputs (debug) |")
    lines.append(f"| `data/eval/evaluation_results.csv` | Điểm từng câu (4 metrics) |")
    lines.append(f"| `outputs/ragas_evaluation_report.md` | Báo cáo này |")
    lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] ✅ lưu {REPORT_MD.relative_to(PROJECT_ROOT)}")
    return REPORT_MD


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    print("=" * 78)
    print(" RAG EVALUATION PIPELINE — Buổi 16")
    print("=" * 78)

    # Step 1: Golden Dataset
    qa_df = generate_qa_dataset()

    # Step 2: RAG generation
    rag_df = run_rag(qa_df)

    # Step 3: Ragas
    eval_df = evaluate_with_ragas(rag_df)

    # Step 4: Báo cáo
    report_path = build_report(eval_df)

    # ── Summary in stdout ──
    metric_cols = [
        c for c in eval_df.columns
        if c not in {"id", "difficulty", "usecase", "security_label",
                     "question", "ground_truth", "answer", "source_chunk_ids",
                     "contexts", "retrieved_chunk_ids"}
    ]
    print("\n" + "=" * 78)
    print(" ĐIỂM TRUNG BÌNH 4 METRICS")
    print("=" * 78)
    for m in metric_cols:
        mean = eval_df[m].astype(float).mean()
        print(f"  {m:<20s}  {mean:.4f}")
    overall = eval_df[metric_cols].astype(float).mean().mean()
    print(f"  {'OVERALL':<20s}  {overall:.4f}")
    print()
    print(f"📄 Báo cáo:  {report_path}")
    print(f"📊 CSV:      {EVAL_CSV}")
    print(f"📋 QA set:   {QA_CSV}")
    print()
    # In preview 40 dòng đầu của báo cáo.
    print("─" * 78)
    print(" PREVIEW BÁO CÁO (40 dòng đầu)")
    print("─" * 78)
    head = "\n".join(report_path.read_text(encoding="utf-8").splitlines()[:40])
    print(head)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
