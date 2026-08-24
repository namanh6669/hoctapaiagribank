"""BƯỚC 3: Entity Extraction và Metadata Enrichment bằng Gemini.

Input : ner_kb/cleaned_documents.csv
Output:
  - ner_kb/extracted_entities_raw.csv  (mỗi entity 1 dòng)
  - ner_kb/enriched_metadata.csv       (mỗi document 1 dòng)
KHÔNG sửa metadata.csv / content.csv.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
INPUT_PATH = BASE / "cleaned_documents.csv"
ENTITIES_OUT = BASE / "extracted_entities_raw.csv"
META_OUT = BASE / "enriched_metadata.csv"

# 4 entity types the user wants.
ALLOWED_TYPES = {"CoQuan", "NguoiKy", "DoiTuongApDung", "LinhVuc"}

# Mapping từ metadata gốc sang entity type
META_TO_ENTITY = {
    "co_quan_ban_hanh": "CoQuan",
    "nguoi_ky": "NguoiKy",
    "linh_vuc": "LinhVuc",
}

# Các trường được Gemini "làm giàu" nếu metadata gốc rỗng / "Chưa phân loại"
ENRICHABLE_FIELDS = [
    "co_quan_ban_hanh",
    "nguoi_ky",
    "linh_vuc",
    "pham_vi",
    "nganh",
    "chuc_danh",
]

# Tên field do Gemini đề xuất cho "đối tượng áp dụng" (không có sẵn trong metadata)
DOI_TUONG_FIELD = "doi_tuong_ap_dung"

# Cắt content khi đưa vào prompt để tránh vượt context
MAX_CONTENT_CHARS = 50000

# Default model id — có thể override qua env GEMINI_MODEL
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Prompt
SYSTEM_INSTRUCTION = (
    "Bạn là một hệ thống trích xuất thực thể từ văn bản pháp luật tiếng Việt. "
    "Nhiệm vụ: trích xuất CÁC THỰC THỂ sau từ văn bản được cung cấp và "
    "(nếu có thể) đề xuất giá trị cho các trường metadata đang trống.\n\n"
    "CÁC LOẠI THỰC THỂ:\n"
    "- CoQuan: cơ quan ban hành (ví dụ: Quốc hội, Chính phủ, "
    "Ngân hàng Nhà nước Việt Nam, Bộ Tài chính, ...)\n"
    "- NguoiKy: người ký văn bản (ví dụ: Vương Đình Huệ, Nguyễn Xuân Phúc, ...)\n"
    "- DoiTuongApDung: chủ thể mà văn bản điều chỉnh (ví dụ: tổ chức tín dụng, "
    "ngân hàng thương mại, doanh nghiệp bảo hiểm, ...)\n"
    "- LinhVuc: lĩnh vực pháp lý (ví dụ: Tín dụng, Ngân hàng, Bảo hiểm, "
    "Chứng khoán, ...)\n\n"
    "QUY TẮC:\n"
    "- Mỗi thực thể PHẢI có \"evidence\" (trích đoạn ngắn ≤120 ký tự từ văn bản).\n"
    "- Nếu không tìm được thực thể có bằng chứng rõ ràng, BỎ QUA (không trả về).\n"
    "- Đề xuất metadata chỉ trả khi có bằng chứng trong văn bản.\n"
    "- Trả về JSON thuần theo schema; không kèm giải thích ngoài JSON."
)

USER_PROMPT_TEMPLATE = """Văn bản:
{title}

Số hiệu: {so_ky_hieu}
Loại văn bản: {loai_van_ban}

Metadata hiện có:
{metadata_block}

Nội dung văn bản (đã làm sạch HTML, có thể đã cắt bớt):
---
{content}
---

Hãy trả về JSON. Trong "entities", LIỆT KÊ tất cả thực thể tìm được — bao gồm cả
những giá trị đã có trong metadata (nhưng vẫn cần evidence). Trong
"metadata_suggestions", chỉ đề xuất giá trị cho các trường metadata ĐANG TRỐNG
trong bảng "Metadata hiện có" (key == "<trống>")."""


# ---------- Gemini plumbing ----------

def build_gemini_client():
    """Tạo google-genai client. Import trong hàm để báo lỗi rõ ràng."""
    from google import genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment/.env")
    return genai.Client(api_key=api_key)


def build_response_schema() -> dict:
    """Schema ép Gemini trả JSON đúng cấu trúc."""
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string"},
                        "entity_type": {
                            "type": "string",
                            "enum": ["CoQuan", "NguoiKy", "DoiTuongApDung", "LinhVuc"],
                        },
                        "evidence": {"type": "string"},
                        "confidence": {
                            "type": "number",
                        },
                    },
                    "required": ["entity", "entity_type", "evidence"],
                },
            },
            "metadata_suggestions": {
                "type": "object",
                "description": "key là tên trường metadata, value là giá trị đề xuất",
                "properties": {
                    "co_quan_ban_hanh": {"type": "string"},
                    "nguoi_ky": {"type": "string"},
                    "linh_vuc": {"type": "string"},
                    "pham_vi": {"type": "string"},
                    "nganh": {"type": "string"},
                    "chuc_danh": {"type": "string"},
                    "doi_tuong_ap_dung": {"type": "string"},
                },
            },
        },
        "required": ["entities"],
    }


def call_gemini_for_doc(client, model: str, doc: dict) -> dict:
    """Gọi Gemini cho 1 document. Raise nếu có lỗi — caller xử lý.

    Có retry/backoff cho 429 (rate-limit) và 5xx (server).
    """
    from google.genai import types
    from google.genai.errors import ClientError

    metadata_lines = []
    for k in ENRICHABLE_FIELDS + [DOI_TUONG_FIELD]:
        v = str(doc.get(k, "")).strip()
        label = f"key::{k}" if not v else f"key::{k}={v}"
        metadata_lines.append(label)
    metadata_block = "\n".join(metadata_lines)

    content = str(doc.get("content_clean", ""))
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + "\n\n[...đã cắt bớt...]"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=doc.get("title", ""),
        so_ky_hieu=doc.get("so_ky_hieu", ""),
        loai_van_ban=doc.get("loai_van_ban", ""),
        metadata_block=metadata_block,
        content=content,
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=build_response_schema(),
        temperature=0.0,
    )

    # Retry loop: 429 quota / 5xx / network
    last_err: Exception | None = None
    for attempt in range(4):  # 4 attempts: 0,1,2,3
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=config,
            )
            break
        except ClientError as e:
            last_err = e
            # 429 = RESOURCE_EXHAUSTED, 5xx = server. Retry với backoff.
            msg = str(e)
            transient = "RESOURCE_EXHAUSTED" in msg or "429" in msg or "500" in msg or "503" in msg
            if not transient or attempt == 3:
                raise
            wait_s = 60 * (attempt + 1)  # 60s, 120s, 180s
            print(f"      [retry {attempt+1}] {type(e).__name__}, sleeping {wait_s}s...", flush=True)
            time.sleep(wait_s)
    else:
        raise last_err or RuntimeError("exhausted retries")

    # 1) Empty response
    if not response or not response.text:
        raise RuntimeError("empty response from Gemini")

    # 2) Malformed JSON
    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"malformed JSON: {e}; raw={response.text[:200]}")

    # 3) Missing field
    if "entities" not in parsed or not isinstance(parsed["entities"], list):
        raise RuntimeError(f"missing 'entities' field; raw={str(parsed)[:200]}")

    return parsed


# ---------- Per-doc processing ----------

def derive_entities_from_metadata(doc: pd.Series) -> list[dict]:
    """Từ metadata gốc, sinh entity với method='metadata', confidence=1.0."""
    rows = []
    for col, etype in META_TO_ENTITY.items():
        v = str(doc.get(col, "")).strip()
        if not v or v == "Chưa phân loại":
            continue
        # Evidence: dùng title + so_ky_hieu (đây là thông tin từ chính văn bản)
        ev = f"metadata.{col} of id={doc['id']} (so_ky_hieu={doc['so_ky_hieu']})"
        rows.append({
            "entity": v,
            "entity_type": etype,
            "source": doc["id"],
            "method": "metadata",
            "confidence": 1.0,
            "evidence": ev,
        })
    return rows


def normalize_gemini_entity(item: dict, source_id: str) -> dict | None:
    """Validate một entity từ Gemini. Trả None nếu không hợp lệ."""
    if not isinstance(item, dict):
        return None
    entity = item.get("entity")
    etype = item.get("entity_type")
    evidence = item.get("evidence")
    if not isinstance(entity, str) or not entity.strip():
        return None
    if not isinstance(etype, str) or etype not in ALLOWED_TYPES:
        return None
    if not isinstance(evidence, str) or not evidence.strip():
        return None  # spec: no evidence → no entity
    conf = item.get("confidence", 0.7)
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.7
    conf = max(0.0, min(1.0, conf))
    return {
        "entity": entity.strip(),
        "entity_type": etype,
        "source": source_id,
        "method": "gemini",
        "confidence": conf,
        "evidence": evidence.strip()[:200],
    }


def process_doc(idx: int, doc: pd.Series, client, model: str) -> dict:
    """Xử lý 1 document. Trả về dict có keys:
      entities: list[dict]
      suggestions: dict
      error: str | None
    """
    result: dict = {"entities": [], "suggestions": {}, "error": None}
    try:
        # Bước A: lấy entities từ metadata gốc (ưu tiên)
        result["entities"] = derive_entities_from_metadata(doc)

        # Bước B: gọi Gemini
        parsed = call_gemini_for_doc(client, model, doc.to_dict())

        # Bước C: thêm entities từ Gemini (chỉ loại chưa có)
        existing_keys = {(e["entity_type"], e["entity"].lower()) for e in result["entities"]}
        for item in parsed.get("entities", []):
            norm = normalize_gemini_entity(item, doc["id"])
            if norm is None:
                continue
            key = (norm["entity_type"], norm["entity"].lower())
            if key in existing_keys:
                # Ưu tiên metadata gốc — nhưng vẫn ghi nhận evidence từ Gemini
                # (bỏ qua để tránh trùng)
                continue
            existing_keys.add(key)
            result["entities"].append(norm)

        # Bước D: thu thập metadata_suggestions
        suggestions = parsed.get("metadata_suggestions", {})
        if isinstance(suggestions, dict):
            for k, v in suggestions.items():
                if isinstance(v, str) and v.strip():
                    result["suggestions"][k] = v.strip()
    except Exception as e:  # noqa: BLE001 — intentional broad catch for per-doc isolation
        result["error"] = f"{type(e).__name__}: {e}"
    return result


# ---------- Main ----------

def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    banner("STEP 3 — GEMINI ENTITY EXTRACTION & METADATA ENRICHMENT")

    # Load .env
    if not load_dotenv(BASE / ".env"):
        print("WARN: .env not loaded (check files)")

    # Snapshot MD5 của metadata.csv và content.csv TRƯỚC khi chạy
    import hashlib
    pre_hashes = {}
    for name in ("metadata.csv", "content.csv"):
        p = BASE / name
        if p.exists():
            pre_hashes[name] = hashlib.md5(p.read_bytes()).hexdigest()

    # Load data
    df = pd.read_csv(INPUT_PATH, dtype=str, keep_default_na=False)
    print(f"Loaded {INPUT_PATH.name}: {len(df)} rows")

    # Gemini client + model
    try:
        client = build_gemini_client()
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}")
        return 1

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    print(f"Using model: {model}")

    # Per-doc processing
    all_entities: list[dict] = []
    enriched_rows: list[dict] = []
    errors: list[dict] = []
    success = 0
    failure = 0

    for idx, row in df.iterrows():
        doc_id = row["id"]
        print(f"  [{idx+1}/{len(df)}] id={doc_id} ({row['so_ky_hieu']}) ...", end="", flush=True)
        out = process_doc(idx, row, client, model)
        if out["error"]:
            failure += 1
            errors.append({"doc_id": doc_id, "so_ky_hieu": row["so_ky_hieu"], "error": out["error"]})
            print(f" FAIL ({out['error'][:80]})")
        else:
            success += 1
            all_entities.extend(out["entities"])

            # Build enriched metadata row
            erow = row.to_dict()
            for k in ENRICHABLE_FIELDS + [DOI_TUONG_FIELD]:
                original = str(erow.get(k, "")).strip()
                suggested = out["suggestions"].get(k, "")
                if (not original or original == "Chưa phân loại") and suggested:
                    erow[f"enriched_{k}"] = suggested
                    erow[f"method_{k}"] = "gemini"
                else:
                    erow[f"enriched_{k}"] = original
                    erow[f"method_{k}"] = "metadata" if original else "missing"
            enriched_rows.append(erow)
            print(f" OK ({len(out['entities'])} entities)")

        # Small delay để tránh rate-limit
        time.sleep(0.5)

    # ----- Save outputs -----
    banner("SAVING OUTPUTS")

    entities_df = pd.DataFrame(
        all_entities,
        columns=["entity", "entity_type", "source", "method", "confidence", "evidence"],
    )
    entities_df.to_csv(ENTITIES_OUT, index=False, encoding="utf-8")
    print(f"  {ENTITIES_OUT.name}: {len(entities_df)} rows x {entities_df.shape[1]} cols")

    enriched_df = pd.DataFrame(enriched_rows)
    enriched_df.to_csv(META_OUT, index=False, encoding="utf-8")
    print(f"  {META_OUT.name}: {len(enriched_df)} rows x {enriched_df.shape[1]} cols")

    # ----- Verify originals untouched -----
    banner("ORIGINAL FILES INTEGRITY CHECK")
    for name, expected in pre_hashes.items():
        p = BASE / name
        actual = hashlib.md5(p.read_bytes()).hexdigest()
        ok = "✓" if actual == expected else "✗ CHANGED"
        print(f"  {name}: {ok} (md5={actual})")

    # ----- Report -----
    banner("STEP 3 REPORT")
    print(f"  Successful documents : {success}")
    print(f"  Failed documents     : {failure}")
    print(f"  Total entities       : {len(entities_df)}")
    print()
    print(f"  Entities by type:")
    for t, n in entities_df["entity_type"].value_counts().items():
        print(f"    {t:20s} : {n}")
    print()
    print(f"  Entities by method:")
    for m, n in entities_df["method"].value_counts().items():
        print(f"    {m:20s} : {n}")

    # Đếm số giá trị metadata được làm giàu
    enriched_field_count = 0
    for k in ENRICHABLE_FIELDS + [DOI_TUONG_FIELD]:
        col = f"method_{k}"
        if col in enriched_df.columns:
            enriched_field_count += int((enriched_df[col] == "gemini").sum())
    print(f"\n  Metadata fields enriched by Gemini : {enriched_field_count}")

    # 5 ví dụ metadata gốc vs enriched (chỉ với field enrichable)
    print("\n  5 examples of original vs enriched metadata:")
    if len(enriched_df) > 0:
        cols_to_show = []
        for k in ENRICHABLE_FIELDS + [DOI_TUONG_FIELD]:
            if f"enriched_{k}" in enriched_df.columns:
                # Chỉ thêm cột enriched_/method_; cột gốc <k> có thể không tồn tại
                # (ví dụ doi_tuong_ap_dung không có sẵn trong metadata gốc)
                if k in enriched_df.columns:
                    cols_to_show.append(k)
                cols_to_show.extend([f"enriched_{k}", f"method_{k}"])
        sample = enriched_df[cols_to_show].head(5)
        for idx, r in sample.iterrows():
            print(f"\n    doc_id={enriched_df.iloc[idx]['id']} ({enriched_df.iloc[idx]['so_ky_hieu']})")
            for k in ENRICHABLE_FIELDS + [DOI_TUONG_FIELD]:
                orig = (r.get(k, "") or "") if k in enriched_df.columns else ""
                enr = (r.get(f"enriched_{k}", "") or "")
                col_method = r.get(f"method_{k}", "")
                if orig or enr:
                    print(f"      {k:25s}: original={orig[:35]!r:38s} | enriched={enr[:35]!r:38s} | method={col_method}")

    if errors:
        print("\n  ERRORS:")
        for e in errors:
            print(f"    - {e['so_ky_hieu']} (id={e['doc_id']}): {e['error']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
