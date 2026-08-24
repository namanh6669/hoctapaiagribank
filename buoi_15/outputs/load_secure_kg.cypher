// Buổi 15 — Secure KG update (auto-generated)
// Source: buoi_15/data/processed/chunks_secure.csv (1463 chunks)
// MERGE by id; SET allowed_roles (List<String>) + lab_session = 'buoi_15'
// Counts: VanBan=30 · DieuKhoan=1463
//
// ⚠️  KHÔNG DETACH DELETE. KHÔNG string-interpolate.

// ───────────────────────────────────────────
// (:VanBan) — document-level access (INTERSECTION of chunk roles)
// ───────────────────────────────────────────
UNWIND $vanbans AS v
MERGE (vb:VanBan {id: v.id})
SET vb.allowed_roles = v.allowed_roles,
    vb.security_label = v.security_label,
    vb.title          = coalesce(v.title, vb.title, ''),
    vb.lab_session    = 'buoi_15';

// ───────────────────────────────────────────
// (:DieuKhoan) — chunk-level access (per-chunk)
// ───────────────────────────────────────────
UNWIND $dieukhoans AS d
MERGE (dk:DieuKhoan {id: d.id})
SET dk.allowed_roles = d.allowed_roles,
    dk.security_label = d.security_label,
    dk.document_id    = d.document_id,
    dk.lab_session    = 'buoi_15';

// ───────────────────────────────────────────
// Verification queries (run after the two MERGE blocks)
// ───────────────────────────────────────────
// 1) Count nodes carrying allowed_roles:
//    MATCH (n) WHERE n.allowed_roles IS NOT NULL
//    RETURN labels(n) AS l, count(n) AS c ORDER BY c DESC;
//
// 2) Sample 1 VanBan + linked DieuKhoan:
//    MATCH (vb:VanBan {lab_session:'buoi_15'})
//    WITH vb, rand() AS r ORDER BY r LIMIT 1
//    OPTIONAL MATCH (vb)-[:CONTAINS]->(dk:DieuKhoan)
//    RETURN vb {.*, allowed_roles: vb.allowed_roles} AS vanban,
//           collect(dk {.*, allowed_roles: dk.allowed_roles})[..3] AS dks;
