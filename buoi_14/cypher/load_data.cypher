// Buổi 14 — Mini KG data load (auto-generated)
// Source: buoi_14/data/processed/chunks_normalized.csv + ../ kb+hops/metadata.csv + ../ kb+hops/relationships.csv
// Thứ tự chạy: schema.cypher → load_data.cypher → demo_queries.cypher
// Mọi node/edge đều có lab_session = "buoi_14" để cô lập.

// Counts: VanBan=30 · DieuKhoan=1463 · Entity=63
//         CONTAINS=1463 · NEXT=1433
//         AP_DUNG_CHO=61
//         BAN_HANH_BOI=30
//         CONTAINS=1463
//         KY_BOI=30
//         NEXT=1433
//         SUA_DOI_BO_SUNG=9
//         THAM_CHIEU=10
//         THAY_THE_BOI=3
//         THUOC_LINH_VUC=23

// ============================================================
// (:VanBan) (30)
// ============================================================
UNWIND $vanbans AS v
MERGE (vb:VanBan {id: v.id})
SET vb.title            = v.title,
    vb.document_type    = v.document_type,
    vb.status           = v.status,
    vb.so_ky_hieu       = v.so_ky_hieu,
    vb.ngay_ban_hanh    = v.ngay_ban_hanh,
    vb.ngay_co_hieu_luc = v.ngay_co_hieu_luc,
    vb.ngay_het_hieu_luc= v.ngay_het_hieu_luc,
    vb.co_quan_ban_hanh = v.co_quan_ban_hanh,
    vb.nguoi_ky         = v.nguoi_ky,
    vb.nganh            = v.nganh,
    vb.linh_vuc         = v.linh_vuc,
    vb.lab_session      = 'buoi_14';

// ============================================================
// (:DieuKhoan) (1463)
// ============================================================
UNWIND $dieukhoans AS d
MERGE (dk:DieuKhoan {id: d.id})
SET dk.document_id   = d.document_id,
    dk.text          = d.text,
    dk.article       = d.article,
    dk.clause        = d.clause,
    dk.chapter       = d.chapter,
    dk.section       = d.section,
    dk.article_title = d.article_title,
    dk.so_ky_hieu    = d.so_ky_hieu,
    dk.ngay_ban_hanh = d.ngay_ban_hanh,
    dk.document_type = d.document_type,
    dk.title         = d.title,
    dk.status        = d.status,
    dk.lab_session   = 'buoi_14';

// ============================================================
// (:Entity) (63)
// ============================================================
UNWIND $entities AS e
MERGE (en:Entity {id: e.id})
SET en.name       = e.name,
    en.name_alt   = coalesce(e.name_alt, null),
    en.lab_session= 'buoi_14';

// ============================================================
// (:VanBan)-[:CONTAINS]->(:DieuKhoan) (1463)
// ============================================================
UNWIND $contains AS c
MATCH (vb:VanBan {id: c.src, lab_session: 'buoi_14'})
MATCH (dk:DieuKhoan {id: c.tgt, lab_session: 'buoi_14'})
MERGE (vb)-[r:CONTAINS]->(dk)
SET r.lab_session = 'buoi_14', r.method = 'chunks_normalized', r.confidence = 1.0, r.evidence = '', r.source = 'chunks_normalized.csv';

// ============================================================
// (:DieuKhoan)-[:NEXT]->(:DieuKhoan) (1433)
// chain trong cùng 1 VanBan theo thứ tự chunk trong chunks_normalized.csv
// ============================================================
UNWIND $nexts AS n
MATCH (a:DieuKhoan {id: n.src, lab_session: 'buoi_14'})
MATCH (b:DieuKhoan {id: n.tgt, lab_session: 'buoi_14'})
MERGE (a)-[r:NEXT]->(b)
SET r.lab_session = 'buoi_14', r.method = 'chunks_normalized', r.confidence = 1.0, r.evidence = '', r.source = 'chunks_normalized.csv (order)';

// ============================================================
// Edges từ relationships.csv
// ============================================================
// --- THAM_CHIEU (17) — VanBan → VanBan ---
UNWIND $rels_tham_chieu AS r
MATCH (s:VanBan {id: r.src, lab_session: 'buoi_14'})
MATCH (t:VanBan {id: r.tgt, lab_session: 'buoi_14'})
MERGE (s)-[rel:THAM_CHIEU]->(t)
SET rel.confidence = r.confidence,
    rel.method     = r.method,
    rel.evidence   = r.evidence,
    rel.source     = 'kb+hops/relationships.csv',
    rel.lab_session= 'buoi_14';

// --- THAY_THE_BOI (3) — VanBan → VanBan ---
UNWIND $rels_thay_the_boi AS r
MATCH (s:VanBan {id: r.src, lab_session: 'buoi_14'})
MATCH (t:VanBan {id: r.tgt, lab_session: 'buoi_14'})
MERGE (s)-[rel:THAY_THE_BOI]->(t)
SET rel.confidence = r.confidence,
    rel.method     = r.method,
    rel.evidence   = r.evidence,
    rel.source     = 'kb+hops/relationships.csv',
    rel.lab_session= 'buoi_14';

// --- SUA_DOI_BO_SUNG (9) — VanBan → VanBan ---
UNWIND $rels_sua_doi_bo_sung AS r
MATCH (s:VanBan {id: r.src, lab_session: 'buoi_14'})
MATCH (t:VanBan {id: r.tgt, lab_session: 'buoi_14'})
MERGE (s)-[rel:SUA_DOI_BO_SUNG]->(t)
SET rel.confidence = r.confidence,
    rel.method     = r.method,
    rel.evidence   = r.evidence,
    rel.source     = 'kb+hops/relationships.csv',
    rel.lab_session= 'buoi_14';

// --- BAN_HANH_BOI (30) — VanBan → Entity ---
UNWIND $rels_ban_hanh_boi AS r
MATCH (s:VanBan {id: r.src, lab_session: 'buoi_14'})
MATCH (t:Entity {id: r.tgt, lab_session: 'buoi_14'})
MERGE (s)-[rel:BAN_HANH_BOI]->(t)
SET rel.confidence = r.confidence,
    rel.method     = r.method,
    rel.evidence   = r.evidence,
    rel.source     = 'kb+hops/relationships.csv',
    rel.lab_session= 'buoi_14';

// --- KY_BOI (30) — VanBan → Entity ---
UNWIND $rels_ky_boi AS r
MATCH (s:VanBan {id: r.src, lab_session: 'buoi_14'})
MATCH (t:Entity {id: r.tgt, lab_session: 'buoi_14'})
MERGE (s)-[rel:KY_BOI]->(t)
SET rel.confidence = r.confidence,
    rel.method     = r.method,
    rel.evidence   = r.evidence,
    rel.source     = 'kb+hops/relationships.csv',
    rel.lab_session= 'buoi_14';

// --- THUOC_LINH_VUC (23) — VanBan → Entity ---
UNWIND $rels_thuoc_linh_vuc AS r
MATCH (s:VanBan {id: r.src, lab_session: 'buoi_14'})
MATCH (t:Entity {id: r.tgt, lab_session: 'buoi_14'})
MERGE (s)-[rel:THUOC_LINH_VUC]->(t)
SET rel.confidence = r.confidence,
    rel.method     = r.method,
    rel.evidence   = r.evidence,
    rel.source     = 'kb+hops/relationships.csv',
    rel.lab_session= 'buoi_14';

// --- AP_DUNG_CHO (61) — VanBan → Entity ---
UNWIND $rels_ap_dung_cho AS r
MATCH (s:VanBan {id: r.src, lab_session: 'buoi_14'})
MATCH (t:Entity {id: r.tgt, lab_session: 'buoi_14'})
MERGE (s)-[rel:AP_DUNG_CHO]->(t)
SET rel.confidence = r.confidence,
    rel.method     = r.method,
    rel.evidence   = r.evidence,
    rel.source     = 'kb+hops/relationships.csv',
    rel.lab_session= 'buoi_14';
