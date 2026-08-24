// =============================================================================
// Wiki Risk Graph — Demo queries (Cypher 5.x, tương thích Neo4j Browser)
// =============================================================================
// Dữ liệu sau khi load: 12 RuiRo, 10 KiemSoat, 12 SuKienRuiRo, 22 edge.
// Tất cả câu truy vấn đều dùng cú pháp chuẩn, chạy được trong Neo4j Browser,
// Cypher-shell, hoặc `neo4j.execute_query(...)` từ Python driver.
//
// Quy ước tham số: $risk_id, $control_id ... — khi gọi từ driver, truyền
// qua tham số thay vì string-interpolate (tránh Cypher injection).
// =============================================================================


// ----- A. Xem toàn bộ graph -----

// A1. Đếm node theo nhãn
MATCH (n)
RETURN labels(n)[0] AS label, count(*) AS so_node
ORDER BY so_node DESC;

// A2. Đếm edge theo loại quan hệ
MATCH ()-[r]->()
RETURN type(r) AS loai_quan_he, count(*) AS so_edge
ORDER BY so_edge DESC;

// A3. Một đoạn đồ thị nhỏ để quan sát (giới hạn 25 bộ ba)
MATCH (ks:KiemSoat)-[:MITIGATES]->(rr:RuiRo)-[:OBSERVED_AS]->(sk:SuKienRuiRo)
RETURN ks.id AS kiem_soat, rr.id AS rui_ro, sk.id AS su_kien
LIMIT 25;


// ----- B. Tìm kiểm soát giảm thiểu một rủi ro -----
// Tham số: $risk_id — ví dụ 'RR-006'

:param risk_id => 'RR-006';

MATCH (ks:KiemSoat)-[m:MITIGATES]->(rr:RuiRo {id: $risk_id})
RETURN ks.id             AS control_id,
       ks.name           AS control_name,
       ks.control_type   AS control_type,
       ks.frequency      AS frequency,
       ks.effectiveness  AS effectiveness,
       m.evidence_quote  AS evidence,
       m.verification_status AS verification_status
ORDER BY ks.id;


// ----- C. Tìm sự kiện của một rủi ro -----
// Tham số: $risk_id

:param risk_id => 'RR-006';

MATCH (rr:RuiRo {id: $risk_id})-[o:OBSERVED_AS]->(sk:SuKienRuiRo)
RETURN sk.id               AS su_kien_id,
       sk.occurred_at      AS ngay_xay_ra,
       sk.discovered_at    AS ngay_phat_hien,
       sk.severity         AS muc_do,
       sk.loss_amount_vnd  AS ton_that_vnd,
       sk.description      AS mo_ta,
       o.verification_status AS verification_status
ORDER BY sk.occurred_at DESC;


// ----- D. Đường đi KiemSoat -> RuiRo -> SuKienRuiRo -----

// D1. Tất cả đường đi (chỉ liệt kê)
MATCH path = (ks:KiemSoat)-[:MITIGATES]->(rr:RuiRo)-[:OBSERVED_AS]->(sk:SuKienRuiRo)
RETURN ks.id AS kiem_soat,
       rr.id AS rui_ro,
       sk.id AS su_kien,
       length(path) AS so_hop
ORDER BY rr.id, ks.id;

// D2. Truy vết từ một KiemSoat cụ thể (parameterized)
// Tham số: $control_id — ví dụ 'KS-006'

:param control_id => 'KS-006';

MATCH path = (ks:KiemSoat {id: $control_id})
              -[:MITIGATES]->(rr:RuiRo)
              -[:OBSERVED_AS]->(sk:SuKienRuiRo)
RETURN ks.id AS control,
       rr.id AS risk,
       sk.id AS event,
       [n IN nodes(path) | n.id] AS ids_tren_duong_di,
       [r IN relationships(path) | type(r)] AS cac_quan_he_tren_duong;


// ----- E. Rủi ro không có bất kỳ kiểm soát nào (dùng subquery EXISTS) -----

MATCH (rr:RuiRo)
WHERE NOT EXISTS {
  MATCH (:KiemSoat)-[:MITIGATES]->(rr)
}
RETURN rr.id              AS rui_ro_id,
       rr.name            AS ten,
       rr.category        AS phan_loai,
       rr.inherent_level  AS inherent,
       rr.residual_level  AS residual,
       rr.owner_unit_id   AS don_vi_so_huu_ma
ORDER BY rr.id;


// ----- F. Quan hệ chưa VERIFIED -----
// (Tìm mọi edge có verification_status khác 'VERIFIED' — nếu trả về 0 dòng
//  nghĩa là mọi quan hệ trong bộ dữ liệu đều ở trạng thái VERIFIED.)

MATCH (s)-[r]->(t)
WHERE r.verification_status <> 'VERIFIED'
RETURN labels(s)[0]    AS source_label,
       s.id            AS source_id,
       type(r)         AS quan_he,
       r.verification_status AS verification_status,
       r.data_origin   AS data_origin,
       r.confidence    AS confidence,
       labels(t)[0]    AS target_label,
       t.id            AS target_id
ORDER BY r.verification_status DESC, source_id;
