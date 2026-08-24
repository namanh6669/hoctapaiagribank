// Buổi 14 — Demo queries trên Mini KG (VanBan / DieuKhoan)
//
// Tất cả tham số `$xxx` cần được set bằng :param trong Neo4j Browser hoặc
// truyền vào từ Python driver. Kết quả mẫu được sinh tự động từ
// `scripts/load_mini_kg.py` (xem outputs/kg_build_report.md) dựa trên
// in-memory NetworkX graph.


// ============================================================
// Q1. Đếm node theo label và edge theo type (chỉ lab buoi_14)
// ============================================================
MATCH (n)
WHERE n.lab_session = 'buoi_14'
RETURN labels(n) AS label, count(*) AS n
ORDER BY n DESC;

MATCH ()-[r]->()
WHERE r.lab_session = 'buoi_14'
RETURN type(r) AS type, count(*) AS n
ORDER BY n DESC;


// ============================================================
// Q2. Liệt kê VanBan theo loại văn bản (document_type) và trạng thái
//     Thay $doc_type bằng 'Thông tư' / 'Nghị định' / 'Luật' / 'Văn bản hợp nhất'.
//     Thay $status    bằng 'Còn hiệu lực' / 'Hết hiệu lực một phần' / 'Chưa xác định'.
// ============================================================
MATCH (v:VanBan {lab_session: 'buoi_14'})
WHERE v.document_type = $doc_type
  AND v.status        = $status
RETURN v.id            AS id,
       v.so_ky_hieu    AS so_ky_hieu,
       v.title         AS title,
       v.ngay_ban_hanh AS ngay_ban_hanh,
       v.co_quan_ban_hanh AS co_quan_ban_hanh
ORDER BY v.ngay_ban_hanh DESC;


// ============================================================
// Q3. Lấy toàn bộ Điều khoản của VanBan X (theo so_ky_hieu)
//     :NEXT đảm bảo trả về theo đúng thứ tự trong văn bản.
//     Thay $so_ky_hieu (vd '32/2024/QH15').
// ============================================================
MATCH (v:VanBan {so_ky_hieu: $so_ky_hieu, lab_session: 'buoi_14'})
MATCH path = (v)-[:CONTAINS]->(start:DieuKhoan)-[:NEXT*0..]->(d:DieuKhoan)
WHERE ALL(rel IN relationships(path) WHERE rel.lab_session = 'buoi_14')
RETURN d.article      AS article,
       d.chapter      AS chapter,
       left(d.text, 200) AS text_preview,
       d.id           AS chunk_id
ORDER BY length(path);  // đi qua chuỗi NEXT để đảm bảo thứ tự


// ============================================================
// Q4. Điều khoản cụ thể (article Y trong VanBan X)
//     Thay $so_ky_hieu (vd '01/2014/TT-NHNN') và $article (vd '5').
// ============================================================
MATCH (v:VanBan {so_ky_hieu: $so_ky_hieu, lab_session: 'buoi_14'})-[:CONTAINS]->(d:DieuKhoan)
WHERE d.article = $article
  AND d.lab_session = 'buoi_14'
RETURN d.id            AS chunk_id,
       v.so_ky_hieu    AS van_ban,
       d.article       AS article,
       d.text          AS full_text;


// ============================================================
// Q5. Văn bản nào tham chiếu / thay thế / sửa đổi VanBan X
//     Thay $so_ky_hieu.
// ============================================================
MATCH (v:VanBan {so_ky_hieu: $so_ky_hieu, lab_session: 'buoi_14'})
MATCH (v)-[r:THAM_CHIEU|THAY_THE_BOI|SUA_DOI_BO_SUNG]->(other:VanBan)
RETURN other.so_ky_hieu AS van_ban_lien_quan,
       other.title      AS title,
       type(r)          AS loai_quan_he,
       r.confidence     AS confidence,
       r.method         AS method
ORDER BY type(r), other.so_ky_hieu;


// ============================================================
// Q6. Đối tượng áp dụng của VanBan X
//     Thay $so_ky_hieu.
// ============================================================
MATCH (v:VanBan {so_ky_hieu: $so_ky_hieu, lab_session: 'buoi_14'})-[r:AP_DUNG_CHO]->(e:Entity)
RETURN e.name        AS doi_tuong_ap_dung,
       r.confidence  AS confidence,
       r.method      AS method
ORDER BY confidence DESC, e.name;


// ============================================================
// Q7. Lĩnh vực phổ biến (top 10 theo số VanBan tham chiếu)
// ============================================================
MATCH (e:Entity {lab_session: 'buoi_14'})<-[r:THUOC_LINH_VUC]-(:VanBan {lab_session: 'buoi_14'})
WITH e, count(*) AS so_van_ban
RETURN e.name AS linh_vuc, so_van_ban
ORDER BY so_van_ban DESC
LIMIT 10;


// ============================================================
// Q8. VanBan nào có nhiều Điều khoản (article) nhất?
// ============================================================
MATCH (v:VanBan {lab_session: 'buoi_14'})-[:CONTAINS]->(d:DieuKhoan)
WHERE d.article <> '' AND d.lab_session = 'buoi_14'
WITH v, count(d) AS so_dieu
RETURN v.so_ky_hieu AS so_ky_hieu,
       v.title      AS title,
       so_dieu
ORDER BY so_dieu DESC
LIMIT 10;


// ============================================================
// Q9. Đường đi giữa 2 VanBan (THAM_CHIEU / THAY_THE_BOI / SUA_DOI_BO_SUNG)
//     Thay $from_id, $to_id (UUID từ metadata.id).
// ============================================================
MATCH p = shortestPath(
  (a:VanBan {id: $from_id, lab_session: 'buoi_14'})
  -[:THAM_CHIEU|THAY_THE_BOI|SUA_DOI_BO_SUNG*..4]-
  (b:VanBan {id: $to_id,   lab_session: 'buoi_14'})
)
RETURN [n IN nodes(p) | n.so_ky_hieu] AS path,
       length(p)                       AS hops;


// ============================================================
// Q10. Đếm Orphan (Điều khoản không có VanBan cha, hoặc VanBan không CONTAINS Điều nào)
// ============================================================
MATCH (d:DieuKhoan {lab_session: 'buoi_14'})
WHERE NOT EXISTS { MATCH (:VanBan {lab_session:'buoi_14'})-[:CONTAINS]->(d) }
RETURN count(d) AS orphan_dieukhoan;

MATCH (v:VanBan {lab_session: 'buoi_14'})
WHERE NOT EXISTS { MATCH (v)-[:CONTAINS]->(:DieuKhoan {lab_session:'buoi_14'}) }
RETURN count(v) AS orphan_vanban;
