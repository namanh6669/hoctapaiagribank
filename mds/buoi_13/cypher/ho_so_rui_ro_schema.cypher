// Neo4j 5.x: schema MVP cho Wiki tri thức rủi ro.

CREATE CONSTRAINT van_ban_id IF NOT EXISTS FOR (node:VanBan) REQUIRE node.id IS UNIQUE;
CREATE CONSTRAINT dieu_khoan_id IF NOT EXISTS FOR (node:DieuKhoan) REQUIRE node.id IS UNIQUE;
CREATE CONSTRAINT quy_trinh_id IF NOT EXISTS FOR (node:QuyTrinh) REQUIRE node.id IS UNIQUE;
CREATE CONSTRAINT buoc_quy_trinh_id IF NOT EXISTS FOR (node:BuocQuyTrinh) REQUIRE node.id IS UNIQUE;
CREATE CONSTRAINT don_vi_id IF NOT EXISTS FOR (node:DonVi) REQUIRE node.id IS UNIQUE;
CREATE CONSTRAINT vai_tro_id IF NOT EXISTS FOR (node:VaiTro) REQUIRE node.id IS UNIQUE;
CREATE CONSTRAINT rui_ro_id IF NOT EXISTS FOR (node:RuiRo) REQUIRE node.id IS UNIQUE;
CREATE CONSTRAINT kiem_soat_id IF NOT EXISTS FOR (node:KiemSoat) REQUIRE node.id IS UNIQUE;
CREATE CONSTRAINT su_kien_rui_ro_id IF NOT EXISTS FOR (node:SuKienRuiRo) REQUIRE node.id IS UNIQUE;
CREATE CONSTRAINT bang_chung_id IF NOT EXISTS FOR (node:BangChung) REQUIRE node.id IS UNIQUE;

CREATE INDEX van_ban_status IF NOT EXISTS FOR (node:VanBan) ON (node.status);
CREATE INDEX rui_ro_category IF NOT EXISTS FOR (node:RuiRo) ON (node.category);

// Điều chỉnh dimensions cho đúng mô hình embedding đang dùng trước khi chạy.
CREATE VECTOR INDEX dieu_khoan_embedding IF NOT EXISTS
FOR (node:DieuKhoan) ON node.embedding
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}};

// Tìm điều khoản tương đồng, sau đó suy luận qua quan hệ nghiệp vụ đã xác minh.
CALL db.index.vector.queryNodes('dieu_khoan_embedding', $top_k, $query_embedding)
YIELD node AS dieu_khoan, score
MATCH (van_ban:VanBan)-[:CONTAINS]->(dieu_khoan)
WHERE van_ban.status <> 'HET_HIEU_LUC' OR $include_history = true
OPTIONAL MATCH (dieu_khoan)-[requirement:REQUIRES]->(kiem_soat:KiemSoat)
WHERE coalesce(requirement.verification_status, 'VERIFIED') = 'VERIFIED'
OPTIONAL MATCH (kiem_soat)-[mitigation:MITIGATES]->(rui_ro:RuiRo)
WHERE coalesce(mitigation.verification_status, 'VERIFIED') = 'VERIFIED'
RETURN dieu_khoan.id AS source_id, dieu_khoan.text AS excerpt, score,
       van_ban.id AS van_ban_id, van_ban.title AS van_ban,
       collect(DISTINCT kiem_soat.name) AS kiem_soat,
       collect(DISTINCT rui_ro.name) AS rui_ro
ORDER BY score DESC
LIMIT 20;

// Hàng chờ chuyên gia kiểm duyệt các liên kết do mô hình đề xuất.
MATCH (source)-[relationship]->(target)
WHERE relationship.verification_status = 'PROPOSED'
RETURN labels(source) AS loai_nguon, source.id AS id_nguon,
       type(relationship) AS quan_he, relationship.confidence AS confidence,
       relationship.source_chunk_id AS source_chunk_id,
       labels(target) AS loai_dich, target.id AS id_dich
ORDER BY confidence DESC;
