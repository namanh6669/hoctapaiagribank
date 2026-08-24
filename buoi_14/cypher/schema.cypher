// Buổi 14 — Mini Knowledge Graph schema  (ontology VanBan / DieuKhoan)
// Chạy Neo4j 5.x, 6.x. KHÔNG xoá graph hiện có. Mọi node/edge của buổi này
// mang `lab_session = "buoi_14"` để phân biệt với các lab khác.

// ============================================================
// Constraints (idempotent)
// ============================================================
CREATE CONSTRAINT vanban_id IF NOT EXISTS
  FOR (v:VanBan) REQUIRE v.id IS UNIQUE;

CREATE CONSTRAINT dieukhoan_id IF NOT EXISTS
  FOR (d:DieuKhoan) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT entity_id IF NOT EXISTS
  FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// ============================================================
// Indexes (hỗ trợ truy vấn theo so_ky_hieu / article)
// ============================================================
CREATE INDEX vanban_so_ky_hieu IF NOT EXISTS
  FOR (v:VanBan) ON (v.so_ky_hieu);

CREATE INDEX vanban_doc_type IF NOT EXISTS
  FOR (v:VanBan) ON (v.document_type);

CREATE INDEX dieukhoan_doc_id IF NOT EXISTS
  FOR (d:DieuKhoan) ON (d.document_id);

CREATE INDEX dieukhoan_article IF NOT EXISTS
  FOR (d:DieuKhoan) ON (d.article);

CREATE INDEX entity_name IF NOT EXISTS
  FOR (e:Entity) ON (e.name);

// Index để phạm vi cleanup nếu cần — KHÔNG dùng để xoá graph hiện có
CREATE INDEX any_lab_session IF NOT EXISTS
  FOR (n) ON (n.lab_session);

CREATE INDEX any_lab_session_rel IF NOT EXISTS
  FOR ()-[r]->() ON (r.lab_session);

// ============================================================
// Node labels & thuộc tính (tham khảo)
// ============================================================
//
// :VanBan — văn bản pháp luật
//   id            string  (UUID/metadata.id)
//   title         string  (metadata.title)
//   document_type string  (metadata.loai_van_ban: Luật / Nghị định / Thông tư / VBHN)
//   status        string  (metadata.tinh_trang_hieu_luc)
//   so_ky_hieu    string  (vd "32/2024/QH15")
//   ngay_ban_hanh string  (metadata.ngay_ban_hanh)
//   ngay_co_hieu_luc, ngay_het_hieu_luc, co_quan_ban_hanh, nguoi_ky, nganh, linh_vuc
//   lab_session   string  -- "buoi_14"  (BẮT BUỘC để phân biệt các lab)
//
// :DieuKhoan — đoạn nội dung (chunk) của VanBan
//   id            string   (chunk_id UUID)
//   document_id   string   (metadata.id, FK sang VanBan.id)
//   text          string   (chunk content; có thể rất dài — nên giữ text đầy đủ)
//   article       string   (số điều vd "5"; trống cho preamble)
//   clause        string   (số khoản vd "3"; trống khi không tách được)
//   chapter       string   (số chương vd "II"; trống khi không có)
//   section       string   (mục; trống)
//   so_ky_hieu    string   (propagate từ VanBan)
//   ngay_ban_hanh string   (propagate để dễ truy vấn)
//   lab_session   string   -- "buoi_14"
//
// :Entity — chủ thể (cơ quan / người ký / lĩnh vực / đối tượng áp dụng)
//   id            string   (E0001..E0076 từ relationships.target_id)
//   name          string   (relationships.target_name)
//   name_alt      string?  (các biến thể tên khác nếu có)
//   lab_session   string   -- "buoi_14"
//
// ============================================================
// Relationship types (mapping có kiểm soát từ dữ liệu thực)
// ============================================================
//
// BẮT BUỘC:
//   (:VanBan)-[:CONTAINS]->(:DieuKhoan)
//       — mỗi Điều khoản (chunk) thuộc về 1 văn bản
//
// CẤU TRÚC:
//   (:DieuKhoan)-[:NEXT]->(:DieuKhoan)
//       — chuỗi tuần tự giữa các Điều TRONG CÙNG 1 văn bản, theo thứ tự trong
//         chunks_normalized.csv (preamble trước, rồi Điều 1, Điều 2, ...)
//       — chỉ sinh giữa những cặp Điều có article (số điều) — KHÔNG nối preamble
//
// TỪ relationships.csv (Document↔Document):
//   (:VanBan)-[:THAM_CHIEU]->(:VanBan)
//   (:VanBan)-[:THAY_THE_BOI]->(:VanBan)
//   (:VanBan)-[:SUA_DOI_BO_SUNG]->(:VanBan)
//   — dùng cùng loại với hướng đi trong relationships.csv; KHÔNG tạo relation
//     mới ngoài 3 loại này.
//
// TỪ relationships.csv (Document→Entity):
//   (:VanBan)-[:BAN_HANH_BOI]->(:Entity)
//   (:VanBan)-[:KY_BOI]->(:Entity)
//   (:VanBan)-[:THUOC_LINH_VUC]->(:Entity)
//   (:VanBan)-[:AP_DUNG_CHO]->(:Entity)
//
// Mọi edge đều có:
//   confidence : float   (rule=0.9/1.0, metadata=1.0, gemini=0.75)
//   method     : string  (rule / metadata / gemini)
//   evidence   : string  (snippet trích dẫn, có thể cắt ngắn)
//   lab_session: string  -- "buoi_14" (BẮT BUỘC)
//   source     : string  -- "kb+hops/relationships.csv" hoặc "chunks_normalized.csv"
