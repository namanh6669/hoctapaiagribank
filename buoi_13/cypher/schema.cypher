// =============================================================================
// Wiki Risk Graph — MVP Schema (Neo4j 5.x)
// =============================================================================
// Phạm vi: CHỈ 3 nhãn và 2 loại quan hệ (theo MVP của buổi 13).
// KHÔNG tạo các nhãn VanBan / DieuKhoan / QuyTrinh / DonVi / VaiTro / BangChung
// — chúng thuộc phần Graph RAG nâng cao, không nằm trong phạm vi MVP.
// =============================================================================


// ----- Ràng buộc duy nhất: dùng `id` làm khóa -----

CREATE CONSTRAINT rui_ro_id IF NOT EXISTS
  FOR (n:RuiRo) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT kiem_soat_id IF NOT EXISTS
  FOR (n:KiemSoat) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT su_kien_rui_ro_id IF NOT EXISTS
  FOR (n:SuKienRuiRo) REQUIRE n.id IS UNIQUE;


// ----- Index phụ trợ (chỉ trên các trường đang có dữ liệu) -----

CREATE INDEX rui_ro_category IF NOT EXISTS
  FOR (n:RuiRo) ON (n.category);

CREATE INDEX rui_ro_owner_unit_id IF NOT EXISTS
  FOR (n:RuiRo) ON (n.owner_unit_id);

CREATE INDEX kiem_soat_control_type IF NOT EXISTS
  FOR (n:KiemSoat) ON (n.control_type);

CREATE INDEX kiem_soat_effectiveness IF NOT EXISTS
  FOR (n:KiemSoat) ON (n.effectiveness);

CREATE INDEX su_kien_severity IF NOT EXISTS
  FOR (n:SuKienRuiRo) ON (n.severity);

CREATE INDEX su_kien_loss_amount IF NOT EXISTS
  FOR (n:SuKienRuiRo) ON (n.loss_amount_vnd);
