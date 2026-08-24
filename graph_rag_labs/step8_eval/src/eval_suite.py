"""5 câu hỏi kiểm thử đại diện cho các tình huống tra cứu phức tạp.

Mỗi câu hỏi được thiết kế để bắt buộc multi-hop:
- Q1: thay thế + nội dung văn bản bị thay thế
- Q2: hợp nhất + nội dung cụ thể
- Q3: sửa đổi + nội dung sửa đổi
- Q4: căn cứ + chức năng của cơ quan
- Q5: hoạt động + sửa đổi bổ sung
"""
from __future__ import annotations

# Each question is a tuple (id, query, expected_answer_outline).
# The expected outline is what a "perfect" answer should contain
# given the underlying legal knowledge graph. Used for grading the
# automatic output, not as a hard constraint.
EVAL_QUESTIONS: list[tuple[str, str, str]] = [
    (
        "Q1",
        "Nghị định 46/2023/NĐ-CP thay thế cho nghị định nào, và nghị định bị thay thế đó có nội dung gì nổi bật về kinh doanh bảo hiểm?",
        "Nghị định 46/2023 thay thế Nghị định 90/2013. NĐ 90/2013 quy định chi tiết về kinh doanh bảo hiểm, tổ chức và hoạt động của doanh nghiệp bảo hiểm.",
    ),
    (
        "Q2",
        "Văn bản hợp nhất số 52/VBHN-NHNN được hợp nhất từ văn bản nào, và quy định về hồ sơ, thủ tục cấp giấy phép lần đầu của ngân hàng thương mại gồm những tài liệu gì?",
        "VBHN 52/2020 hợp nhất từ Thông tư 22/2014 và các văn bản sửa đổi. Hồ sơ cấp giấy phép lần đầu gồm: đơn đề nghị, đề án thành lập, phương án kinh doanh 5 năm, vốn điều lệ tối thiểu, v.v.",
    ),
    (
        "Q3",
        "Thông tư số 01/2025/TT-NHNN quy định về cấp giấy phép quỹ tín dụng nhân dân được sửa đổi, bổ sung bởi văn bản nào, và những nội dung sửa đổi bổ sung chính là gì?",
        "Thông tư 01/2025/TT-NHNN được sửa đổi bởi Thông tư 02/2025/TT-NHNN. Nội dung sửa đổi: điều kiện cấp giấy phép, mức vốn điều lệ tối thiểu, thủ tục thành lập.",
    ),
    (
        "Q4",
        "Thông tư số 41/2016/TT-NHNN về tỷ lệ an toàn vốn của ngân hàng căn cứ vào luật nào, và luật đó quy định chức năng nhiệm vụ của cơ quan nào?",
        "Thông tư 41/2016 căn cứ Luật Các tổ chức tín dụng 2010 (sửa đổi 2017). Luật này quy định chức năng NHNN là cơ quan quản lý nhà nước về tiền tệ, ngân hàng, hoạt động ngân hàng.",
    ),
    (
        "Q5",
        "Hoạt động giao nhận, vận chuyển tiền mặt và tài sản quý của Ngân hàng Nhà nước được điều chỉnh bởi Thông tư nào, và Thông tư đó có được sửa đổi bổ sung bởi văn bản nào không?",
        "Điều chỉnh bởi Thông tư 21/2012/TT-NHNN (và các phiên bản sửa đổi). Có thể sửa đổi bổ sung bởi các Thông tư sau nếu có.",
    ),
]