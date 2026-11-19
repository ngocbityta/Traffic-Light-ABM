# Tóm tắt kiểm tra hàm generate demand

Ngày chạy 2026-11-19, commit `186afe7`. Báo cáo đầy đủ: `demand_check_report.md`. Chạy lại: `uv run --no-project --python 3.12 --with "numpy<2" python validation/check_demand.py`.

## Kết luận

Hàm `generate_demand` trong `python-simulation/simulation/python_simulation/demand.py` (hàm train LVL dùng) chạy đúng: 53/53 kiểm tra PASS, 0 FAIL. Hàm GAMA trong `train/logic/demand.py` (site lvl) cũng đạt.

| Câu hỏi | Kết quả | Bằng chứng |
|---|---|---|
| Thông số sinh ra đúng chưa | Đạt | 20 seed × 8000 tick, khoảng 85 000 xe: xe máy 85,04 % (mong đợi 85), ô tô 11,90 % (12), xe nặng 3,06 % (3); trái 19,99 %, phải 19,95 %, thẳng 60,07 % (20/20/60); tốc độ khớp; số xe 84 989 (mong đợi 85 333). Mọi chỉ số lệch dưới 2,1 sigma, ngưỡng chấp nhận 5 sigma |
| Seed data chính xác chưa | Đạt | Hash trả về khớp sha256 của file; `read_demand` đọc lại không lỗi; simulator replay tạo đúng số xe bằng số dòng CSV (158, 164, 155) |
| Cùng seed ra cùng data không | Đạt | Hash trùng khi sinh lại trong cùng process, đổi `PYTHONHASHSEED` (0, 1, 12345), Python 3.10 / 3.11 / 3.12 / 3.14. Seed 1001 luôn ra `a2204feacd6b…` (4299 dòng). 4 seed × 2 hàm cho 8 hash khác nhau |

Kiểm chứng ngược: sinh cố ý với `motorbike_ratio=0.5` thì script báo FAIL (z khoảng 287), nên phép kiểm đủ nhạy để bắt lỗi lớn.

## Cách kiểm tra (ngắn)

1. Sinh nhiều file, đếm tỷ lệ và so với giá trị mong đợi; lệch quá 5 sigma là FAIL.
2. Đọc lại file và cho simulator chạy thử.
3. Sinh cùng seed nhiều lần, nhiều điều kiện, so mã sha256.

## Giới hạn cần biết

- Giá trị mong đợi lấy từ chính tham số đặt cứng trong code (2400 xe/giờ mỗi nhánh, 85 % xe máy, rẽ 20/20, xe nặng 3 %). Phép kiểm xác nhận hàm sinh đúng như code khai báo, **không** xác nhận các số đó đúng với ngã tư Lê Văn Lương thật. Chính code ghi các số này là giả định, chờ số đếm khảo sát.
- Quy tắc làn lấy từ `turn_lane_bounds`; chỉ kiểm xe không vào sai khoảng làn, chưa kiểm khoảng đó khớp vạch kẻ thực tế, chưa kiểm xe được bốc đều giữa các làn.
- Hai hàm mã hóa hướng rẽ ngược nhau (Python 0 = trái; GAMA 0 = thẳng), file của hai hàm không dùng chéo được. Quy ước GAMA lấy từ `collision_avoidance.gaml`; `LeVanLuong.gaml` mà `lvl_env.py` trỏ tới không có trong repo.
- Chưa kiểm cách `train.py` gán seed cho episode và worker.
- Chưa lấy commit của đồng nghiệp về (`git fetch`); kết quả áp dụng cho code hiện có trong repo.
- Tham số sinh không được lưu cùng seed trong `training_config.json`.
