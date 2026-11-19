# Kiểm chứng mô phỏng với dữ liệu AMIT (giai đoạn 1, 17/11/2026)

Kế hoạch: `claude/ke_hoach_kiem_chung_truoc_quay_amit_20261117.md` trong project GAMA.

## Dữ liệu
AMIT-SI (NYCU Dataverse, doi:10.57770/2SFJ2G, CC BY-NC 4.0): 6 nút có đèn ở Đài Loan
(A01–A06), 21 recording, 10 Hz. Máy ảo không tải được trực tiếp từ dataverse (proxy chặn),
nên lần chạy này đọc zip trong trình duyệt và chạy **cùng mã Python** (Pyodide, kiểm tra
SHA-256 trùng với file trong thư mục này). Chỉ bảng tổng hợp được đưa về:
`results/amit_field_key_metrics.csv`.

Khi đã tải dữ liệu về `data/amit/A01 … A06`, chạy lại trên máy:

```bash
python3 validation/amit_site_summary.py data/amit/A01 --out validation/results/A01_field.json
```

## Mã
| File | Việc |
|---|---|
| `amit_events.py` | Một dòng mỗi xe từ file AMIT: cắt vạch dừng / vạch ra, xếp hàng, tốc độ, gia tốc ngang/dọc, điểm xung đột rẽ trái × đi thẳng đối diện |
| `amit_metrics.py` | Chỉ số: xả bão hòa theo pha xanh, chấp nhận khe, phân bố, GEH, Theil U |
| `amit_site_summary.py` | Gộp các recording của một nút → JSON; `key_metrics` cho bảng so sánh |
| `sim_to_amit.py` | Chạy `python_sim` và ghi kết quả đúng định dạng AMIT (10 Hz, vạch dừng/vạch ra, đèn) |
| `layer2_theory.py` | Lớp 2: xả bão hòa so với IDM và HCM; trễ so với Webster |
| `layer3_compare.py` | Lớp 3: bảng mô phỏng vs AMIT |

Dữ liệu thật và mô phỏng đi qua cùng `amit_events` → `amit_metrics` → `key_metrics`.

## Lặp lại các lần chạy
```bash
# mô phỏng (mỗi seed ~2,5 phút; 120 s khởi động + 900 s ghi)
python3 validation/sim_to_amit.py --out <dir>/hanoi_default --seeds 1 2 3 --ticks 4500 --warmup 120
python3 validation/sim_to_amit.py --out <dir>/amit_like --seeds 1 2 3 --ticks 4500 --warmup 120 \
    --sim-config validation/configs/amit_like_a01.json
python3 validation/amit_site_summary.py <dir>/amit_like --out <dir>/amit_like_summary.json
python3 validation/amit_site_summary.py <dir>/hanoi_default --out <dir>/hanoi_default_summary.json
python3 validation/layer3_compare.py --field validation/results/amit_field_key_metrics.csv \
    --sim amit_like=<dir>/amit_like_summary.json hanoi_default=<dir>/hanoi_default_summary.json \
    --match A01 --out validation/results/layer3_comparison
# lớp 2
python3 validation/layer2_theory.py discharge --seeds 1 2 --arms SE NW --out validation/results/layer2_discharge.json
python3 validation/layer2_theory.py delay --flows 300 1200 2100 --seeds 1 2 --out validation/results/layer2_delay.json
```
`amit_like_a01.json`: thành phần dòng xe, tỷ lệ rẽ và chu kỳ đèn của A01 (xe máy 68 %,
xe nặng 6 %, rẽ trái 11 %, rẽ phải 14 %, xanh 30/44 s gồm 4 s vàng, đỏ toàn phần 3 s,
1500 xe/h/nhánh) trên hình học LVL–HĐT. `hanoi_default`: cấu hình mặc định.

`validation/sim_runs/` chứa quỹ đạo mô phỏng lớn (~110 MB/seed), không đưa vào git.

## Kết quả
`results/layer3_comparison.md` (bảng đầy đủ), `results/layer2_*.json`,
`results/sim_*_summary.json`. Tóm tắt trong project GAMA:
`claude/ket_qua_kiem_chung_amit_giai_doan1_20261117.md`.
