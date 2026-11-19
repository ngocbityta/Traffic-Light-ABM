# Báo cáo kiểm tra hàm generate demand

Ngày chạy: 2026-11-19 · commit `186afe7` · Python 3.12.13 · lệnh: `uv run --no-project --python 3.12 --with "numpy<2" python validation/check_demand.py`

## 1. Kết luận

| Câu hỏi | `python_simulation.demand` (train LVL dùng) | `train/logic/demand.py` site=lvl (GAMA) |
|---|---|---|
| Thông số sinh ra đúng chưa | ĐẠT | ĐẠT |
| Seed data chính xác chưa (hash, đọc lại, replay) | ĐẠT | ĐẠT |
| Cùng seed ra cùng data không (cùng process, khác process, khác `PYTHONHASHSEED`, khác Python) | ĐẠT | ĐẠT |

Dung sai thống kê: 5 sigma. 20 seed (1001..1020), horizon 8000.

## 2. Tham số sinh

- Python: flow 2400 xe/h mỗi nhánh, DT 0.2 s ⇒ p = 0.1333 mỗi tick mỗi nhánh; xe máy 0.85, xe nặng 0.03, ô tô 0.12; trái 0.2, phải 0.2; mã rẽ 0=trái 1=phải 2=thẳng; cycle bắt đầu từ 1.
- GAMA (site lvl): p = 0.0667 mỗi bước 0,1 s mỗi nhánh (cùng 2400 xe/h); xe máy 0.85 / ô tô 0.15; mã rẽ 0=thẳng 1=trái 2=phải (theo `collision_avoidance.gaml:481`); cycle bắt đầu từ 0; làn luôn 0.

### Giải nghĩa các thuật ngữ

| Từ | Nghĩa | Giá trị mặc định |
|---|---|---|
| `DT` | Bước thời gian của simulator Python: mỗi tick mô phỏng trôi 0,2 giây | 0,2 s |
| tick / `cycle` | Số thứ tự bước mô phỏng, **không phải** chu kỳ đèn. Cột đầu của CSV. Tick n ứng với giây n × DT | Python từ 1; GAMA từ 0 |
| `horizon` | Số tick sinh demand. 8000 tick × 0,2 s = 1600 s (khoảng 27 phút) | 8000 trong lần kiểm này |
| `seed` | Số nguyên đưa vào `random.Random(seed)`. Cùng seed ⇒ cùng dãy số ngẫu nhiên ⇒ cùng file CSV | 1001..1020 (thống kê) |
| nhánh / `arm` | Hướng vào ngã tư: NE, SE, SW, NW. Trong CSV là số 0..3 theo thứ tự đó | 4 nhánh |
| `flow_per_approach` | Lưu lượng đến mỗi nhánh, xe/giờ | 2400 |
| xác suất đến `p` | Mỗi tick, mỗi nhánh tung một lần: có xe mới với xác suất p = flow / 3600 × DT. 2400/3600 × 0,2 = 0,1333, trung bình 1 xe mỗi 7,5 tick (1,5 s) | 0,1333 (GAMA: 0,0667 vì tick 0,1 s) |
| Bernoulli | Phép thử có/không với xác suất p. Số xe sau N phép thử tuân theo phân phối nhị thức Binomial(N, p) | |
| `motorbike_ratio` | Tỷ lệ xe máy trong số xe sinh ra | 0,85 |
| `heavy_ratio` | Tỷ lệ xe nặng (xe tải, xe buýt) | 0,03 |
| ô tô | Phần còn lại: 1 − 0,85 − 0,03 | 0,12 |
| `p_left`, `p_right` | Xác suất mỗi xe rẽ trái, rẽ phải; còn lại đi thẳng | 0,2 và 0,2 (thẳng 0,6) |
| mã loại xe | Cột 3 của CSV: 0 xe máy, 1 ô tô, 2 xe nặng | |
| mã hướng rẽ | Cột 5 của CSV. Python: 0 trái, 1 phải, 2 thẳng. GAMA: 0 thẳng, 1 trái, 2 phải | |
| tốc độ | Cột 4 của CSV, km/h, bốc từ phân phối chuẩn gauss(μ, σ), tối thiểu 1 km/h | xe máy 40 ± 5, ô tô 50 ± 5, xe nặng 50 ± 2,5 |
| làn / sublane | Cột 6 của CSV: chỉ số làn con tính từ lề phải, mỗi làn con rộng 1,05 m. Xe máy chiếm 1, ô tô 2, xe nặng 3 làn con. Làn bốc ngẫu nhiên trong giới hạn của hướng rẽ | |
| sigma, z | Độ lệch giữa số đo và mong đợi, tính theo đơn vị độ lệch chuẩn của sai số ngẫu nhiên. |z| ≤ 5 là chấp nhận | 5 |
| sha256, hash | Chuỗi 64 ký tự tóm tắt nội dung file. Chỉ cần sai một byte là hash khác hoàn toàn | |
| `PYTHONHASHSEED` | Biến môi trường làm thứ tự duyệt set/dict của chuỗi thay đổi giữa các lần chạy Python. Dùng để chứng minh kết quả không phụ thuộc thứ tự đó | 0, 1, 12345 |

## 3. Tiêu chí so sánh và kiểm tra

Giá trị mong đợi lấy từ tham số trong mã của từng hàm (mục 2), không lấy từ kết quả đo. PASS = đạt tiêu chí của dòng đó; chỉ cần một dòng FAIL thì cả nhóm là KHÔNG ĐẠT.

| Nhóm | Cách so sánh | Tiêu chí PASS |
|---|---|---|
| Số xe tổng, số xe mỗi nhánh | Mỗi (seed, tick, nhánh) là một phép thử Bernoulli xác suất p. Số xe n ~ Binomial(N, p). z = (n − N·p) / √(N·p·(1−p)) | \|z\| ≤ 5 (mỗi nhánh lấy nhánh lệch nhất) |
| Tỷ lệ loại xe, tỷ lệ hướng rẽ | Với n xe sinh ra, số xe mỗi loại ~ Binomial(n, p loại đó). Cùng công thức z | \|z\| ≤ 5; không được có loại xe ngoài danh sách |
| Tốc độ trung bình | z = (trung bình đo − μ) / (σ/√m), m = số xe loại đó | \|z\| ≤ 5 |
| Độ lệch chuẩn tốc độ | z = (s/σ − 1) · √(2(m−1)) (xấp xỉ phân phối của độ lệch chuẩn mẫu) | \|z\| ≤ 5 |
| Tốc độ nhỏ nhất | So với sàn của hàm sinh (Python ≥ 1 km/h) | Không dòng nào dưới sàn |
| Cấu trúc file | Đếm dòng vi phạm trên mọi file: 6 cột, cycle trong khoảng cho phép, nhánh 0..3, tối đa 1 xe mỗi (cycle, nhánh), sắp theo (cycle, nhánh) | 0 dòng vi phạm |
| Làn | Mỗi dòng so với `turn_lane_bounds` (rẽ) hoặc 0..num_lanes − số làn xe chiếm (thẳng); bản GAMA phải bằng 0 | 0 dòng ngoài giới hạn |
| Hash = file | sha256 của file đọc lại so với giá trị hàm trả về | Trùng cho mọi seed |
| `read_demand` | Đọc lại từng file bằng bộ đọc/kiểm của simulator Python | Không ném lỗi |
| Replay | `LeVanLuongSimulator(demand_file=...)` chạy `horizon` tick, so `sim.created` với số dòng CSV | Bằng nhau |
| Tất định | So `sha256` và số dòng của 4 seed cố định × 2 hàm (horizon 8000) giữa: sinh lần 2 trong cùng process, subprocess `PYTHONHASHSEED` 0 / 1 / 12345, Python 3.10 / 3.11 / 3.14 | Mọi hash trùng hoàn toàn (so từng byte qua sha256), không dùng dung sai |
| Khác seed | Đếm số hash khác nhau trong 8 hash (4 seed × 2 hàm) | Mọi hash khác nhau |

Chọn 5 sigma để giảm báo động giả: có khoảng 25 phép so sánh thống kê mỗi lần chạy, ngưỡng 5 sigma cho xác suất báo nhầm cỡ 1 phần 1,7 triệu mỗi phép. Đổi lại, với cỡ mẫu 20 seed (khoảng 85 000 xe), tỷ lệ lệch dưới khoảng 0,6 đến 0,7 điểm % có thể không bị phát hiện. Kiểm chứng độ nhạy: chạy `--negative-control` (sinh với `motorbike_ratio=0.5`) phải ra FAIL.

Mục 'Mong đợi' trong bảng kết quả: tỷ lệ hiển thị theo p; số xe hiển thị là N·p; tốc độ hiển thị là μ hoặc σ của `DEFAULT_VEHICLE_PARAMS` (Python) hoặc của mã GAMA. Cột z để trống ở các kiểm tra đếm (đạt khi bằng 0).

## 4. Kết quả chi tiết

### `python_simulation.demand` (Python)

**Thông số**

| Kiểm tra | Mong đợi | Đo được | z | Kết quả |
|---|---|---|---|---|
| số xe tổng (20 seed × 8000 tick × 4 nhánh) | 85333.3 | 84989 | -1.27 | PASS |
| số xe mỗi nhánh (độ lệch lớn nhất) (z từng nhánh: {'NE': 0.66, 'SE': -1.75, 'SW': -0.78, 'NW': -0.66}) | 21333.3 | {1: 21095, 3: 21244, 2: 21227, 0: 21423} | -1.75 | PASS |
| tỷ lệ loại xe: xe máy | 0.850 | 0.8504 | 0.32 | PASS |
| tỷ lệ loại xe: ô tô | 0.120 | 0.1190 | -0.91 | PASS |
| tỷ lệ loại xe: xe nặng | 0.030 | 0.0306 | 1.07 | PASS |
| không có loại xe ngoài danh sách | {} | [] |  | PASS |
| tỷ lệ hướng rẽ: trái (mã 0) | 0.200 | 0.1999 | -0.1 | PASS |
| tỷ lệ hướng rẽ: phải (mã 1) | 0.200 | 0.1995 | -0.39 | PASS |
| tỷ lệ hướng rẽ: thẳng (mã 2) | 0.600 | 0.6007 | 0.4 | PASS |
| tốc độ trung bình xe máy (km/h) | 40.0 | 39.991 | -0.48 | PASS |
| độ lệch chuẩn tốc độ xe máy (km/h) | 5.0 | 5.018 | 1.37 | PASS |
| tốc độ trung bình ô tô (km/h) | 50.0 | 49.996 | -0.08 | PASS |
| độ lệch chuẩn tốc độ ô tô (km/h) | 5.0 | 4.993 | -0.19 | PASS |
| tốc độ trung bình xe nặng (km/h) | 50.0 | 50.028 | 0.56 | PASS |
| độ lệch chuẩn tốc độ xe nặng (km/h) | 2.5 | 2.49 | -0.3 | PASS |
| tốc độ nhỏ nhất | >= 1 km/h | 17.909 |  | PASS |

**Cấu trúc file**

| Kiểm tra | Mong đợi | Đo được | z | Kết quả |
|---|---|---|---|---|
| 6 cột mỗi dòng | 0 | 0 |  | PASS |
| cycle trong 1..8000 | 0 | 0 |  | PASS |
| nhánh trong 0..3 | 0 | 0 |  | PASS |
| tối đa 1 xe mỗi (cycle, nhánh) | 0 | 0 |  | PASS |
| sắp theo (cycle, nhánh) | 0 | 0 |  | PASS |
| làn nằm trong giới hạn của hướng rẽ và loại xe (36 tổ hợp (nhánh, loại, hướng) đã gặp) | 0 | 0 |  | PASS |

**Seed data**

| Kiểm tra | Mong đợi | Đo được | z | Kết quả |
|---|---|---|---|---|
| hash hàm trả về = sha256 của file | 0 | 0 |  | PASS |
| read_demand đọc lại không lỗi | 0 | 0 |  | PASS |
| replay (horizon 300): số xe simulator tạo = số dòng CSV | {12: 158, 1001: 164, 1002: 155} | {12: 158, 1001: 164, 1002: 155} |  | PASS |

### `train/logic/demand.py` (GAMA)

**Thông số**

| Kiểm tra | Mong đợi | Đo được | z | Kết quả |
|---|---|---|---|---|
| số xe tổng (20 seed × 8000 tick × 4 nhánh) | 42666.7 | 42670 | 0.02 | PASS |
| số xe mỗi nhánh (độ lệch lớn nhất) (z từng nhánh: {0: 2.11, 1: -1.63, 2: -0.09, 3: -0.36}) | 10666.7 | {1: 10504, 2: 10658, 3: 10631, 0: 10877} | 2.11 | PASS |
| tỷ lệ loại xe: xe máy | 0.850 | 0.8516 | 0.9 | PASS |
| tỷ lệ loại xe: ô tô | 0.150 | 0.1484 | -0.9 | PASS |
| không có loại xe ngoài danh sách | {} | [] |  | PASS |
| tỷ lệ hướng rẽ: thẳng (mã 0) | 0.600 | 0.5992 | -0.33 | PASS |
| tỷ lệ hướng rẽ: trái (mã 1) | 0.200 | 0.1998 | -0.08 | PASS |
| tỷ lệ hướng rẽ: phải (mã 2) | 0.200 | 0.2009 | 0.48 | PASS |
| tốc độ trung bình xe máy (km/h) | 40.0 | 40.039 | 1.5 | PASS |
| độ lệch chuẩn tốc độ xe máy (km/h) | 5.0 | 5.014 | 0.76 | PASS |
| tốc độ trung bình ô tô (km/h) | 50.0 | 50.008 | 0.13 | PASS |
| độ lệch chuẩn tốc độ ô tô (km/h) | 5.0 | 4.968 | -0.71 | PASS |
| tốc độ nhỏ nhất | >= 1 km/h | 20.8 |  | PASS |

**Cấu trúc file**

| Kiểm tra | Mong đợi | Đo được | z | Kết quả |
|---|---|---|---|---|
| 6 cột mỗi dòng | 0 | 0 |  | PASS |
| cycle trong 0..7999 | 0 | 0 |  | PASS |
| nhánh trong 0..3 | 0 | 0 |  | PASS |
| tối đa 1 xe mỗi (cycle, nhánh) | 0 | 0 |  | PASS |
| sắp theo (cycle, nhánh) | 0 | 0 |  | PASS |
| làn luôn bằng 0 (GAMA tự chọn làn) | 0 | 0 |  | PASS |

**Seed data**

| Kiểm tra | Mong đợi | Đo được | z | Kết quả |
|---|---|---|---|---|
| hash hàm trả về = sha256 của file | 0 | 0 |  | PASS |
| file GAMA qua read_demand của Python (thông tin) (không dùng chéo) | - | bị từ chối: Demand line 1: invalid cycle, arm, or vehicle type |  | INFO |

### Tính tất định

| Kiểm tra | Mong đợi | Đo được | z | Kết quả |
|---|---|---|---|---|
| cùng seed, sinh 2 lần trong 1 process: cùng hash | trùng | trùng |  | PASS |
| so hash: subprocess PYTHONHASHSEED=0 (Python 3.12.13) | trùng | trùng |  | PASS |
| so hash: subprocess PYTHONHASHSEED=1 (Python 3.12.13) | trùng | trùng |  | PASS |
| so hash: subprocess PYTHONHASHSEED=12345 (Python 3.12.13) | trùng | trùng |  | PASS |
| so hash: Python 3.10 (uv) (Python 3.10.20) | trùng | trùng |  | PASS |
| so hash: Python 3.11 (uv) (Python 3.11.15) | trùng | trùng |  | PASS |
| so hash: Python 3.14 (uv) (Python 3.14.9) | trùng | trùng |  | PASS |
| khác seed thì khác hash | 8 hash khác nhau | 8 hash khác nhau |  | PASS |

**Hash theo từng lần chạy** (sha256, số dòng; horizon 8000)

| Lần chạy | python/1001 | python/1002 | python/10000001 | python/20000000 | gama/1001 | gama/1002 | gama/10000001 | gama/20000000 |
|---|---|---|---|---|---|---|---|---|
| process chính (Python 3.12.13) | `a2204feacd6b` (4299) | `58cdf7285f8b` (4246) | `b7327108279a` (4246) | `49b200b6a43b` (4361) | `447b8bb0d679` (2164) | `f7937b43b820` (2061) | `5176153fc07f` (2121) | `5b9692ffc48c` (2084) |
| subprocess PYTHONHASHSEED=0 (Python 3.12.13) | `a2204feacd6b` (4299) | `58cdf7285f8b` (4246) | `b7327108279a` (4246) | `49b200b6a43b` (4361) | `447b8bb0d679` (2164) | `f7937b43b820` (2061) | `5176153fc07f` (2121) | `5b9692ffc48c` (2084) |
| subprocess PYTHONHASHSEED=1 (Python 3.12.13) | `a2204feacd6b` (4299) | `58cdf7285f8b` (4246) | `b7327108279a` (4246) | `49b200b6a43b` (4361) | `447b8bb0d679` (2164) | `f7937b43b820` (2061) | `5176153fc07f` (2121) | `5b9692ffc48c` (2084) |
| subprocess PYTHONHASHSEED=12345 (Python 3.12.13) | `a2204feacd6b` (4299) | `58cdf7285f8b` (4246) | `b7327108279a` (4246) | `49b200b6a43b` (4361) | `447b8bb0d679` (2164) | `f7937b43b820` (2061) | `5176153fc07f` (2121) | `5b9692ffc48c` (2084) |
| Python 3.10 (uv) (Python 3.10.20) | `a2204feacd6b` (4299) | `58cdf7285f8b` (4246) | `b7327108279a` (4246) | `49b200b6a43b` (4361) | `447b8bb0d679` (2164) | `f7937b43b820` (2061) | `5176153fc07f` (2121) | `5b9692ffc48c` (2084) |
| Python 3.11 (uv) (Python 3.11.15) | `a2204feacd6b` (4299) | `58cdf7285f8b` (4246) | `b7327108279a` (4246) | `49b200b6a43b` (4361) | `447b8bb0d679` (2164) | `f7937b43b820` (2061) | `5176153fc07f` (2121) | `5b9692ffc48c` (2084) |
| Python 3.14 (uv) (Python 3.14.9) | `a2204feacd6b` (4299) | `58cdf7285f8b` (4246) | `b7327108279a` (4246) | `49b200b6a43b` (4361) | `447b8bb0d679` (2164) | `f7937b43b820` (2061) | `5176153fc07f` (2121) | `5b9692ffc48c` (2084) |

## 5. Ghi chú và điểm chưa kiểm

- Hai hàm mã hóa hướng rẽ khác nhau (Python 0=trái, GAMA 0=thẳng). File sinh từ hàm GAMA không đưa vào simulator Python.
- `train/logic/lvl_env.py:18` trỏ tới `gama-simulation/simulation/models/LeVanLuong.gaml`, file này không có trong repo; quy ước mã rẽ của GAMA được lấy từ `train/benchmarks/gama/includes/collision_avoidance.gaml`, chưa đối chiếu với model LVL.
- Ánh xạ nhánh của bản GAMA (slot 0..3 ⇒ E,N,W,S theo chú thích; model đọc N,S,E,W) chưa kiểm chứng; kiểm tra này chỉ xác nhận số xe mỗi nhánh đều nhau.
- Tham số sinh (flow, tỷ lệ, p_left/p_right) không được lưu cùng seed trong `training_config.json`; report này kiểm giá trị mặc định đang có trong mã.
- Không kiểm cách `train.py` gán seed cho episode / worker (seed_base + ep, validation seed); chỉ kiểm hàm sinh demand khi đã có seed.
- Replay vào simulator chỉ thử với horizon 300 và 3 seed (chạy 8000 tick × nhiều seed rất chậm).
- Kiểm tra thống kê chỉ phát hiện lệch lớn hơn dung sai; không chứng minh phân phối đúng từng chi tiết.
- Hash chỉ chứng minh cùng kết quả trên các phiên bản Python đã chạy (xem bảng hash).

## 6. Chạy lại

```bash
uv run --no-project --python 3.12 --with "numpy<2" python validation/check_demand.py
uv run --no-project --python 3.12 --with "numpy<2" python validation/check_demand.py --negative-control  # phải ra FAIL
```
