# Đánh giá phần chuyển mô phỏng từ GAMA sang Python

Ngày: 10/11/2026 · Bản đánh giá lần 2, trên `main` tại commit `9fe6a86`.

**Phạm vi:** chỉ đánh giá việc **bản Python có tái hiện đúng mô hình GAMA hay không**, và có thay được GAMA khi train/đánh giá RL hay không. Không đánh giá mô hình GAMA gốc đúng hay sai so với giao thông thực.

| File | Port từ |
|---|---|
| `LeVanLuongIntersection/python_sim.py` | `LeVanLuongIntersection/models/LeVanLuong.gaml` (driving skill GAMA 2025.6.4) |
| `LeVanLuongIntersection/demand.py` | reflex `spawn` của GAML, đổi sang lịch xe CSV dùng chung |
| `dqn_py/lvl_env.py` | experiment `dqn_server_experiment`: state/reward dựng bằng Python cho cả hai backend |
| `dqn_py/python_env.py` | `Traffic/models/intersection-dqn-python.gaml` (bài toán legacy) |
| `scripts/compare_lvl_python_gama.py` | Đối chiếu GAMA ↔ Python |
| `scripts/run_vehicle_behavior_suite.py` (mới) | 30 test hành vi xe trên bản Python |

**Cách làm:** đọc diff `8bda5c5..9fe6a86`, so lại từng hàm với GAML, chạy toàn bộ test và một episode 8.000 tick. Máy này **không có GAMA headless**, nên tôi không chạy lại được các phép đối chiếu với GAMA.

---

## 1. Kết luận

So với lần đánh giá trước, **nhóm đã sửa phần lớn lỗi**: encoding, thứ tự đo chỉ số, mặt nạ làn C5, tốc độ trung bình C1, tuổi pha xanh, đơn vị thời gian, bộ đếm chờ pha của legacy. Script đối chiếu cũng đã so thêm 3 tích phân reward. MOBIL và cách xe dừng đèn đỏ được port sát GAMA hơn, các test bám sự kiện GAMA vẫn qua, và tổng số xe qua của seed 1 vẫn là 4.066 (README ghi GAMA cũng là 4.066).

**Tuy vậy vẫn chưa đủ để nói “Python thay được GAMA” trong nghiên cứu.** Lý do:
1. **Lỗi mới:** đặc trưng `green_cycles` giờ khác nhau giữa hai backend (mục 3.1). Agent train trên Python sẽ thấy đầu vào khác hẳn khi chạy trên GAMA.
2. Bản sửa bộ đếm chờ pha của legacy chưa khớp GAML (mục 3.2).
3. Script đối chiếu có in sai số tích phân nhưng **không dùng chúng làm tiêu chí đạt/không đạt** (mục 3.3).
4. Vẫn chưa có bằng chứng parity trong repo và chưa có phép thử chéo train/đánh giá (mục 3.4).
5. Bộ test mới chỉ kiểm tra Python tự nhất quán với chính nó, không so với GAMA, nhưng lại kết luận “sẵn sàng thay thế hoàn toàn” (mục 3.5).

| Tiêu chí | Lần 1 (`1cc5b61`) | Lần 2 (`9fe6a86`) |
|---|---|---|
| Chạy được trên Windows | ✗ lỗi encoding | ✓ (trừ in emoji của suite mới) |
| Thứ tự đo chỉ số giống GAMA | ✗ đo sau khi xe chạy | ✓ đo trước khi xe chạy |
| State spatial đúng tài liệu | ✗ | Phần lớn ✓, còn C4 và ô 6 m |
| State giống nhau giữa 2 backend | ✓ | **✗ `green_cycles` khác** |
| So tích phân reward với GAMA | ✗ | Có in, **chưa dùng để đánh giá** |
| Bộ đếm chờ pha legacy | ✗ không chạy | Chạy, nhưng thiếu điều kiện “không có xe” |
| Bằng chứng parity trong repo | ✗ | ✗ |
| Phép thử chéo train/đánh giá | ✗ | ✗ |
| **Dùng thay GAMA cho nghiên cứu?** | Chưa | **Chưa, nhưng gần hơn đáng kể** |

---

## 2. Trạng thái các nhận xét lần trước

| # | Nhận xét lần 1 | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Lỗi encoding UTF-8 trên Windows | ✅ Đã sửa | `read_text`/`open` đều có `encoding="utf-8"`; `test_python_sim.py` qua 10/10 không cần `PYTHONUTF8` |
| 2 | Python đo chỉ số **sau** khi xe chạy, GAMA đo **trước** | ✅ Đã sửa | `step()` giờ là `signal → spawn → measure → move`, đúng thứ tự reflex global rồi agent của GAMA |
| 3 | `front_row` định nghĩa khác nhau | ⚠️ Chưa sửa | GAML vẫn dùng `all_agents where distance_to stop_pt <= 1.5` (Euclid, mọi agent). Test EDGE-04 chỉ kiểm tra định nghĩa của Python |
| 4 | Mốc vị trí xe (tâm/đầu) chưa có test | ⚠️ Chưa có | Không có test nào so ô spatial của cùng một xe trên hai backend |
| 5 | 18,56% cặp xe-tick lệch làn | ❓ Chưa đo lại | MOBIL được sửa theo GAMA (mục 2.1), nhưng không có số liệu mới |
| 6 | Script đối chiếu chỉ so số xe, không có ngưỡng | 🟡 Một phần | Đã so 3 tích phân và có `--enforce`, nhưng ngưỡng không gồm tích phân (mục 3.3) |
| 7 | Thư mục `reports/` (bằng chứng parity) bị xóa | ❌ Chưa sửa | README vẫn trỏ tới các file không tồn tại |
| 8 | Python chậm hơn GAMA | ❌ Chưa cải thiện | Máy này (Python 3.14): **48 s**/episode 8.000 tick. README (Python 3.11): 22 s; GAMA: 13 s |
| 9 | `seed or 1` biến seed 0 thành 1 | ✅ Đã sửa | |
| 10 | Vá `elapsed_seconds` bằng `DT / 0.1` | ✅ Đã sửa | Base class dùng `step_seconds`; `train.py` dùng `env.step_seconds` |
| 11 | State: ô 6 m thay vì 8 m | ❌ Chưa sửa | Vẫn `/ 6.0` → 20 ô phủ 120 m |
| 12 | State: C1 ghi đè | ✅ Đã sửa | Lấy trung bình theo số xe trong ô |
| 13 | State: C4 là hằng số ±1 | ❌ Chưa sửa | Vẫn `1.0 if arm in (0, 2) else -1.0` |
| 14 | State: C5 không phải mặt nạ làn | ✅ Đã sửa | Đặt 1 trên đúng số sublane của từng nhánh (NW = 14) |
| 15 | `green_cycles` luôn 0 khi quyết định | 🟡 Sửa một nửa | Đúng trên Python, **sai trên GAMA** (mục 3.1) |
| 16 | `average_waiting_time`/`average_travel_time` là NaN | 🟡 Đã có số | Nhưng công thức chưa đúng tên gọi (mục 3.6) |
| 17 | Legacy: `_phase_wait` không bao giờ tăng | 🟡 Sửa một phần | Thiếu điều kiện reset khi pha không có xe (mục 3.2) |
| 18 | Legacy: `_leader_gaps()` gọi trong vòng lặp từng xe | 🟡 Cải thiện | Tính một lần mỗi tick, nhưng tính lại mỗi khi có xe đổi làn |
| 19 | Legacy: parity không đạt (lệch 42,5%) | ❓ Chưa đo lại | |

---

## 3. Những vấn đề còn lại và vấn đề mới

### 3.1 Lỗi mới: `green_cycles` khác nhau giữa hai backend (nghiêm trọng)
- `LVLPythonEnv.read_state` truyền `green_cycles=sim.green_ticks`, tức số tick pha hiện tại đã xanh liên tục (có thể lên tới hàng trăm).
- `LVLGamaEnv.read_state` **không truyền** `green_cycles`, nên `_observation` rơi về giá trị mặc định `cycle % 50`, và giá trị này **luôn bằng 0 tại tick quyết định**. GAML (`LeVanLuong.gaml`) cũng không có biến đếm thời gian xanh để đọc.
- `featurize` đưa giá trị này vào mạng (`aux[4] = min(1.0, green_cycles / 600)`). Với Python, agent thấy giá trị khác 0. Với GAMA, agent luôn thấy 0.

Hệ quả: **chính sách train trên Python rồi đánh giá trên GAMA (hoặc ngược lại) nhận đầu vào khác nhau**, và `compare_lvl_training_envs.py` có thể không bắt được vì nó chỉ so spatial MAE và reward.

Cách sửa: thêm biến `green_ticks` vào GAML (tăng trong reflex `signal` khi `phase != "none"`, reset khi về `"none"`), đọc nó trong `STATE_EXPR`, rồi truyền vào `_observation` giống bên Python. Thêm một test khẳng định hai backend cho cùng `green_cycles` sau cùng chuỗi action.

Lưu ý thêm: hệ số chuẩn hóa `600` được đặt cho bước 0,1 s của legacy (tối đa 60 s). Với LVL, bước 0,2 s nên 600 tick = 120 s. Nên chuẩn hóa theo giây, không theo tick.

### 3.2 Legacy: bộ đếm chờ pha chưa khớp GAML
GAML (`intersection-dqn-python.gaml:279-282`):
```
if (phase_demand[a] = 0 or (global_phase = a * 2 and !dqn_all_red)) { phase_wait_cycles[a] <- 0; }
else { phase_wait_cycles[a] <- phase_wait_cycles[a] + 1; }
```
Python (`python_env.py:306-310`) chỉ reset khi pha đó đang xanh, **thiếu điều kiện `phase_demand[a] = 0`**. Hệ quả: khi bật `dqn_service_guard`, một pha không có xe nào vẫn tích lũy thời gian chờ. Sau 1.200 tick, mask sẽ **ép agent phục vụ một pha trống**, trong khi GAMA không bao giờ làm vậy. Test `LEGACY-01` không bắt được lỗi này vì không thử trường hợp pha không có xe.

### 3.3 Script đối chiếu: có số liệu nhưng tiêu chí chưa đúng
`compare_lvl_python_gama.py --enforce` chỉ kiểm tra:
- `phase_match`,
- `final_active_relative_error <= tolerance`,
- `final_passed_error <= max(5, 1% horizon)`.

Sai số `queue_time_integral`, `system_time_integral`, `weighted_delay_integral` **chỉ được in ra** mà không được kiểm tra. Đây lại chính là ba đại lượng tạo ra reward R1, R2, R4. Đề xuất: thêm điều kiện sai số tương đối của từng tích phân ≤ `tolerance`, và kiểm tra ở mọi mốc lấy mẫu chứ không chỉ cuối lượt.

### 3.4 Thiếu bằng chứng và phép thử chéo
- Thư mục `reports/` vẫn không có trong repo. README vẫn dẫn ra các file trong đó, nên mọi con số parity (4.066/4.054/4.117 xe, MAE reward 0,19, 18,56% lệch làn) không kiểm lại được. Cũng chưa có số liệu đo lại sau các thay đổi MOBIL ở commit này.
- Chưa có phép thử quan trọng nhất: **train trên Python, đánh giá trên GAMA (và ngược lại), so thứ hạng các chính sách**.

### 3.5 Bộ test hành vi mới: hữu ích nhưng kết luận quá mức
`scripts/run_vehicle_behavior_suite.py` có 30 test, tôi chạy và qua cả 30. Bộ này hữu ích làm test hồi quy: bảo toàn số xe, không vượt đèn đỏ, cooldown MOBIL, biên sublane, mặt nạ C5, không NaN, kẹt xe nặng vẫn chạy.

Nhưng:
- **Không test nào so với GAMA.** Bộ này chỉ kiểm tra Python không tự mâu thuẫn, không chứng minh Python giống GAMA. Dòng kết luận cuối script (“sẵn sàng thay thế hoàn toàn cho quá trình huấn luyện RL”) **không được các test này chứng minh**, và không nên đưa vào báo cáo.
- Script dùng file `.py` riêng thay vì pytest, nên không chạy cùng `pytest`/CI.
- Trên Windows, script **crash ngay dòng đầu** vì in emoji ra console cp1252 (`UnicodeEncodeError`). Phải chạy với `PYTHONUTF8=1`. Cách sửa: thêm `sys.stdout.reconfigure(encoding="utf-8")` hoặc bỏ emoji.
- Một số test kiểm tra lại đúng định nghĩa đang có, chứ không kiểm tra định nghĩa đúng. Ví dụ EDGE-04 xác nhận `front_row` của Python, trong khi vấn đề là nó khác GAML (mục 2, dòng 3).

### 3.6 Chỉ số thời gian chờ và thời gian di chuyển: tên không đúng nội dung
`lvl_env._observation`:
- `average_waiting_time = queue_time_integral / created`: tổng thời gian dừng của **mọi xe đã tạo** (kể cả xe còn trên mạng) chia cho số xe tạo. Đây là “thời gian dừng trung bình tính tới hiện tại”, không phải thời gian chờ trung bình của xe đã đi qua.
- `average_travel_time = system_time_integral / passed`: tử số gồm cả thời gian của các xe **còn trên mạng**, mẫu số chỉ có xe đã ra. Khi mạng đông, giá trị này bị phóng đại (seed 1: 213.870 / 4.066 ≈ 52,6 s, trong đó có phần của 146 xe chưa ra).
- `max_waiting_time` vẫn là NaN.

Hai backend tính giống nhau nên không ảnh hưởng parity. Nhưng nếu báo cáo dùng các cột này làm “thời gian chờ/di chuyển trung bình” thì sai định nghĩa. Muốn đúng thì phải ghi thời gian vào/ra của từng xe ở cả GAML và Python.

### 3.7 Các thay đổi động học mới: đúng hướng, cần đối chiếu lại
Commit `9fe6a86` sửa lại một phần lõi `_move`/`_choose_lane`:
- Xe trong vòng 0,01 m trước vạch khi đèn đỏ được snap vào nút và dừng hẳn, theo `moveAcrossSegments` của GAMA.
- Xe đi qua nút giờ được đưa vào lane map của đường mới, gọi lại MOBIL trên đường mới rồi mới tính IDM.
- MOBIL tính `prior_gap = back_gap + vehicle.length + front_gap` và dùng `leading_distance` đã lưu của xe sau, giống cách GAMA lưu thuộc tính `leading_distance`.
- Thêm thuộc tính `vehicle.road` làm nguồn chính để biết xe đang ở road nào.

Các thay đổi này có chú thích dẫn tới hàm Java tương ứng của GAMA, và 10 test bám sự kiện GAMA vẫn qua. Đây là dấu hiệu tốt. Hai điểm cần lưu ý:
- Ở nhánh `else` của `_move` (xe không qua nút trong tick), `vehicle.road` **không được cập nhật**. Nếu xe trên connector có quãng phanh vượt qua cuối connector trong lúc `predicted_speed <= 0`, nó sẽ có progress lớn hơn chiều dài road mà vẫn bị coi là đang trên connector. Trường hợp này hiếm nhưng nên có `assert progress <= road length` trong debug.
- Vì lõi động học đổi, **mọi số liệu parity cũ (kể cả 18,56% lệch làn) đều cần đo lại** trước khi trích dẫn.

### 3.8 Hiệu năng
Một episode 8.000 tick (seed 1, demand CSV) mất **48 giây** trên máy này với Python 3.14. README ghi khoảng 22 s trên máy khác với Python 3.11, còn GAMA khoảng 13 s. Commit này không cải thiện tốc độ. Nếu lý do chuyển sang Python là để train nhanh hơn thì vẫn chưa đạt; lợi ích hiện tại chỉ là không cần JVM hay GAMA Server.

---

## 4. Việc nên làm tiếp

| # | Việc | Ưu tiên |
|---|---|---|
| 1 | Thêm `green_ticks` vào GAML và `LVLGamaEnv.read_state`; test hai backend cho cùng `green_cycles`; chuẩn hóa theo giây | **Ngay** |
| 2 | Thêm điều kiện `phase_demand[a] = 0` vào bộ đếm chờ pha của `python_env.py`, kèm test | Ngay |
| 3 | `--enforce` kiểm tra cả 3 tích phân reward, ở mọi mốc lấy mẫu | Cao |
| 4 | Chạy lại đối chiếu GAMA ↔ Python sau commit này (≥ 10 seed × 8.000 tick), commit CSV và báo cáo vào repo, sửa link trong README | Cao |
| 5 | Thống nhất định nghĩa `front_row` giữa GAML và Python | Trung bình |
| 6 | Sửa ô 6 m → 8 m và C4 cho khớp tài liệu, hoặc sửa tài liệu cho khớp code | Trung bình |
| 7 | Bỏ câu “sẵn sàng thay thế hoàn toàn” khỏi `run_vehicle_behavior_suite.py`; chuyển sang pytest; sửa lỗi in emoji trên Windows | Trung bình |
| 8 | Phép thử chéo: train trên Python, đánh giá trên GAMA và ngược lại, so thứ hạng chính sách | **Bắt buộc trước khi công bố** |
| 9 | Ghi thời điểm vào/ra từng xe để có thời gian chờ và thời gian di chuyển đúng định nghĩa | Trung bình |
| 10 | Tăng tốc (vector hóa hoặc Numba) nếu cần train dài | Tùy mục tiêu |

**Tiêu chí đề xuất để tuyên bố “Python thay được GAMA”:** trên ≥ 10 seed × 8.000 tick, mọi đặc trưng đầu vào của mạng giống nhau giữa hai backend khi chạy cùng chuỗi action; ba tích phân reward lệch ≤ 2–5% ở mọi mốc; và thứ hạng các chính sách giống nhau khi đánh giá trên hai backend.

---

## 5. Phạm vi chưa kiểm tra

- Không chạy GAMA, nên không đo lại parity sau commit này. Nhận xét 3.1 suy ra từ code (`LVLGamaEnv.read_state` không truyền `green_cycles`, GAML không có biến tương ứng), chưa chạy trên GAMA.
- Không so trực tiếp với mã Java của GAMA (`DrivingSkill`, `MOBIL`), chỉ so với GAML và các chú thích trong code Python.
- Thời gian chạy đo trên một máy, một seed, Python 3.14. Không so được với GAMA trên cùng máy.
