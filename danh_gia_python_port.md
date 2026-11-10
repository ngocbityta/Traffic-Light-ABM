# Đánh giá phần chuyển mô phỏng từ GAMA sang Python

Ngày: 10/11/2026 · Mã nguồn: `main` sau commit `1cc5b61`.

**Phạm vi:** chỉ đánh giá việc **bản Python có tái hiện đúng mô hình GAMA hay không**, và có thay được GAMA khi train/đánh giá RL hay không. Tôi **không** đánh giá mô hình GAMA gốc đúng hay sai so với giao thông thực, vì đó là giả định chung của cả hai bản.

| File | Port từ |
|---|---|
| `LeVanLuongIntersection/python_sim.py` | `LeVanLuongIntersection/models/LeVanLuong.gaml` (driving skill GAMA 2025.6.4) |
| `LeVanLuongIntersection/demand.py` | reflex `spawn` của GAML, đổi sang lịch xe CSV dùng chung |
| `dqn_py/lvl_env.py` | experiment `dqn_server_experiment`: state/reward giờ dựng bằng Python cho cả hai backend |
| `dqn_py/python_env.py` | `Traffic/models/intersection-dqn-python.gaml` (bài toán legacy, cầu vượt/hầm) |
| `scripts/compare_lvl_*`, `compare_python_*`, `diagnose_*` | Công cụ đối chiếu |

**Cách làm:** đọc từng hàm Python và so với GAML tương ứng, chạy test, và tự chạy `python_sim.py` 8.000 tick. Máy này **không có GAMA headless**, nên tôi không chạy lại được các script đối chiếu GAMA ↔ Python.

---

## 1. Kết luận

**Bản LVL (`python_sim.py`) là bản port nghiêm túc, đúng hướng. Ở mức số xe tạo ra và qua nút, nó đã gần GAMA. Nhưng nó chưa đạt mức “thay thế được GAMA” cho một nghiên cứu, vì:**
1. Chưa có bằng chứng tương đương ở các chỉ số mà reward và kết quả nghiên cứu thật sự dùng (tích phân hàng chờ, tốc độ, trễ từng xe).
2. Tôi tìm ra ít nhất một chênh lệch có hệ thống về thứ tự đo giữa hai bản (mục 3.1).
3. File báo cáo parity mà README dẫn ra đã bị xóa khỏi repo.
4. Python chậm hơn GAMA, nên lý do tốc độ để chuyển chưa thành hiện thực.

**Bản legacy (`python_env.py`) chưa thay được GAMA.** Chính README ghi lệch tới 42,5% số xe vào và 38,7% tốc độ trung bình ở 4.000 tick, và tôi tìm thêm một lỗi logic (mục 4.1).

| Tiêu chí | LVL `python_sim.py` | Legacy `python_env.py` |
|---|---|---|
| Logic đèn, lịch quyết định | Khớp | Gần khớp, guard bị lỗi |
| Phát xe theo CSV | Khớp | Khớp |
| Tổng số xe qua (theo README) | Lệch 0–2 xe / 4.000+ xe | Lệch 7,1% |
| Từng xe (làn, tốc độ) | Lệch 18,56% cặp xe-tick về làn | Lệch lớn |
| Chỉ số dùng cho reward | Chưa đối chiếu trực tiếp, có lệch hệ thống | Lệch 10,7% tích phân hàng chờ |
| Bằng chứng trong repo | Đã bị xóa | Đã bị xóa |
| Chạy trên Windows | Lỗi encoding | Chạy được |
| Tốc độ so với GAMA | Chậm hơn (khoảng 22 s so với 13 s/episode) | Không đo |
| **Dùng thay GAMA cho nghiên cứu?** | **Chưa, gần được** | **Không** |

---

## 2. Những chỗ port đúng (LVL)

### 2.1 Chép đúng tham số
So từng giá trị trong `vehicle.init` (GAML dòng 219–237) với `_create_vehicle` (Python dòng 172–193):

| Tham số | GAML | Python | |
|---|---|---|---|
| Chiều dài xe | 4,5 / 1,9 m | 4,5 / 1,9 | ✓ |
| Số sublane chiếm | 2 / 1 | 2 / 1 | ✓ |
| `max_acceleration` | 2,5 / 2,0 | 2,5 / 2,0 | ✓ |
| `time_headway` | 1,2 / 0,6 | 1,2 / 0,6 | ✓ |
| `min_safety_distance` | 1,0 / 0,3 | 1,0 / 0,3 | ✓ |
| `politeness_factor` | 0,1 | 0,1 (dòng 325) | ✓ |
| `acc_gain_threshold` | 0,05 | 0,05 (dòng 326) | ✓ |
| `lane_change_limit <- -1` (đổi sang làn bất kỳ) | -1 | xét mọi làn (dòng 303) | ✓ |
| `maxspeed` connector / road khác | 40 / 60 km/h | 40 / 60 (dòng 432) | ✓ |
| Bước mô phỏng | 0,2 s | `DT = 0.2` | ✓ |
| Hình học | `roads.geojson` | cùng file, tự tính chiều dài polyline | ✓ |

### 2.2 Đèn tín hiệu khớp từng tick
- Chế độ cố định: `_phase()` dùng `(cycle-1)*DT` để bù việc biến `time` của GAMA trễ một bước trong reflex (dòng 111–112). Chi tiết này rất dễ bỏ sót.
- Chế độ train: `_advance_signal()` chép đúng reflex `signal` (GAML 73–88): đỏ toàn phần 15 tick, `pending_phase`, cờ chờ action mỗi 50 tick. GAML dùng `(cycle+1) mod 50`, Python dùng `cycle % 50` sau khi đã tăng `cycle` trước, nên hai cách tương đương.
- README ghi 159/159 quyết định và pha khớp trên 3 seed × 8.000 tick. Code khớp với số liệu đó.

### 2.3 Phát xe theo CSV
- `demand.py` sinh đúng 6 cột mà reflex `spawn_replay_demand` đọc. Hai bên dùng chung chỉ số `[NE, SE, SW, NW]` và `[left, right, through]`.
- Điều kiện thời gian `rows[0] > cycle + 1` (GAML) tương đương `row[0] <= self.cycle` (Python, `cycle` đã tăng trước).
- Cả hai kiểm tra định dạng. Python còn từ chối CSV không sắp xếp.
- `train.py` chặn `--backend python` khi không dùng demand ghép cặp. Đây là quyết định đúng, vì RNG của GAMA và Python khác nhau.

### 2.4 Bám sát cách driving skill của GAMA cập nhật
Đây là điểm mạnh nhất của bản port:
- **Cập nhật tuần tự theo thứ tự tạo xe** (`_move`, dòng 196–290): xe sau thấy vị trí mới của xe trước trong cùng tick, giống cách GAMA duyệt agent. Nếu làm kiểu “tính hết rồi cập nhật đồng loạt” thì không bao giờ khớp được từng xe.
- **IDM** (`_idm`) theo đúng công thức `IDM.computeAcceleration` của GAMA. Gặp đèn đỏ hay cuối tuyến thì coi như có xe ảo đứng yên, cách thêm `min_safety_distance`.
- **Qua nút trong tick** (dòng 230–257): đi đến nút bằng tốc độ dự đoán của đường cũ, rồi tính lại IDM với giới hạn tốc độ đường mới cho thời gian còn lại. Đây là chi tiết của `moveAcrossSegments` trong GAMA.
- **Snap 0,01 m ở cuối tuyến** (dòng 281) giống GAMA. Thiếu chi tiết này thì xe không bao giờ “đến đích”.
- **Tìm xe trước ở đường kế tiếp khi đèn xanh** và **tìm xe sau xuyên qua nút nguồn khi đang trên connector** (`_neighbors`), theo `findLeader/findFollower`.
- **MOBIL**: không tính hai lần xe sau đang là follower hiện tại, bỏ qua khi xe sau đã có leader gần hơn, có ngưỡng phanh an toàn −4 m/s², cooldown 4 s, xử lý trường hợp xe đè lên nhau khi vừa sinh (`overlap = old_gap < 0`).
- Test hồi quy bám vào sự kiện GAMA cụ thể: đổi làn ở tick 37 và tốc độ IDM ở tick 68 của seed 1. Đây là kiểu test đúng cho một bản port.

### 2.5 Đối chiếu observation bằng chung mã
`lvl_env._observation()` nhận danh sách xe thô `(speed, max_speed, lane, occupied, road, remaining)` từ **cả** GAMA (qua biểu thức `VEHICLES_EXPR`) và Python. Vì vậy chênh lệch state hay reward chỉ đến từ động học, không đến từ cách tính feature. Có thêm kiểm tra `len(vehicles) == created - passed` để phát hiện snapshot GAMA bị thiếu. Đây là thiết kế tốt.

---

## 3. Những chỗ port chưa đúng hoặc chưa đủ (LVL)

### 3.1 Thứ tự đo chỉ số lệch một pha so với GAMA (nghiêm trọng với reward)
Trong GAMA, ở mỗi cycle **agent world (global) chạy reflex trước, sau đó mới đến các agent `vehicle`**. Thứ tự trong một cycle của GAML vì vậy là:

```
signal → spawn_replay_demand → measure_front_row (đo) → vehicle.choose_path → vehicle.move (đi)
```

Còn Python (`step`, dòng 471–476) làm:

```
_advance_signal → _spawn → _move (đi) → _measure (đo)
```

Hệ quả:
- **GAMA đo trước khi xe chạy:** xe vừa sinh có `speed = 0` nên bị đếm là đang dừng (< 1 km/h) và có trễ 1,0 trong `weighted_delay_integral`. Python đo sau khi xe đã tăng tốc một bước (khoảng 0,4–0,5 m/s ≈ 1,4–1,8 km/h), nên xe đó **không** bị tính là đang dừng.
- Với khoảng 4.200 xe mỗi episode, ước tính `queue_time_integral` của GAMA lớn hơn Python khoảng **4.200 × 0,2 s ≈ 840 xe·s**. `weighted_delay_integral` cũng lệch tương tự. `system_time_integral` lệch ít hơn vì chỉ dịch thời điểm một tick.
- `queue_time_integral` → R2 (`stopped_time_integral` = `queue_time_integral` trong `lvl_env`), `weighted_delay_integral` → R4. Hai reward này **có sai lệch hệ thống giữa hai backend**, không chỉ là nhiễu.
- `front_row_max` cũng đo ở hai thời điểm khác nhau.

Lưu ý: tôi suy ra điều này từ quy tắc lập lịch của GAMA và từ code, chưa đo được vì không có GAMA. Kiểm tra: so `queue_time_integral` của hai backend ở tick 1–5 của seed 1. **Cách sửa:** trong Python, chuyển `_measure()` lên trước `_move()` (sau `_spawn()`).

### 3.2 Định nghĩa `front_row` khác nhau
- GAML (dòng 180–188): `all_agents where (distance_to stop_pt <= 1.5)`. Đây là khoảng cách **Euclid** từ `location` tới điểm vạch dừng, tính trên **mọi agent**, kể cả xe của nhánh khác, xe đã qua vạch hay đang trên connector, và các agent không phải xe (`road`, `area`, `intersection`, được ép kiểu `vehicle(v)`).
- Python (dòng 463–465): chỉ tính xe **của nhánh đó**, **trước vạch**, với `0 <= approach_end - position <= 1.5`.

Hai bản đo hai đại lượng khác nhau. Việc “`front_row_max` khớp cả bốn nhánh” trong README có thể chỉ là trùng hợp do giá trị bão hòa (số sublane tối đa). Một trong hai bên cần sửa cho cùng định nghĩa, và có lẽ GAML mới là bên sai.

### 3.3 Mốc vị trí xe (tâm hay đầu xe) chưa được làm rõ
Python coi `position` là **tâm xe** (khoảng cách bumper = khoảng cách tâm − nửa tổng chiều dài, dòng 371). Phía GAMA, `lvl_env` đọc `distance_to_current_target`. Nếu GAMA đo từ một mốc khác (đầu xe hoặc tâm) thì ô DTSE/spatial (6 m) của ô tô 4,5 m có thể lệch 1 ô. README cho spatial MAE khoảng 0,022 nên có vẻ không lệch nhiều, nhưng **nên có một test khẳng định** cùng một xe trên hai backend rơi vào cùng ô.

### 3.4 Sai lệch từng xe vẫn đáng kể
README tự ghi **18,56% cặp xe-tick lệch làn** sau 200 tick. Với MOBIL được phép nhảy mọi làn, một lần chọn khác là quỹ đạo tách hẳn. Nguyên nhân có thể nằm ở các chi tiết chưa port hết như thứ tự xét làn, hòa điểm `gain` hay sai số dấu phẩy động. Về mặt tổng thể (số xe qua) thì khớp vì nút không bão hòa, nhưng **state spatial là ảnh sublane theo từng xe**. Agent train trên Python sẽ thấy phân bố hàng chờ theo làn khác với khi chạy trên GAMA.

### 3.5 Phương pháp đối chiếu còn yếu
- `compare_lvl_python_gama.py` chỉ so `active`, `created`, `passed`, `front_row_max`, **không** so `queue_time_integral`, `system_time_integral`, `weighted_delay_integral`, tốc độ trung bình. Script cũng **không có ngưỡng đạt/không đạt** (khác `compare_python_gama.py` của legacy, có `--tolerance --enforce`).
- `created` khớp là đương nhiên vì cùng CSV, không chứng minh gì. Còn `passed` gần như không nhạy với động học khi mọi xe vào đều ra.
- `compare_lvl_training_envs.py` có so reward R1 và spatial MAE, đây là hướng đúng. Nhưng chỉ có 3 seed và chỉ có MAE trung bình, không có phân phối hay giá trị lớn nhất.
- **Thiếu phép thử quan trọng nhất:** train trên Python rồi đánh giá trên GAMA (và ngược lại), xem thứ hạng các chính sách có giữ nguyên không. Chính README thừa nhận “chưa chứng minh quá trình cập nhật trọng số khi train hoàn toàn tương đương”.

### 3.6 Bằng chứng parity đã bị xóa
README dẫn ra `reports/lvl_python_gama_parity_current_20261110.md`, `reports/python_parity_20261109/…` và `reports/python_parity_20261110/`. Commit `1cc5b61` **xóa toàn bộ thư mục `reports/`**. Mọi con số parity trong README hiện không kiểm lại được từ repo. Với nghiên cứu, đây là vấn đề về tính tái lập.

### 3.7 Lỗi encoding trên Windows
`python_sim.py:27` gọi `Path(path).read_text()` mà không truyền `encoding`. Trên Windows, Python mặc định dùng cp1252 nên vỡ khi gặp ký tự tiếng Việt trong GeoJSON. Kết quả chạy:
- `test_python_sim.py`: **10/10 lỗi**, qua 10/10 khi đặt `PYTHONUTF8=1`.
- `lvl_env.py` gọi `road_lengths()` ngay khi import, nên **`train.py`/`evaluate.py --scenario lvl` hỏng ngay từ đầu** trên Windows.

Cách sửa: thêm `encoding="utf-8"` vào `read_text()` và vào các `open()` ở `python_sim.py`/`demand.py`.

### 3.8 Hiệu năng: lý do để port chưa thành hiện thực
- README: Python khoảng **21,6–22,1 s** mỗi episode 8.000 tick, GAMA khoảng **13,3–13,4 s**. Bản Python chậm hơn khoảng 1,6 lần.
- Chạy 2 worker chỉ nhanh hơn 1 worker khoảng 1,24 lần (README).
- Lợi ích còn lại là không cần JVM hay GAMA Server, dễ chạy song song và dễ chạy trên Colab. Nhưng nếu muốn train hàng triệu bước thì cần vector hóa theo làn (NumPy), dùng PyPy hoặc viết phần lõi bằng Numba hay Cython.
- Ở cùng lưu lượng, thời gian chạy tăng theo số xe trên mạng (lần chạy của tôi: 1.200 → 8.000 xe/h làm thời gian tăng khoảng 10 lần).

### 3.9 Các điểm nhỏ trong code port
- `LVLPythonEnv.reset`: `seed or 1` biến seed 0 thành 1. Ở chế độ replay RNG không được dùng nên hiện chưa ảnh hưởng.
- `_LVLCommon.step` sửa `elapsed_seconds` bằng phép nhân `DT / 0.1`, vì `GamaTrafficEnv.step` tính cứng `advanced * .1`. `train.py` còn viết cứng `.2 if scenario == "lvl" else .1` ở 3 chỗ. Nên để base class dùng `self.step_seconds`, nếu không thì đổi `DT` sẽ làm sai discount mà không báo lỗi.
- `_free_idm_cache` dùng `id(vehicle)` làm khóa. Hiện an toàn vì xe chỉ bị xóa cuối `_move`, nhưng dễ vỡ khi đổi thứ tự.
- Chế độ phát xe ngẫu nhiên (không CSV) dùng RNG Python nên không bao giờ khớp GAMA. Điều này đúng như README nói, nhưng `python_sim.py --seed` vẫn cho chạy kiểu này mà không cảnh báo.

### 3.10 State ở `lvl_env.py` khác với state GAMA trước đây
Trước commit này, khi train `--site lvl`, spatial state được dựng trong GAMA, và README LVL mô tả C5 là mặt nạ theo số sublane thật, ô 8 m. Bản chuyển sang Python dựng lại state trong `_observation()` nhưng **ngữ nghĩa thay đổi**:
- ô dọc **6 m** thay vì 8 m,
- C4 là **hằng số ±1 theo trục đường**,
- C5 là ô có xe, **không còn là mặt nạ làn hợp lệ**,
- C1 ghi đè bằng xe cuối cùng thay vì lấy trung bình,
- `green_cycles = cycle % 50` luôn bằng 0 ở tick quyết định,
- `average_waiting_time`/`average_travel_time` là `NaN`.

Hai backend vẫn khớp nhau vì dùng chung hàm, nhưng **không còn khớp với tài liệu và với state GAMA cũ**. Kết quả train trước và sau commit không so sánh trực tiếp được.

---

## 4. Bản legacy `python_env.py`

### 4.1 Lỗi: bộ đếm chờ pha không bao giờ tăng
`self._phase_wait` chỉ được gán `[0]*4` (dòng 138, 184) rồi đọc ở dòng 782. GAML (`intersection-dqn-python.gaml:278–284`) tăng `phase_wait_cycles[a]` mỗi tick và ép phục vụ pha đợi quá 1.200 tick. Khi bật `dqn_service_guard`, **action mask của Python khác GAMA**: pha bị bỏ đói không bao giờ được ép phục vụ.

### 4.2 Port gần đúng, nhiều hằng số khớp cho vừa trace
- Docstring tự nhận đây là “deliberately explicit approximation”, `fidelity = "approximate-longitudinal-v1"`. Điều này minh bạch, tốt.
- Nhiều hằng số được chỉnh để trace khớp GAMA thay vì port đúng cơ chế: `FLYOVER_PORTAL_CLEARANCE = 66.0` (GAML là 70 m), các hệ số phanh `2.5`, `3.0`, `2.0` ở dòng 664–667, `ramp_stop_ticks >= 50`. Khớp kiểu này dễ vỡ khi đổi demand hoặc horizon. Đây chính là lý do parity trôi xa ở 4.000 tick.
- Đổi làn là dịch chuyển tức thời kèm `lateral_peak` giả lập, không phải nội suy như GAMA.

### 4.3 Hiệu năng
`_choose_lanes` gọi `self._leader_gaps()` (dựng lại và sắp xếp mọi nhóm xe) **bên trong vòng lặp từng xe** (dòng 507), nên mỗi tick tốn khoảng O(n² log n).

### 4.4 Kết quả parity (theo README)
Seed 1002, 4.000 tick: lệch 42,5% số xe nhận, 7,1% xe qua nút, 38,7% tốc độ trung bình, 10,7% tích phân hàng chờ, và lịch pha đã trôi. **Không đạt ngưỡng 10%.** Không nên dùng `--scenario legacy --backend python` cho bất kỳ kết quả nào.

---

## 5. Để Python thay được GAMA cho nghiên cứu

| # | Việc | Ưu tiên |
|---|---|---|
| 1 | Sửa encoding UTF-8 (`python_sim.py`, `demand.py`) | Ngay |
| 2 | Sửa thứ tự đo: `_measure()` trước `_move()`, rồi đối chiếu lại `queue_time_integral`/`weighted_delay_integral` ở vài tick đầu | Ngay |
| 3 | Thống nhất định nghĩa `front_row` giữa GAML và Python | Cao |
| 4 | Commit lại thư mục `reports/` (hoặc một thư mục bằng chứng parity) cùng CSV gốc | Cao |
| 5 | Mở rộng `compare_lvl_python_gama.py`: so cả 3 tích phân reward, tốc độ trung bình, hàng chờ theo nhánh; có `--tolerance --enforce`; chạy ≥ 10 seed | Cao |
| 6 | Test khẳng định một xe cụ thể rơi vào cùng ô spatial trên cả hai backend | Trung bình |
| 7 | Tìm nguyên nhân 18,56% lệch làn (log từng quyết định MOBIL của hai bên ở tick lệch đầu tiên) | Trung bình |
| 8 | Phép thử chéo: train trên Python, đánh giá trên GAMA và ngược lại; so thứ hạng chính sách | **Bắt buộc trước khi công bố** |
| 9 | Ghi lại nghĩa thật của 6 kênh state, hoặc sửa `_observation` cho khớp tài liệu | Trung bình |
| 10 | Legacy: sửa `_phase_wait`, đưa `_leader_gaps()` ra ngoài vòng lặp; hoặc ghi rõ là không hỗ trợ | Thấp (nếu đã bỏ legacy) |
| 11 | Tăng tốc (vector hóa hoặc Numba) nếu lý do port là tốc độ | Tùy mục tiêu |

**Tiêu chí đề xuất để tuyên bố “Python thay được GAMA”:** trên ≥ 10 seed × 8.000 tick, ba tích phân reward lệch ≤ 2–5%, và chính sách tốt nhất (cùng thứ tự xếp hạng của các baseline) giống nhau khi đánh giá trên hai backend.

---

## 6. Phạm vi chưa kiểm tra

- Không chạy GAMA, nên không tự đo lại các con số parity và chưa xác nhận bằng số liệu chênh lệch ở mục 3.1.
- Không so trực tiếp với mã nguồn Java của GAMA (`DrivingSkill`, `MOBIL`), chỉ so với GAML và các chú thích trong code Python.
- Không đọc chi tiết `observer.html`, `lvl_observer_server.py` và các script `probe_*`/`prototype_*`, vì không ảnh hưởng kết quả.
