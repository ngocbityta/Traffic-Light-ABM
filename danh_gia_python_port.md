# Đánh giá phần chuyển mô phỏng GAMA sang Python

Ngày đánh giá: 10/11/2026 · Mã nguồn: `main` sau merge `2cb1060` (gồm commit upstream `1cc5b61`).

Phạm vi: toàn bộ phần mới thêm để thay GAMA bằng Python khi train RL:

| File | Vai trò |
|---|---|
| `LeVanLuongIntersection/python_sim.py` | Mô phỏng Python cho nút Lê Văn Lương (LVL), IDM + MOBIL, đọc `roads.geojson` |
| `LeVanLuongIntersection/demand.py` | Sinh lịch xe CSV dùng chung cho GAMA và Python |
| `LeVanLuongIntersection/models/LeVanLuong.gaml` | Mô hình GAMA tham chiếu của nút LVL |
| `dqn_py/lvl_env.py` | Env RL cho LVL (cả backend GAMA và Python), dựng state/reward chung |
| `dqn_py/python_env.py` | Bản Python của mô hình legacy `Traffic/` (cầu vượt, hầm, 4 pha) |
| `dqn_py/train.py`, `dqn_py/evaluate.py` | Thêm `--scenario lvl/legacy`, `--backend python/gama`, nhiều worker Python |
| `scripts/compare_*`, `diagnose_*`, `benchmark_*` | Đối chiếu GAMA ↔ Python, benchmark tốc độ |

Cách làm: đọc code, so từng dòng với GAML, chạy test, rồi tự chạy thêm một script dò (probe) trên `python_sim.py` ở 4 mức lưu lượng, mỗi mức 8.000 tick (1.600 s mô phỏng). Tôi **không** chạy lại các script đối chiếu với GAMA vì máy này không có GAMA headless.

---

## 1. Kết luận ngắn

**Trả lời thẳng: ở trạng thái hiện tại, code đủ tốt để làm môi trường thử nghiệm và phát triển thuật toán RL. Chưa đủ để làm nền cho một nghiên cứu lớn có kết luận định lượng về giao thông thực (ví dụ “giảm X% thời gian chờ tại nút Lê Văn Lương”).**

Lý do chính không nằm ở việc chuyển GAMA sang Python, mà ở **chính mô hình giao thông** (cả GAMA lẫn Python đều mang cùng các giả định):

1. Xe trên các hướng cắt nhau trong nút **đi xuyên qua nhau**, không có xung đột, không nhường đường. Rẽ trái đi cùng pha với đi thẳng ngược chiều mà không mất thời gian.
2. Năng lực thông qua gần như không giới hạn: ở 8.000 xe/h/nhánh (32.000 xe/h cả nút), vẫn có 94% xe qua được nút trong 1.600 s. Ngoài đời, nút này bão hòa sớm hơn nhiều.
3. Phanh tới khoảng 80 m/s² (khoảng 8g) và đổi làn tức thời qua tối đa 15 sublane trong một tick 0,2 s. Về vật lý, những chuyển động này không thể xảy ra.
4. Toàn bộ nhu cầu (2.400 xe/h/nhánh, 85% xe máy, 20%/20% rẽ trái/phải) và hình học sublane là **giả định**, chưa hiệu chỉnh bằng số đếm hay video thực.

Còn về việc chuyển sang Python thì làm khá kỹ: cùng lịch xe, cùng thứ tự cập nhật tuần tự như GAMA, cùng mã dựng state/reward. Tổng số xe qua nút khớp GAMA ở mức vĩ mô. Tuy vậy vẫn còn một số lỗi và chỗ chưa khớp, liệt kê ở mục 4 và 5.

| Hạng mục | Mức sẵn sàng |
|---|---|
| Hạ tầng train RL (env, nhiều worker, checkpoint, resume, seed) | Tốt |
| Tính tái lập (CSV demand, seed, hash mã nguồn) | Tốt |
| Tương đương GAMA ↔ Python ở mức tổng (số xe, pha) | Khá (theo số liệu README, chưa kiểm lại được) |
| Tương đương ở mức từng xe | Chưa đạt (README tự nêu 18,56% cặp xe-tick lệch làn) |
| Độ thực của mô hình giao thông | Yếu, chưa dùng được cho kết luận thực tế |
| Hiệu chỉnh và kiểm chứng bằng dữ liệu thực | Chưa có |
| State/feature cho RL | Có lỗi thiết kế (mục 5) |
| Bản legacy `python_env.py` | Chưa đạt parity, tự README ghi “không đạt” |

---

## 2. Kết quả chạy kiểm tra

### 2.1 Test

| Bộ test | Kết quả |
|---|---|
| `dqn_py/test_python_env.py` | 10/10 qua |
| `LeVanLuongIntersection/test_python_sim.py` | **10/10 lỗi trên Windows**, qua 10/10 khi đặt `PYTHONUTF8=1` |

Nguyên nhân: `python_sim.py:27` gọi `Path(path).read_text()` mà không truyền `encoding="utf-8"`. Trên Windows, Python đọc bằng cp1252 nên vỡ ở ký tự tiếng Việt trong GeoJSON. **Lỗi này làm `train.py --scenario lvl` hỏng ngay khi import trên Windows** (vì `lvl_env.py` gọi `road_lengths()` lúc import), và notebook/Colab trên Windows cũng bị ảnh hưởng. Cách sửa: thêm `encoding="utf-8"` ở `read_text()` và ở các lệnh `open()` đọc/ghi CSV.

### 2.2 Dò độ thực của `python_sim.py` (seed 1, 8.000 tick = 1.600 s, đèn cố định 50/40/3 s)

| Lưu lượng/nhánh | Xe tạo | Xe qua | % qua | Xe trên mạng (đỉnh) | Phanh lớn nhất | Nhảy làn lớn nhất | Mẫu xe chồng nhau | Mẫu có rẽ trái cắt hướng khác |
|---|---|---|---|---|---|---|---|---|
| 1.200 xe/h | 2.122 | 2.060 | 97% | 100 | 79 m/s² | 15 sublane | 0 | 73/160 |
| 2.400 xe/h (mặc định) | 4.212 | 4.066 | 97% | 192 | 72 m/s² | 15 sublane | 2 | 108/160 |
| 4.800 xe/h | 8.515 | 8.187 | 96% | 381 | 82 m/s² | 15 sublane | 79 | 139/160 |
| 8.000 xe/h | 14.237 | 13.405 | 94% | 880 | 80 m/s² | 15 sublane | 195 | 156/160 |
| 2.400, chế độ train (đổi pha mỗi 50 s) | 4.212 | 4.079 | 97% | 188 | 74 m/s² | 15 sublane | 1 | 101/160 |

Cách đọc:
- **% xe qua gần như không đổi khi lưu lượng tăng gần 7 lần.** Nút không bao giờ bão hòa. Một bài toán điều khiển đèn mà nút không bao giờ ùn thì chênh lệch giữa các chính sách sẽ nhỏ và kém tin cậy.
- **Phanh 70–80 m/s²**: xe đang chạy dừng gần như ngay lập tức khi đèn chuyển đỏ hoặc gặp xe phía trước. IDM gốc không chặn giới hạn phanh; code chỉ kẹp `distance = min(distance, gap)` nên xe “dừng kiểu dịch chuyển tức thời”.
- **Nhảy 15 sublane trong một tick**: MOBIL xét mọi làn trên đường (`python_sim.py:303`) và đổi làn tức thời, không có chuyển động ngang. Điều này khớp với GAMA (`lane_change_limit <- -1`), nhưng xe máy có thể sang hẳn từ mép phải qua mép trái trong 0,2 s.
- **Xe chồng nhau**: không có điều kiện chặn khi phát xe (xem 4.3); khi đông, xe mới đặt đè lên xe đang đứng ở đầu đường.
- **Rẽ trái cắt hướng khác**: trong 108/160 mẫu ở lưu lượng mặc định, có xe rẽ trái nằm trong nút cùng lúc với xe của nhánh khác, nhưng không xe nào thấy xe nào.
- Thời gian trong bảng gồm cả chi phí của script dò; README ghi khoảng 22 s mỗi episode 8.000 tick khi chạy riêng, chậm hơn GAMA (khoảng 13 s).

---

## 3. Những chỗ hợp lý (điểm mạnh)

### 3.1 Thiết kế đối chiếu GAMA ↔ Python

- **Lịch xe CSV dùng chung** (`demand.py`): cả hai backend nhận cùng tick, nhánh, loại xe, hướng rẽ, làn, tốc độ mong muốn. Đây là điều kiện bắt buộc để so cặp, và nhóm đã làm đúng. `train.py` cũng chặn `--backend python` khi không dùng `--demand-mode paired`.
- **Cùng thứ tự cập nhật tuần tự**: `_move()` cho từng xe lần lượt chọn làn rồi di chuyển, xe sau đọc trạng thái mới của xe trước, giống cách GAMA duyệt agent. Phần lớn bản port tự viết bỏ qua chi tiết này rồi lệch mà không rõ vì sao.
- **Chú ý lệch một tick**: `_phase()` dùng `(cycle-1)*DT` đúng như `time` của GAMA trong reflex; GAML dùng `(cycle+1) mod 50` còn Python dùng `cycle % 50`. Các chỗ lệch này đều được ghi chú.
- **Chuyển đường giữa tick** (`python_sim.py:230-257`): mô phỏng đúng việc GAMA tính lại IDM với giới hạn tốc độ của đường mới cho phần thời gian còn lại của tick.
- **Tìm xe sau xuyên nút nguồn** và **tìm xe trước ở đường kế tiếp khi đèn xanh** (`_neighbors`) bám theo hành vi `findLeader/findFollower` của GAMA.
- **Mã state/reward chung** (`lvl_env._observation`): GAMA và Python đều đưa danh sách xe thô vào cùng một hàm Python, nên khác biệt observation chỉ còn đến từ động học, không đến từ cách tính feature.
- **Test bám vào sự kiện GAMA cụ thể** (ví dụ đổi làn ở tick 37, tốc độ IDM ở tick 68). Đây là test hồi quy có giá trị.

### 3.2 Hạ tầng RL

- Nhiều worker Python bằng `ProcessPoolExecutor`, một learner chung, gộp theo thứ tự seed.
- Checkpoint, `training_state.pt` có replay/optimizer/RNG, resume bị từ chối nếu mã mô hình đổi (hash file nguồn), ghi `backend` và `fidelity` vào summary để không lẫn kết quả.
- Seed validation và test tách riêng (10.000.001+, 20.000.001+).
- Reward R1–R4 là **gia số của chi phí tích lũy theo từng tick**, kể cả lúc đỏ toàn phần. Định nghĩa này đúng về mặt toán và tránh được lỗi “chỉ đo tại thời điểm quyết định”.
- `transition_reward` báo lỗi nếu chi phí tích lũy giảm. Đây là một kiểm tra bất biến tốt.
- `evaluate.py` từ chối checkpoint của scenario khác.

### 3.3 Hình học

- Hình học lấy từ ảnh vệ tinh, có ghi sai số, nguồn ảnh, và chỗ nào còn nghi ngờ (nhánh SW sát ngưỡng 16 sublane, NW bị cây che).
- Cùng một `roads.geojson` cho cả hai backend, Python tự tính chiều dài polyline.
- README tự nêu các giả định chưa có số liệu và các điểm chưa khớp từng xe. Sự minh bạch này rất đáng giữ.

---

## 4. Những chỗ chưa hợp lý: mô hình giao thông (ảnh hưởng cả GAMA và Python)

Đây là nhóm vấn đề **quyết định** việc có dùng được cho nghiên cứu hay không.

### 4.1 Không có xung đột trong nút (nghiêm trọng)
`advanced_driving` chỉ xét xe trên cùng một road; mỗi connector là một road riêng. Python port đúng y như vậy. Hệ quả:
- Rẽ trái của NE và đi thẳng của SW cùng xanh nhưng không cắt nhau, không có “chờ khe hở” (gap acceptance).
- Không có hiện tượng xe kẹt giữa nút, không có “khóa nút” (gridlock), không có “xe tràn vào nút khi hết xanh”.
- Năng lực thông qua bị đánh giá quá cao, nên mọi khác biệt giữa các chính sách điều khiển đèn bị thu nhỏ hoặc sai lệch.

Với nút có tới 85% xe máy ở Hà Nội, hành vi đan xen trong vùng xung đột 1.919 m² **chính là hiện tượng cần mô hình hóa**. Thiếu phần này thì kết luận về đèn tín hiệu khó thuyết phục hội đồng.

**Đề xuất**: thêm điểm xung đột giữa các connector (giao điểm hình học), mỗi điểm cho một xe chiếm tại một thời điểm, rẽ trái phải nhường xe đi thẳng ngược chiều (gap acceptance, ví dụ khe chấp nhận khoảng 4–5 s với ô tô, khoảng 2–3 s với xe máy). Phần này **dễ làm ở Python hơn GAMA**: đây là lợi thế thật sự của việc đã có bản Python.

### 4.2 Động học chưa có giới hạn vật lý
- Phanh không giới hạn (khoảng 80 m/s² đo được). Nên chặn ở mức phanh khẩn cấp, khoảng 6–9 m/s², và xử lý vùng lưỡng lự (dilemma zone) bằng pha vàng.
- Đổi làn tức thời, nhảy nhiều làn. Nên giới hạn mỗi lần đổi ±1 sublane (hoặc ±2 với xe máy), có thời gian đổi làn và vận tốc ngang.
- **Không có pha vàng** ở LVL (cả GAML và Python): xanh chuyển thẳng sang đỏ toàn phần 3 s. Đỏ toàn phần không đợi nút trống (khác với mô hình legacy có kiểm tra `_junction_clear`).
- Xe đỗ khi gặp đèn đỏ dựa trên tâm xe tới vạch dừng (`stop_gap = approach_end - position + gap`), nên nửa đầu xe vượt vạch. Đây là sai số nhỏ, nhưng ảnh hưởng tới ô DTSE đầu tiên.

### 4.3 Không có hàng đợi ngoài mạng (cả GAMA và Python)
`_spawn()` (Python) và `spawn_replay_demand` (GAML) tạo xe ở đầu nhánh vào **bất kể có xe đang đứng ở đó hay không**. Hệ quả:
- Khi ùn tới đầu nhánh, xe mới chồng lên xe cũ (bảng 2.2: 195 mẫu chồng nhau ở 8.000 xe/h).
- README gốc ghi “xe chưa vào được nhánh nằm trong hàng đợi ngoài mạng (tính vào R1, R2, R4)”, nhưng `lvl_env.py:78,81` đặt cứng `external_wait_integral=0.0` và `external_all_time_integral=0.0`. **Tài liệu không đúng với code.** R2 của LVL thực chất chỉ còn thời gian dừng trong mạng.

Mô hình legacy (`python_env.py`) có xử lý đúng việc này (`_pending`, `source_clearance`). Nên mang cơ chế đó sang LVL.

### 4.4 Lane không gắn với hướng rẽ
Làn ban đầu được bốc ngẫu nhiên đều trên mọi sublane, bất kể xe rẽ trái hay rẽ phải. MOBIL có thiên hướng về bên phải (`+0.25` nếu lane nhỏ hơn) nhưng không có động cơ chuyển sang làn phù hợp với hướng rẽ. Khi vào connector, làn chỉ bị kẹp `min(lane, num_lanes-occupied)`. Kết quả là xe rẽ trái có thể đi từ sublane sát bó vỉa phải. Điều này làm sai phân bố hàng chờ theo làn, đúng thứ mà state spatial muốn học.

### 4.5 Nhu cầu và hành vi chưa hiệu chỉnh
- 2.400 xe/h/nhánh, 85% xe máy, 20/20% rẽ, tốc độ mong muốn `gauss(40/50, 5)`, tham số IDM (a=2.0/2.5, b=3.0, T=0.6/1.2 s, s0=0.3/1.0 m) đều là giả định.
- Chỉ có 2 loại xe (xe máy, ô tô), không có xe buýt (trong khi nhánh LVL có làn BRT), xe tải, người đi bộ, xe quay đầu.
- Lưu lượng không đổi theo thời gian (không có cao điểm), các nhánh như nhau.

Một nghiên cứu lớn cần ít nhất: đếm xe theo hướng rẽ theo 15 phút, đo độ dài hàng chờ hoặc thời gian qua nút từ video, rồi **hiệu chỉnh** tham số (IDM, khe chấp nhận, lưu lượng bão hòa) và **kiểm chứng** trên một tập dữ liệu khác.

---

## 5. Những chỗ chưa hợp lý: code của phần chuyển sang Python và env RL

### 5.1 `lvl_env.py`: state spatial không đúng như mô tả
README và đề xuất mô tả 6 kênh: C0–C2 là tỷ phần footprint theo loại xe, C3 là tốc độ dọc trung bình, C4 là vận tốc ngang, C5 là mặt nạ làn hợp lệ, ô dọc 8 m. Trong code (`lvl_env.py:52-61`):

| Kênh | Mô tả trong tài liệu | Code đang làm |
|---|---|---|
| C0 | footprint xe máy | ô có xe (0/1), mọi loại xe |
| C1 | footprint ô tô | tỷ lệ tốc độ của **xe cuối cùng ghi vào ô** (ghi đè, không trung bình) |
| C2 | footprint xe nặng | nhánh đang đỏ (0/1) |
| C3 | tốc độ dọc | là ô tô (0/1) |
| C4 | vận tốc ngang | **hằng số +1 cho NE/SW, −1 cho SE/NW**, không mang thông tin động |
| C5 | mặt nạ làn hợp lệ | lại là ô có xe (0/1), trùng C0 |

Ngoài ra ô dọc là **6 m** (`/ 6.0`), không phải 8 m; 20 ô chỉ phủ 120 m. Mặt nạ làn (NW chỉ có 14 sublane) không có trong state. Đây không phải lỗi làm sập chương trình, nhưng nếu báo cáo mô tả state theo tài liệu thì **mô tả sẽ sai với cái agent thực sự nhìn thấy**. Đây là rủi ro lớn khi bảo vệ hoặc khi bài báo bị phản biện.

### 5.2 `lvl_env.py`: biến toàn cục mất thông tin
- `green_cycles = cycle % 50` luôn bằng **0 tại mọi tick quyết định** (quyết định xảy ra đúng bội của 50). Agent không biết pha hiện tại đã xanh bao lâu.
- `pressure` chỉ là tổng hàng chờ đầu vào, không trừ đầu ra; hai thành phần cho action 1/3 luôn bằng 0. Gọi là “Max Pressure” cho nhánh Hybrid của LVL là chưa chính xác: thực chất là chọn theo hàng chờ dài nhất.
- `average_waiting_time`, `max_waiting_time`, `average_travel_time` trả về `NaN`. Thông thường train.py có kiểm tra `isfinite` khi hiển thị, nhưng mọi bảng so sánh hoặc biểu đồ dùng các cột này sẽ rỗng. **Với LVL hiện không có chỉ số thời gian chờ hay thời gian di chuyển của từng xe**, trong khi đó là chỉ số chính mà các bài báo điều khiển đèn thường báo cáo.
- `decision_count = cycle // 50` là số ước lượng, không phải số quyết định thật.

### 5.3 `lvl_env.py`: action space LVL bị thu hẹp và tài liệu mâu thuẫn
- Code: `MASK = [True, False, True, False]`, chỉ có 2 pha (LVL / HĐT), mỗi pha cho cả thẳng, trái, phải cùng xanh.
- `LeVanLuongIntersection/README.md`, mục “Train RL”, vẫn mô tả 4 action, rẽ trái được bảo vệ, vàng 3 s, đỏ toàn phần tới khi nút trống, bước 0,1 s. README chính lại ghi 2 pha, đỏ toàn phần 15 tick, bước 0,2 s.
- Chính sách 2 pha với chu kỳ quyết định cố định 10 s là bài toán RL rất hẹp: agent chỉ chọn “giữ” hoặc “đổi” mỗi 10 s. Baseline cố định 50/40 s đã gần tối ưu cho nút đối xứng như hiện tại, nên khó chứng minh RL hơn hẳn.

### 5.4 `_LVLCommon.step`: vá đơn vị thời gian bằng nhân hệ số
`info["elapsed_seconds"] *= DT / 0.1` sửa giá trị mà `GamaTrafficEnv.step` tính cứng `advanced * .1`. Hiện cho kết quả đúng, nhưng `train.py` cũng hardcode `.2 if scenario == "lvl" else .1` ở ít nhất 3 chỗ. Nên để base class dùng `self.step_seconds`. Nếu sau này đổi bước mô phỏng, chỗ nào quên sửa sẽ làm sai discount theo thời gian mà không báo lỗi.

### 5.5 `python_sim.py`: các điểm kỹ thuật khác
- **Lỗi encoding trên Windows** (mục 2.1).
- `LVLPythonEnv.reset` dùng `seed or 1`, nên seed 0 bị đổi thành 1. Ở chế độ replay, RNG không được dùng nên hiện không ảnh hưởng.
- `_free_idm_cache` lấy khóa theo `id(vehicle)`. Hiện an toàn vì xe chỉ bị xóa cuối `_move`, nhưng dễ vỡ nếu đổi thứ tự xóa.
- Mọi lớp mô phỏng viết bằng vòng lặp Python thuần, nên **chậm hơn GAMA** (README: khoảng 22 s so với khoảng 13 s mỗi episode). Lợi ích duy nhất của Python về tốc độ là chạy nhiều worker không cần JVM. README cũng chỉ đo được tăng tốc 1,24 lần với 2 worker. Một nghiên cứu lớn cần hàng triệu bước, nên tốc độ hiện tại là một nút thắt. Nếu mục tiêu là tốc độ, nên vector hóa (NumPy theo làn) hoặc dùng SUMO/CityFlow.

### 5.6 `python_env.py` (legacy): lỗi và parity chưa đạt
- **Lỗi**: `_phase_wait` không bao giờ được cập nhật (luôn `[0,0,0,0]`), trong khi GAML (`intersection-dqn-python.gaml:278-284`) tăng mỗi tick và ép phục vụ pha đợi quá 1.200 tick. Khi bật `dqn_service_guard`, Python **không bao giờ ép phục vụ pha bị bỏ đói**, nên action mask khác GAMA.
- Hiệu năng: `_choose_lanes` gọi `self._leader_gaps()` (dựng lại và sắp xếp toàn bộ nhóm xe) **bên trong vòng lặp từng xe** (`python_env.py:507`), độ phức tạp khoảng O(n² log n) mỗi tick.
- Xe trên các connector khác hướng cũng không thấy nhau (nhóm theo `(origin, maneuver)`).
- Nhiều hằng số “lấy từ trace” như `FLYOVER_PORTAL_CLEARANCE = 66.0` thay cho 70 m của GAML, hay các hệ số `2.5`, `3.0` trong phanh đèn đỏ. Đây là hiệu chỉnh cho khớp GAMA chứ không có cơ sở vật lý. Kiểu “khớp đường cong” này dễ vỡ khi đổi demand.
- README tự ghi: ở 4.000 tick, lệch 42,5% số xe nhận, 38,7% tốc độ trung bình, lịch pha trôi, **không đạt** ngưỡng 10%. Vì vậy **không nên dùng `--scenario legacy --backend python` cho bất kỳ kết quả nào**.

### 5.7 Tài liệu tham chiếu không tồn tại
README trỏ tới `reports/lvl_python_gama_parity_current_20261110.md`, `reports/python_parity_20261109/…`, `reports/python_parity_20261110/`, `reports/research.md`, `audits/…`. Commit `1cc5b61` **xóa cả thư mục `reports/`**, còn `audits/` không có trong repo. Các con số parity trong README (4.066/4.054/4.117 xe, MAE reward 0,19, 18,56% lệch làn) hiện **không kiểm chứng được** từ repo. Hội đồng hoặc người phản biện sẽ hỏi bằng chứng.

### 5.8 Phương pháp đối chiếu
- Đối chiếu chủ yếu bằng **tổng số xe qua cuối lượt**. Đây là chỉ số rất dễ khớp khi nút không bão hòa (mục 2.2): gần như mọi xe vào đều ra, nên chỉ số này gần như chỉ kiểm tra lịch phát xe, không kiểm tra động học.
- Nên đối chiếu thêm: phân bố thời gian qua nút từng xe, độ dài hàng chờ theo nhánh theo thời gian, lưu lượng bão hòa sau xanh (xe/s/sublane), thời gian mất khi khởi động. Nên dùng kiểm định phân phối (KS test) trên nhiều seed, không chỉ seed 1–3.
- Parity cuối cùng cần chứng minh là **thứ hạng chính sách giống nhau**: train trên Python, đánh giá trên GAMA, và ngược lại. README thừa nhận chưa làm phần này.

---

## 6. Có đủ cho nghiên cứu lớn không?

**Dùng được ngay cho:**
- Phát triển, gỡ lỗi thuật toán (Rainbow, 3DQN + pressure), thử nhanh reward R1–R4, ablation kiến trúc mạng.
- Một báo cáo hoặc khóa luận ở mức “đề xuất phương pháp + thử nghiệm trên mô phỏng giả định”, nếu nói rõ các giả định ở mục 4.

**Chưa dùng được cho:**
- Kết luận định lượng về nút Lê Văn Lương thật.
- So sánh “RL tốt hơn Fixed-time/Max-Pressure X%” ở mức bài báo hoặc hội nghị có phản biện: nút không bão hòa, không có xung đột, chỉ 2 pha, và chỉ số chờ từng xe không có.
- Kết luận về an toàn (TTC/PET): xe đi xuyên nhau và phanh 8g làm mọi chỉ số an toàn vô nghĩa.

### Lộ trình đề xuất (theo thứ tự ưu tiên)

| # | Việc | Lý do | Công sức ước lượng |
|---|---|---|---|
| 1 | Sửa encoding UTF-8, sửa `_phase_wait`, khôi phục hoặc commit lại thư mục `reports/` chứa bằng chứng parity | Rẻ, gỡ chỗ chặn ngay | Vài giờ |
| 2 | Sửa `lvl_env._observation` cho khớp tài liệu (6 kênh thật, ô 8 m, mặt nạ làn), sửa `green_cycles`, thêm thời gian chờ/thời gian di chuyển từng xe | State sai làm mọi kết quả RL khó giải thích | 1–2 ngày |
| 3 | Hàng đợi ngoài mạng và điều kiện chặn khi phát xe cho LVL | Đúng với tài liệu R1/R2, tránh xe chồng nhau | 1 ngày |
| 4 | Xung đột trong nút: điểm xung đột + rẽ trái nhường đường + pha vàng + đỏ toàn phần đợi nút trống | Quan trọng nhất về độ thực; làm ở Python dễ hơn GAMA | 1–2 tuần |
| 5 | Giới hạn phanh, đổi làn ±1 có vận tốc ngang, chọn làn theo hướng rẽ | Hành vi xe máy hợp lý | 1 tuần |
| 6 | Mở 4 pha có rẽ trái bảo vệ, hoặc thêm lựa chọn độ dài xanh | Bài toán RL đủ khó để có ý nghĩa | Vài ngày |
| 7 | Thu số liệu thực (đếm theo hướng rẽ, video hàng chờ), hiệu chỉnh và kiểm chứng | Bắt buộc nếu muốn kết luận về nút thật | Vài tuần (gồm thực địa) |
| 8 | Đối chiếu ở mức phân phối và thứ hạng chính sách giữa Python/GAMA trên ≥ 10 seed | Chứng minh Python thay được GAMA | Vài ngày |

**Lưu ý chiến lược**: sau bước 4–5, bản Python sẽ **vượt** GAMA `advanced_driving` về độ thực, và GAMA không còn là “chuẩn vàng” để đối chiếu. Lúc đó nên đổi chuẩn tham chiếu thành **dữ liệu thực** (hoặc SUMO đã hiệu chỉnh), thay vì cố khớp từng xe với GAMA. Nếu nhóm không định làm bước 4–7, cân nhắc chuyển sang SUMO (có sẵn xung đột trong nút, nhường đường, mô hình sublane cho xe máy, TraCI/libsumo để train RL, chạy nhanh hơn nhiều). Như vậy nhóm khỏi phải tự viết lại những thứ đó.

---

## 7. Phạm vi chưa kiểm tra

- Không chạy GAMA, nên không tự kiểm lại các con số parity trong README.
- Không chạy một lần train đầy đủ; chỉ đọc luồng `train.py`/`evaluate.py` và chạy test.
- Không đọc chi tiết `observer.html`, `lvl_observer_server.py` và các script `prototype_*`, `probe_*` (công cụ quan sát và gỡ lỗi, không ảnh hưởng kết quả).
- Script dò ở mục 2.2 chỉ dùng seed 1. Các con số dùng để thấy xu hướng, không phải thống kê.
