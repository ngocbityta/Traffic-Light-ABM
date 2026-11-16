# Python simulation

Mô phỏng độc lập ngã tư Lê Văn Lương – Hoàng Đạo Thúy nằm trong
`simulation/python_simulation/`. Logic Python dùng GeoJSON trong
`simulation/python_simulation/includes/`; model GAML được quản lý riêng ở
`gama-simulation/`.

## Chạy mô phỏng

```bash
python3 python-simulation/simulation/python_simulation/python_sim.py \
  --seed 1 --cycles 4000 --output train/results/lvl_python.csv
```

## Giao diện quan sát

```bash
python3 scripts/lvl_observer_server.py --port 8765 --seed 1
```

Mở `http://127.0.0.1:8765/`. Trang hỗ trợ chạy, tạm dừng, bước mô phỏng,
đặt lại seed và xem thông tin xe. Giao diện nằm trong `ui/`; script dựng bố cục
và GeoJSON nằm trong `scripts/`.

Các kiểm thử Python được đặt trong `tests/`.

## Cấu hình dùng chung cho UI và train

Mô phỏng Python mặc định dùng `junction_model="negotiation"`. Có thể dùng cùng
một file JSON cho giao diện và huấn luyện:

```json
{
  "junction_model": "negotiation",
  "yellow": 4.0,
  "all_red": 2.0,
  "vehicle_params": {"motorbike": {"headway": 0.75}},
  "startup_params": {"reaction_delay": 0.7}
}
```

Truyền `--sim-config config.json` cho `scripts/lvl_observer_server.py` và
`python -m train.logic.train`. Cũng có thể truyền trực tiếp JSON kết quả của
`calibrate_discharge.py` (trường `kwargs`). Training lưu cấu hình đã áp dụng
trong `training_config.json` và `training_summary.json`; evaluate đọc lại từ
summary, hoặc nhận `--sim-config` để ghi đè. Evaluate ghi cấu hình thực dùng
ra file `<output>.sim_config.json`; file này có thể nạp lại bằng `--sim-config`.

Trong chế độ `safe`, xe rẽ phải được đi khi đèn đỏ nhưng phải được cấp quyền
vào nút và nhường xe đi thẳng tại vùng nhập làn. Bộ đếm
`permitted_right_on_red_entries` tách các lượt rẽ phải hợp lệ khỏi
`entries_during_red`.

## Hàng chờ và chọn làn

Xe đi thẳng có thể xuất hiện ở bất kỳ sublane hợp lệ nào trên đường vào,
bao gồm vùng làn rẽ phải. Khi tiến gần nút, xe phải nhập vào vùng làn đi
thẳng trước khi tới hàng đầu chờ đèn; nếu chưa có khe hở an toàn, xe giữ
khoảng cách ở vùng đệm phía trước vạch dừng. Demand ghép cặp và model GAMA
giữ nguyên làn xuất hiện đã yêu cầu.
Ở chế độ Python `safe`, mỗi xe có một khoảng lùi dừng cố định theo seed
(xe máy 0–1,8 m; ô tô 0–1,2 m). Xe máy tiến vào hàng chờ đỏ có thể lọc sang
sublane kế bên khi có đủ khoảng trống trước, sau và bên hông, đồng thời còn
đủ đường để hoàn tất đổi làn trước vạch dừng. Chế độ `gama_2025_6_4` giữ
quy tắc dừng của GAMA để phục vụ đối chiếu.

Quy tắc lọc làn dựa trên quan sát về xe máy trong hàng chờ tại Hà Nội và
TP.HCM trong [nghiên cứu của Minh và cộng sự](https://onlinelibrary.wiley.com/doi/10.1002/atr.144)
và [nghiên cứu về lọc làn ở nút giao có đèn](https://onlinelibrary.wiley.com/doi/10.1155/2022/5662117).
Cách chia sublane và giữ khoảng hở bên hông tham khảo
[tài liệu SUMO](https://eclipse.dev/sumo/docs/Simulation/SublaneModel.html).
Các khoảng lùi và ngưỡng lọc hiện là giả định mô phỏng, chưa được hiệu chỉnh
bằng quỹ đạo xe đo tại ngã tư này.

## Đèn vàng, quyết định dừng/đi (13/11/2026)

`LeVanLuongSimulator(yellow=4.0)` (mặc định 4 s) lấy 4 s cuối mỗi pha xanh làm
đèn vàng; độ dài chu kỳ và thời điểm bắt đầu xanh không đổi. Có thể truyền
`yellow=0` cho các benchmark cần kế hoạch cũ. Ở chế độ train, mỗi lần đổi pha
có 4 s vàng rồi tới 2 s đỏ toàn phần.

Khi vàng bật, mỗi xe chưa qua vạch quyết định dừng/đi một lần theo logit Point
Data Model của Chauhan, Dhamaniya & Arkatkar (2022, TRF 89:222–235):
U = 4,518 + 0,106·v[m/s] − 0,083·x[m] − 1,068·ô tô − 0,923·làn giữa −
1,049·làn sát dải phân cách. Xe không thể dừng với gia tốc ≤ 6 m/s² luôn đi.
Quyết định "đi" còn hiệu lực trong đỏ toàn phần và hết hiệu lực khi pha kế
tiếp xanh. Số liệu Ấn Độ; việc áp dụng cho Hà Nội là giả định.
Xe ở quá xa để tới nút trước pha xanh đối hướng phải chọn dừng.

## Tham số hành vi và hiệu chỉnh

`vehicle_params={"motorbike": {"headway": 0.8}, "car": {"gap": 1.5}}` và
`startup_params={"reaction_delay": 1.0, "discharge_headway": 0.9}` thay giá trị
mặc định (`DEFAULT_VEHICLE_PARAMS`). `scripts/calibrate_discharge.py` đo lưu
lượng bão hòa và thời gian tổn thất khởi động, và tìm tham số khớp với số đo
thực địa bạn cung cấp (`--targets targets.json`).

## Chỉ số an toàn thay thế (SSM)

`scripts/ssm_report.py --seed 1 --ticks 3000 --save run.npz` ghi quỹ đạo
(`python_simulation/trajectory.py`) và tính offline
(`python_simulation/ssm.py`): TTC, MTTC (có gia tốc), time gap, DRAC,
TET/TIT, PET tại các điểm xung đột cấp sublane, TTC 2D trong nút và phanh gấp.

## Điều kiện để train (15/11/2026)

Các thay đổi sau áp dụng cho chế độ `safe`; chế độ `gama_2025_6_4` giữ nguyên
để đối chiếu với GAMA.

- **Xanh tối thiểu.** `min_green=10.0` (s). `sim.action_mask()` chỉ cho đổi pha
  khi pha đang chạy đã xanh đủ lâu; `set_action` với action bị che được coi là
  giữ pha (đếm trong `masked_actions`). Với quyết định mỗi 10 s và 6 s
  vàng + đỏ toàn phần, agent đổi pha nhanh nhất mỗi 20 s.
- **Giữ trống lòng nút.** `negotiation_params={"box_clear": true}` (mặc định):
  xe của pha mới chỉ chờ ở vạch dừng khi nó thật sự phải nhường một xe sót
  của pha cũ, tức xe sót chưa kịp rời vùng xung đột trước lúc xe này tới nơi.
  Xe sót sắp ra khỏi vùng xung đột không giữ ai lại. Khi một xe đã vào nút và
  đang chờ xe sót, chỉ dải ngang của xe đó bị đóng, các sublane khác vẫn mở.
  Đặt `false` để trả lại hành vi lách qua xe sót.
- **Xe trước qua mối vạch dừng.** Một xe chỉ là "xe trước" khi thân nó nằm
  cùng dải ngang (tính theo bề rộng thân xe, kể cả dải nó đang dịch sang). Xe
  ở sublane bên cạnh, hoặc xe đi song song trên connector tách hướng, không
  giữ xe ở vạch.
- **Lưới an toàn chống kẹt.** `jam_timeout=120.0` (s): xe đứng yên trên
  connector lâu hơn ngưỡng bị gỡ khỏi mô phỏng và đếm trong `jam_removed`
  (tương tự time-to-teleport của SUMO). Một lần chạy lành mạnh có
  `jam_removed == 0`; khác 0 nghĩa là có kẹt cần điều tra.
- **Giới hạn động học.** `curve_lat_accel=3.0` m/s² giới hạn tốc độ trên cung
  rẽ (v ≤ √(a·R) theo đường đi thực của sublane), có giảm tốc trước cua với
  `curve_decel=2.0` m/s²; đặt 0 để tắt. `max_decel=7.5` m/s² là giới hạn phanh
  của luật bám xe; các lần bảo vệ vị trí đòi phanh gấp hơn được đếm trong
  `hard_brake_events`. Tốc độ là tốc độ thực trên đường đi của xe: tiến độ
  trên đường tham chiếu được quy đổi theo độ dài cung của sublane.
- **Sinh xe.** Xe vào mô hình với tốc độ mong muốn, giảm theo khoảng cách tới
  xe phía trước cùng dải ngang.
- **Tất định.** Cùng seed cho cùng kết quả ở mọi tiến trình, không phụ thuộc
  `PYTHONHASHSEED`.

Các tham số trên nhận qua `--sim-config` (xem `sim_config.py`). Bộ kiểm thử
`tests/test_train_readiness_20261115.py` và `tests/test_green_yield_20261116.py`
giữ các điều kiện này.
