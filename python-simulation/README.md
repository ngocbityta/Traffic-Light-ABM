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
