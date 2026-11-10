# Ngã tư Lê Văn Lương × Hoàng Đạo Thúy / Hoàng Minh Giám (GAMA)

Ngã tư cạnh Hà Nội Center Point (27 Lê Văn Lương, Thanh Xuân). Model GAMA 2025.06 dùng `advanced_driving`:
mỗi chiều của mỗi nhánh là một `road` với `num_lanes` = số sublane rộng **0,8 m**; xe máy chiếm 1 sublane, ô tô 2 sublane.
Làn BRT được tính vào số sublane (xe máy được dùng).

## Chạy

1. GAMA → *File › Import › Existing Projects into Workspace* → chọn thư mục này.
2. Mở `models/LeVanLuong.gaml` → chạy experiment **main**.
3. Monitor "X vào: sublane / hàng đầu max" = số sublane của nhánh / số sublane bị xe chiếm cùng lúc ở hàng đầu (≤ 1,5 m trước vạch dừng).

### Bản Python độc lập

`python_sim.py` đọc trực tiếp 20 đoạn đường từ `includes/roads.geojson`, dùng cùng tham số phát xe và lịch đèn 0,2 s/tick của GAML. Chuyển động dọc dùng IDM và đổi làn dùng MOBIL được port từ GAMA 2025.6.4. Chạy không cần GAMA:

```bash
python3 LeVanLuongIntersection/python_sim.py --seed 1 --cycles 4000 --output comparison_results/lvl_python.csv
```

Quan sát mô phỏng Python trên trình duyệt (không cần GAMA):

```bash
dqn_py/venv/bin/python scripts/lvl_observer_server.py --port 8765 --seed 1
```

Mở `http://127.0.0.1:8765/`. Trang có nút Chạy, Tạm dừng, Bước 1/50 tick, Đặt lại theo seed và tốc độ phát 1/5/20 tick mỗi lần cập nhật. Chọn **Fixed** để xem lịch đèn cố định hoặc **Train** rồi đặt lại để thử quyết định pha LVL/HDT mỗi 50 tick; mô phỏng sẽ dừng ở tick chờ action (có thể dùng phím 0/2). Di chuột hoặc bấm vào xe để xem ID, road, làn và tốc độ. Xe được đặt trên 20 road từ GeoJSON theo sublane, với màu đèn theo bốn nhánh.

Để so cùng nhu cầu, script benchmark sinh CSV lịch xe gồm thời điểm, nhánh, loại xe, hướng rẽ, làn và tốc độ ban đầu rồi nạp vào cả hai backend:

Đối chiếu số xe đang chạy, pha đèn và số sublane hàng đầu với experiment `python_parity`:

```bash
dqn_py/venv/bin/python scripts/compare_lvl_python_gama.py --start-server --seed 1 --horizon 1000 --output comparison_results/lvl_parity.csv
```

Seed 1 ở 1.000 tick với 540 xe đầu vào đã đồng bộ đúng tick phát xe: Python/GAMA cho 396/397 xe hoàn thành và 144/143 xe còn trên mạng; số sublane hàng đầu tối đa khớp cả bốn nhánh. Làn và tốc độ của từng xe vẫn lệch, nên các kết quả chính sách cần đối chiếu GAMA trước khi dùng nghiên cứu. Kết quả chi tiết nằm trong `reports/python_parity_20261109/lvl_fixed_verified_1000_seed1.csv`.

Benchmark GAMA ↔ Python mới nhất trên 8.000 tick, seed 1/2/3 và cùng demand CSV cho số xe qua cuối lượt khớp đúng cả ba seed: 4.066, 4.054 và 4.117 xe; pha, số xe tạo và bốn chỉ số hàng đầu cuối lượt cũng khớp. Tại các mốc 400 tick, sai khác số xe qua lớn nhất là 7/8/8 xe. Trace vi mô 200 tick vẫn có 18,56% cặp xe-tick lệch làn, nên cần tiếp tục đối chiếu IDM/MOBIL trước khi coi hai backend tương đương từng xe. Python 3.11 mất khoảng 21,6–22,1 giây phần step mỗi episode, GAMA mất 13,3–13,4 giây. Xem `reports/lvl_python_gama_parity_current_20261110.md` và CSV trong `reports/python_parity_20261110/`.

### Train cùng cấu hình trên Python và GAMA

```bash
dqn_py/venv/bin/python dqn_py/train.py --scenario lvl --backend python --algorithm hybrid --reward-mode r1 --total-steps 1000 --output-dir dqn_py/run/lvl_trial
dqn_py/venv/bin/python scripts/compare_lvl_training_envs.py --seed 1 --horizon 500 --output comparison_results/lvl_training_parity.csv
```

Train mặc định dùng hai pha hợp lệ, quyết định mỗi 50 tick và đỏ toàn phần 15 tick khi đổi pha. Bản Python và GAMA chia sẻ lịch nhu cầu CSV và mã dựng observation/reward ở `dqn_py/lvl_env.py`. Benchmark chính sách queue độc lập trên seed 1/2/3, 8.000 tick: cả ba có 159/159 quyết định và pha khớp; số xe qua cuối lượt lệch lần lượt 0/1/2 xe, reward R1 MAE 0,191/0,178/0,204 mỗi quyết định và spatial observation MAE khoảng 0,022. Đây là parity rollout, chưa chứng minh quá trình cập nhật trọng số khi train hoàn toàn tương đương. Xem `reports/lvl_python_gama_parity_current_20261110.md`.

## Số đo (ảnh vệ tinh, sai số ước tính ±0,5 m nếu không ghi khác)

| Nhánh | Phương vị | Chiều vào: rộng → sublane | Chiều ra: rộng → sublane | Dải phân cách |
|---|---|---|---|---|
| NE – Lê Văn Lương (về Láng Hạ) | 53,4° | 13,3 m → **16** (±0,7) | 12,7 m → **15** | 7,8 m (lát gạch, BRT) |
| SE – Hoàng Đạo Thúy (về Nhân Hòa) | 128,5° | 13,5 m → **16** | 9,2 m → **11** | 2,5 m (bồn cây) |
| SW – Lê Văn Lương (về Khuất Duy Tiến) | 232,8° | 12,8 m → **16** ⚠ | 12,4 m → **15** | 7,9 m (cỏ) |
| NW – Hoàng Minh Giám (về Trung Hòa) | 316,4° | 11,9 m → **14** | 9,9 m → **12** (±1,0) | 2,6 m (bồn cây) |

- Góc giữa các nhánh kề nhau: NE–SE 75,1°, SE–SW 104,3°, SW–NW 83,6°, NW–NE 97,0°. Lê Văn Lương gần thẳng; trục Hoàng Đạo Thúy–Hoàng Minh Giám gãy ~8° và lệch tâm tại nút.
- Vùng xung đột (giữa các vạch dừng và mép ngoài vạch qua đường): **1919 m²**. Bó vỉa góc bán kính 9 m (góc Đông đã đối chiếu ảnh, 3 góc còn lại giả định).
- ⚠ SW vào = 12,8 m nằm đúng ngưỡng 16 × 0,8: hụt 5 cm là còn 15 sublane — nên đo thực địa nhánh này.
- NW ra 9,9 m bị tán cây che, suy ra từ khoảng cách tim 2 chiều trên OpenStreetMap.

**Nguồn:** Esri World Imagery 11/2025 (0,28 m/px) cho NE, SW, NW; Esri Wayback 06/2023 và 12/2024 cho SE và kiểm tra chéo
(nhà cao tầng che các nhánh khác nhau trên từng ảnh); tim đường và phương vị từ OpenStreetMap. Ảnh đối chiếu: `docs/doi_chieu_ve_tinh_2025.png`
(nền: Esri World Imagery — Esri, Maxar, Earthstar Geographics; viền tím = bó vỉa, xanh lá = dải phân cách, vàng = vạch dừng, đỏ = vùng xung đột).
Ảnh mô phỏng GAMA lúc t = 46 s: `docs/gama_t46s.png`. Ảnh chụp thực tế: `nga_tu.png`.

## Sửa số đo

Mọi số đo nằm trong `ARMS` / `CORNER_R` ở đầu `scripts/build_layout.py`. Sửa xong chạy lại:

```bash
python scripts/build_layout.py
```

Script in bảng sublane và ghi lại `includes/roads.geojson`, `includes/areas.geojson` (UTM 48N, mét).

## Giả định chưa có số liệu

- Đèn 2 pha cố định (xanh LVL 50 s, xanh HĐT 40 s, đỏ toàn phần 3 s), lưu lượng 2400 xe/h/nhánh, 85% xe máy, rẽ trái/phải 20%/20% — đều là tham số trong experiment.
- Xe trên các connector cắt nhau trong vùng xung đột **không nhìn thấy nhau** (advanced_driving chỉ xử lý xe cùng road); đèn 2 pha tránh phần lớn xung đột, riêng rẽ trái với đi thẳng ngược chiều thì chưa.
- Không có quay đầu, không có người đi bộ.
