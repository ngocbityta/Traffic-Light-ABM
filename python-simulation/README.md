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

### Tăng tốc trên CPython

Để biên dịch phần di chuyển, nhường đường và hình học từ cùng mã Python, chạy
từ thư mục gốc của repo với trình biên dịch C và môi trường Python dùng cho mô phỏng:

```bash
python3 -m pip install "Cython==3.3.1" setuptools
python3 python-simulation/scripts/build_native.py
```

Lần khởi động kế tiếp tự dùng bản biên dịch phù hợp với mã nguồn, kiến trúc máy
và Python hiện tại. Sau khi sửa các mô-đun này, chạy lại lệnh build. Đặt
`LVL_PURE_PYTHON=1` để chạy đối chiếu bằng Python. Cython chỉ cần lúc build.

Nếu train bằng môi trường của repo, build bằng đúng trình thông dịch đó để cả
sáu worker nạp được extension tương ứng:

```bash
train/venv/bin/python -m pip install "Cython==3.3.1" setuptools
train/venv/bin/python python-simulation/scripts/build_native.py
```

Bản build mặc định không có hook profiling của Cython (nhanh hơn khoảng 9 %).
Để `cProfile` thấy được các hàm đã biên dịch, build biến thể profiling và chạy
với cùng biến môi trường:

```bash
LVL_NATIVE_PROFILE=1 python3 python-simulation/scripts/build_native.py
```

Hai biến thể nằm ở hai thư mục build khác nhau; khi chỉ có một biến thể, mô
phỏng nạp biến thể đó.

Đo riêng 1.000 tick với cấu hình mặc định, seed 1 và `cProfile`
(cần `LVL_NATIVE_PROFILE=1` nếu đang dùng bản biên dịch):

```bash
PYTHONPATH=python-simulation/simulation python3 -c 'import cProfile, pstats; from python_simulation.python_sim import LeVanLuongSimulator; sim = LeVanLuongSimulator(seed=1); p = cProfile.Profile(); p.runcall(sim.step, 1000); pstats.Stats(p).strip_dirs().sort_stats("cumtime").print_stats(10)'
```

Các hàm như `_move`, `_neighbors` và `decide` vẫn xuất hiện trong profile.

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
  "heavy_ratio": 0.03,
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

## Chấp nhận khe và lách ngang trong nút (16/11/2026)

- **Chấp nhận khe theo logit** (`negotiation_params={"gap_model": "logit"}`,
  mặc định; `"sumo"` trả lại luật impatience kiểu SUMO). Xe rẽ trái gặp xe đi
  thẳng đang xanh quyết định một lần cho mỗi khe theo Vu & Shimizu (2010):
  xe máy V = 2,449·TG − 0,045·EWT + 0,198·MC1 − 2,257, ô tô V = 2,887·TG +
  0,127·EWT + 0,165·V1 − 4,225; TG = khe dọc − khe ngang (s), EWT thời gian
  chờ, MC1 số xe máy trong nhóm (xe cách nhau ≤ 4 m là một nhóm). Số ngẫu nhiên
  cố định theo (seed, xe, khe). Xe theo sau trong nhóm đi theo xe dẫn đầu.
  `neg.gap_events` ghi quyết định đầu tiên của mỗi khe để so với bài báo.
- **Lách ngang của xe đi thẳng** (`"evade": true`). Xe đi thẳng trong nút
  đang phải phanh vì một xe rẽ trái có thể dịch 1–3 sublane để **cắt đuôi**
  (đi sau đuôi xe rẽ trái) hoặc **lách lên trước** (đi trước, buộc xe rẽ trái
  chưa cam kết phanh nhẹ, ≤ 0,5·a_max). Chọn dải lệch ít nhất vẫn đi tiếp
  được (Trinh và cs. 2018). Hướng lái giới hạn ±30° với xe máy (Trinh 2018),
  ±0,2 rad với ô tô (Kanagaraj & Treiber 2018). Mỗi dải mới được kiểm tra chỗ
  trống trên connector và thời gian với mọi xe xung đột. Xác suất sẵn lòng
  (`evade_cut_*`, `evade_front_*`) đã hiệu chỉnh để tỷ lệ thực hiện khớp
  Vu & Shimizu. Kết quả đo (seed 1–3, 3000 tick): xe máy dẫn đầu cắt đuôi
  17 % và lách lên trước 5 % (bài báo: 20 % và 5 %), ô tô 3 % và 3 % (bài báo:
  5 % và 2 %; chỉ có 89 sự kiện ô tô). `adapter.evade_events` ghi từng lần lách.
- Kiểm tra va chạm ngang khi đổi làn giờ tính cả xe vừa qua mối nối
  connector → đường ra.

Kiểm thử: `tests/test_logit_evade_20261116.py`.

## Xe nặng (16/11/2026)

Mô phỏng có ba loại xe: xe máy, ô tô và **xe nặng** (gộp xe tải, xe buýt, xe
khách). Mã loại trong file demand (cột thứ ba): 0 xe máy, 1 ô tô, 2 xe nặng.
File demand cũ chỉ có 0/1 vẫn đọc được.

| Tham số | Xe nặng | Ô tô | Nguồn |
|---|---|---|---|
| Dài × rộng (m) | 9,0 × 2,45 | 4,5 × 1,7 | SUMO: truck 7,1 × 2,4, bus 12 × 2,5; trộn 59:41 theo số đếm `small_truck_bus` : `heavy_truck_bus` của nút Nguyễn Trãi – Khuất Duy Tiến |
| Số sublane chiếm | 3 | 2 | ⌈rộng / 1,05 m⌉, tương đương `num_lanes_occupied` của GAMA |
| a (m/s²) | 1,25 | 2,5 | SUMO `getDefaultAccel`: truck 1,3, bus 1,2 |
| b (m/s²) | 2,7 | 3,0 | SUMO `getDefaultDecel`: xe nặng 4,0 so với ô tô 4,5; cùng tỷ lệ áp lên b của ô tô |
| s0 (m) | 1,0 | 1,0 | SUMO dùng cùng `minGap` cho ô tô, truck, bus |
| T (s) | 1,5 | 1,2 | `time_headway` mặc định của driving skill GAMA |
| Tốc độ mong muốn | N(50; 2,5) km/h | N(50; 5) km/h | SUMO `speedFactor`: trung bình 1, độ lệch truck 0,05 so với ô tô 0,1 |
| Tỷ lệ trong dòng xe | 3 % | 12 % | 85 % xe máy giữ nguyên; 15 % còn lại chia theo tỷ lệ ô tô : xe nặng của số đếm trên |

Các tham số nằm ở `DEFAULT_VEHICLE_PARAMS` và `DEFAULT_HEAVY_RATIO` trong
`python_sim.py`. Ghi đè bằng `vehicle_params={"heavy": {...}}` và
`heavy_ratio`, qua `--sim-config` (khóa `heavy_ratio`), `--heavy-ratio` của
`python_sim.py`, `lvl_observer_server.py` và `ssm_report.py`.
`heavy_ratio=0` cho kết quả trùng từng bit với mô hình hai loại xe trước đây.

Hành vi riêng:
- **Đèn vàng:** logit Chauhan và cs. (2022) dùng hệ số "Bus & LCV" −1,471 của
  cùng bảng (bài báo ghi hệ số này không có ý nghĩa thống kê ở mức 5 %).
- **Trong nút:** xe nặng dùng luật chấp nhận khe và nhường của ô tô. Xe nặng
  không lách ngang (`evade_cut_heavy = evade_front_heavy = 0`), vì số liệu
  Vu & Shimizu chỉ có xe máy và ô tô dẫn đầu. Khi đếm chỗ chờ trong nút, một
  xe nặng tính bằng 4 xe máy.
- **Đổi làn:** thời gian dịch ngang dài hơn ô tô (hệ số 1,3, giả định).
- **Thân cứng một khối, hai trục (17/11):** xe nặng là một hình chữ nhật
  liền 9,0 × 2,45 m trên hai trục, theo xe thiết kế SU của AASHTO (xe tải đơn /
  buýt hai trục): trục trước cách đầu xe 1,2 m, trục sau cách đuôi 1,7 m,
  chiều dài cơ sở 6,1 m. Trục trước chạy đúng đường sublane; trục sau kéo theo
  kiểu xe đạp động học (kinematic single-track): trục sau luôn hướng về trục
  trước, khoảng cách hai trục giữ đúng 6,1 m, nên ở khúc rẽ trục sau đi lẹm
  vào phía trong cua (off-tracking R − √(R² − WB²), ~1,6 m ở bán kính rẽ
  phải). Vị trí trục sau được lưu từng tick (`_commit_axles`) và tích phân
  theo đường tractrix (`_axles`, bước 0,25 m). Đầu xe chìa ra phía ngoài cua.
  Thân xe, bộ chống chồng thân, đàm phán nút và giao diện đều dùng chung một
  khối này (`_rigid_pose`, `_body_polygons`, `_body_axis_points`). Bản
  16/11 chia xe thành 2 đoạn khớp nối; đã bỏ vì xe tải đơn và buýt thường
  không gập giữa thân.
- **Bán kính quay vòng:** bán kính đường tâm tối thiểu 11,6 m (SU-30: 38 ft).
  Rẽ phải ở nhánh nào không có sublane nào đủ rộng thì xe nặng được sinh với
  hướng đi thẳng (`heavy_turns_rerouted` trong snapshot; ở layout hiện tại là
  rẽ phải từ SE). Xe nặng rẽ phải chạy trên làn rộng nhất của connector
  (`_mapped_lane`) và lái về tâm làn ngay sau vạch dừng.
- **Tốc độ vào cua:** gia tốc ngang tối đa 1,5 m/s² cho xe nặng (ngưỡng khó
  chịu của hành khách đứng trên buýt, Nguyen và cs.), thay vì giá trị chung
  của ô tô; tốc độ hãm trước cua tính theo làn xe nặng sẽ thực sự chạy.
- **Độ chìa thân (`body_sweep`):** phần thân cứng ra ngoài dải sublane của
  đường đi (đầu xe phía ngoài cua, trục sau phía trong cua), tối đa 2,5 m.
  Độ chìa này cộng vào bề rộng trong các luật giữ khoảng cách ngang. Trong
  đàm phán nút, dải xung đột của xe nặng được dời vào phía trong cua và nới
  rộng đúng bằng phần lẹm trong và chìa ngoài của thân cứng (`_rigid_band`).
  Ô tô và xe máy không có độ chìa, giữ nguyên hành vi cũ.
- **Xe bên cạnh không phải xe trước:** ở mối vạch dừng, một xe có tâm còn nằm
  ngang thân xe nặng không được coi là xe dẫn (trước đây điểm dừng đặt sau
  đuôi xe đó, tức sau cả mũi xe nặng: xe nặng dừng từ 5 m/s trong một tick).
  Xe có hơn nửa thân đã vượt lên trước vẫn là xe dẫn, vì qua đoạn gấp khúc
  hai đường đi hội tụ.
- **Không thể dừng kịp ở vạch:** xe nặng đã được cho vào nút mà bị rút quyền
  (một xe khác vừa vào vùng xung đột phía trước) chỉ phanh gấp tối đa
  ~4,2 m/s². Nếu không dừng kịp trước vạch thì nó giữ quyền vào và phanh tới
  điểm dừng trước vùng xung đột, miễn là dừng kịp ở đó (như MSLink của SUMO).
- **Xe sau xe nặng** (`leading_heavy`): khi dự đoán quãng nhích của xe trước,
  nếu xe trước đang bám một xe nặng thì không tính quá khoảng trống thực của
  nó (xe nặng có thể dừng ngắn vì bị giữ ở nút hay đang đổi làn).
- **Mối vạch dừng:** xe trên nhánh vào bám theo một xe nặng vừa qua mối nối
  giữ thêm khoảng cách bằng độ chìa của xe nặng. `_resolve_seam_overlaps`
  chặn xe sau trước khi chạm đuôi xe nặng, chỉ trong quãng vừa đi của tick.
- **Hai xe cùng chờ nhập làn đứng song song** (`_release_mutual_merge`): hành
  lang đi thẳng chỉ có hai vị trí chồng nhau cho xe 3 sublane, nên hai xe nặng
  đứng cạnh nhau ở vùng đệm nhập làn chặn nhau mãi (nhánh SE không xả được ở
  10 % xe nặng). Sau 8 s, xe nhích trước hơn được đi tiếp qua vùng đệm, vượt
  hẳn thân xe kia rồi nhập làn; xe kia nhập sau.
- **Quãng nhích dự đoán của xe trước** (`_leader_advance`): xe trước chưa được
  phép qua vạch và sắp tới điểm dừng có thể phanh tới `max_decel` (khi đèn đổi
  pha). Xe sau không còn tính một bước nhích đầy đủ của nó, nên không bị đẩy
  lùi vài cm nữa.

Kết quả đo với thân cứng (17/11), kế hoạch cố định, 4 seed × 3000 tick
(600 s). Mốc là cùng seed với 0 % xe nặng (trùng từng bit với mô hình trước
đây):

| Chỉ số mỗi seed | 0 % xe nặng | 3 % xe nặng |
|---|---|---|
| Xe qua | 1454–1549 | 1450–1525 |
| Thời gian trong mạng trung bình (s): xe máy / ô tô / xe nặng | 43–44 / 42–45 / — | 46–50 / 46–54 / 59–61 |
| Gia tốc ngang xe nặng: p95 / lớn nhất (m/s²) | — | 1,46–1,49 / 1,94–2,01 |
| Cặp chồng thân có xe nặng | — | 0, 1, 2, 1 (sâu ≤ 0,16 m, 1–2 tick) |
| Cặp chồng thân mọi loại xe | 1–2 | 5–7 |
| Trong đó sâu > 0,1 m | 0–1 | 0–3 |
| Xe bị gỡ do kẹt / lần đẩy lùi | 0 / 0 | 0 / 0 |
| Lần phanh > 9 m/s² (mọi xe / xe nặng) | 15–24 / — | 35–45 / 5–9 |

Chế độ train, 3000 tick: 3 % ngẫu nhiên (seed 1) và đổi pha liên tục (seed 2)
qua 1498 / 1533 xe, 0 kẹt, 0 đẩy lùi. 10 % cố định (seed 1): 1380 xe, 0 kẹt,
2 lần đẩy lùi (lớn nhất 0,12 m).

Nhận xét:
- Cùng số xe đến, thay ô tô bằng xe nặng làm nút xả chậm hơn: lưu lượng một
  pha xanh giảm khoảng 10 % ở 3 % xe nặng. Ở nhu cầu 2400 xe/h mỗi nhánh nút
  gần bão hòa, nên 10 % xe nặng gây ùn nhẹ.
- Phanh gắt của xe nặng còn lại chủ yếu do xe máy hoặc ô tô cắt ngang ngay
  trước mũi xe nặng (khoảng cách 0,1–1,6 m) và do các chốt chặn vị trí ở mối
  nối, cùng cơ chế đã có với ô tô và xe máy ở 0 %.
- Seed 3 có một cặp **xe máy – xe máy** chồng 0,6 m (rẽ trái SW và đi thẳng
  NE, 5 tick, không có quyết định đàm phán giữa hai xe): không có xe nặng
  trong cặp, chưa rõ nguyên nhân.
- **Giữ ô trống khi có xe 3 sublane (chưa sửa):** vòng chờ "xe nặng sót pha →
  xe pha mới len vào → xe phụ chờ xe nặng" chỉ được phá bằng luật ép sau 8 s;
  thời gian xe đứng ở vạch khi đèn xanh vì luật này là 150–840 xe·s / 600 s
  (seed 1–4), so với 8–62 khi không có xe nặng. Đã có từ bản 16/11. Test
  `test_few_vehicles_stand_at_the_line_on_green` đo ở 0 % xe nặng.

Tạo demand cho model GAMA: `generate_demand(..., heavy_ratio=0)`. Các script
đối chiếu với GAMA đã đặt sẵn như vậy, vì model GAML chưa có xe nặng. Train
với backend GAMA cũng sinh demand không có xe nặng. Đánh giá một lượt train
cũ (cấu hình chưa có `heavy_ratio`) dùng 0 % xe nặng như lúc train.

Kiểm thử: `tests/test_heavy_vehicles_20261116.py`.

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
