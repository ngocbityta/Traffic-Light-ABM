# Traffic-Light-ABM

> Hướng nghiên cứu hiện tại dùng GAMA làm môi trường mô phỏng chính. Xem [đề xuất nghiên cứu](reports/research.md) về state sáu kênh và Rainbow DQN / Max Pressure. Cả Rainbow và 3DQN lai Max Pressure đã được nối với bốn reward R1–R4 trên mạng đường GAMA hiện tại; nút khảo sát Lê Văn Lương – Vành Đai 2.5 vẫn cần hiệu chỉnh từ dữ liệu thực địa.

Mô hình train mặc định hiện nay là ngã tư `LeVanLuongIntersection/models/LeVanLuong.gaml`: bốn nhánh đường mặt đất, hai pha đèn, không có hầm hoặc cầu vượt. `dqn_py/train.py` và `dqn_py/evaluate.py` mặc định dùng mô phỏng Python của ngã tư này; `--backend gama` dùng cùng GAML để đối chiếu. Mô hình `Traffic/` có hầm/cầu vượt là bài toán cũ, chỉ chạy khi truyền `--scenario legacy`.

## Train ngã tư Lê Văn Lương

```bash
dqn_py/venv/bin/python dqn_py/train.py --scenario lvl --backend python --algorithm hybrid --reward-mode r1 --total-steps 1000 --episode-horizon 8000 --output-dir dqn_py/run/lvl_trial
dqn_py/venv/bin/python dqn_py/train.py --scenario lvl --backend python --algorithm hybrid --reward-mode r1 --workers 4 --total-steps 1000 --episode-horizon 8000 --output-dir dqn_py/run/lvl_trial_4w
dqn_py/venv/bin/python dqn_py/evaluate.py --scenario lvl --backend python --weights dqn_py/run/lvl_trial/best_weights.pt --episode-horizon 8000 --output comparison_results/lvl_eval.csv
dqn_py/venv/bin/python scripts/compare_lvl_training_envs.py --seed 1 --horizon 500 --output comparison_results/lvl_training_parity.csv
dqn_py/venv/bin/python scripts/run_spatial_matrix.py --scenario lvl --backend python --output-dir comparison_results/lvl_matrix --total-steps 1000 --episode-horizon 8000
```

Hai backend dùng cùng `roads.geojson`, lịch xe CSV và hai action hợp lệ `0` (Lê Văn Lương) / `2` (Hoàng Đạo Thúy). Quyết định mỗi 50 tick (10 s); đổi pha có 15 tick đỏ toàn phần (3 s). Tensor DTSE/spatial và công thức reward được dựng từ cùng mã Python ở `dqn_py/lvl_env.py`. Trong phép ghép cặp 1.000 tick seed 1, 19 tick quyết định và pha đèn trùng, cùng 540 xe được tạo, GAMA/Python cho 415/413 xe hoàn thành. Quỹ đạo và làn của từng xe vẫn có sai số; xem [báo cáo đối chiếu](reports/lvl_python_gama_training_parity_20261109.md) trước khi dùng cho kết luận nghiên cứu.

Python hỗ trợ `--workers N` bằng các tiến trình rollout độc lập và một learner chung. Ở 4.000 tick, thời gian môi trường train một worker gần bằng GAMA; phép đo 2 worker trên episode ngắn nhanh hơn 1 worker khoảng 1,24 lần. Cần đo lại với episode 8.000 tick trước khi chạy train dài.

Các hướng dẫn còn lại về `Traffic/` mô tả bài toán legacy và cần thêm `--scenario legacy` khi gọi `train.py` hoặc `evaluate.py`.

[Kiểm tra GAMA sau cleanup](audits/gama_postcleanup_20261104/README.md): cả năm controller qua bài kiểm tra 1.500 tick, nhưng Fixed-Time và 3DQN còn lỗi ở lượt 4.000 tick. Không xem pipeline GAMA đã đạt kiểm chứng mô phỏng dài.

Mô phỏng vi mô giao thông tại nút giao bốn nhánh bằng **GAMA Platform**, kèm năm bộ điều khiển đèn: Fixed-Time, MaxPressure, Q-Learning, Deep Q-Network (DQN) và Dueling Double DQN (3DQN). Mô hình gồm đường mặt đất, cầu vượt, hầm và các tuyến rẽ. Python và PyTorch được dùng để huấn luyện, đánh giá và chạy benchmark trên cùng lịch xe.

## Nội dung repository

| Đường dẫn | Nội dung |
| --- | --- |
| `Traffic/models/` | Mô hình GAML cho năm bộ điều khiển, benchmark và kiểm thử |
| `Traffic/includes/` | Mạng đường, cảnh quan và chỉ số an toàn |
| `Traffic/data/` | Dữ liệu đếm xe dùng để hiệu chỉnh |
| `dqn_py/` | DQN, môi trường kết nối GAMA Server, huấn luyện và đánh giá |
| `scripts/` | Benchmark, so sánh, kiểm thử và demo DQN |
| `LeVanLuongIntersection/` | Nút Lê Văn Lương × Hoàng Đạo Thúy (hình học vệ tinh); train Rainbow/Hybrid bằng `train.py --site lvl`, xem [README](LeVanLuongIntersection/README.md) |

## Mô phỏng Python độc lập (đang hiệu chỉnh)

`dqn_py/python_env.py` cung cấp cùng giao diện async, lịch nhu cầu CSV, bốn action, DTSE/spatial state và reward R1–R4 cho `train.py`/`evaluate.py`. `--backend python` chạy hoàn toàn trong Python, không khởi động GAMA Server. Chế độ này chỉ nhận `--demand-mode paired` để dùng đúng lịch xe chung. Chuyển động dùng quy tắc bám xe dọc làn do Python thực hiện; GAMA dùng `advanced_driving` cùng các quy tắc nhập làn, nhường đường và ra nút phức tạp hơn. Checkpoint và báo cáo ghi backend để tránh lẫn kết quả.

```bash
dqn_py/venv/bin/python dqn_py/train.py --scenario legacy --backend python --algorithm hybrid --reward-mode r1 --total-steps 1000 --episode-horizon 8000 --output-dir dqn_py/run/hybrid_python_trial
dqn_py/venv/bin/python dqn_py/evaluate.py --scenario legacy --backend python --weights dqn_py/run/hybrid_python_trial/best_weights.pt --episodes 2 --output comparison_results/hybrid_python_trial.csv
dqn_py/venv/bin/python scripts/run_spatial_matrix.py --scenario legacy --backend python --output-dir comparison_results/python_matrix --total-steps 1000 --episode-horizon 8000
```

Fixed-Time và MaxPressure có runner Python riêng trên cùng động học và lịch xe:

```bash
dqn_py/venv/bin/python scripts/run_python_traffic.py --controller fixed --seed 1001 --horizon 8000 --output comparison_results/python_fixed.csv
dqn_py/venv/bin/python scripts/run_python_traffic.py --controller maxpressure --seed 1001 --horizon 8000 --output comparison_results/python_maxpressure.csv
```

Đối chiếu hai controller nền với GAML trên lịch xe ghép cặp bằng `scripts/compare_python_baselines_gama.py --controller fixed|maxpressure --start-server --horizon 1000 --output <file.csv>`.

Kiểm tra trên cùng seed và bốn pha cố định theo lượt quyết định, lưu chuỗi chỉ số, sai lệch và `within_tolerance`:

```bash
dqn_py/venv/bin/python scripts/compare_python_gama.py --start-server --seed 1001 --horizon 4000 --sample-every 200 --tolerance 0.1 --enforce --output comparison_results/python_gama_1001.csv
```

Đối chiếu thực tế ngày 08/11/2026 trên seed 1002 và cùng lịch xe: ở 1.500 tick, sai lệch cuối là 1,5% số xe nhận và 16,5% tích phân hàng chờ; ở 4.000 tick, sai lệch là 42,5% số xe nhận, 7,1% xe qua nút, 38,7% tốc độ trung bình, 10,7% tích phân hàng chờ và lịch pha đã trôi. Các số liệu này **không đạt** ngưỡng 10%. Chi tiết trong `reports/python_parity_20261108/`. Việc chuyển đổi toàn bộ mô phỏng và hiệu chỉnh động học, bao gồm các chỉ số an toàn, vẫn chưa hoàn tất. Bản Python của nút Lê Văn Lương nằm ở `LeVanLuongIntersection/python_sim.py` và cũng cần đối chiếu thêm với GAMA trước khi dùng nghiên cứu.

Slide seminar, tài liệu nghiên cứu, checkpoint, kết quả chạy và ảnh chụp màn hình là dữ liệu cục bộ, không có trong repository. Các script tạo kết quả trong `comparison_results/` hoặc `dqn_py/run/`.

## Cài đặt

- GAMA Platform 2025.6.4 trở lên, Java 17 trở lên.
- Python 3.11 cho script cài đặt mặc định. Có thể đặt `PYTHON_BIN` nếu dùng một bản Python tương thích khác.
- GAMA Headless mặc định ở `/Applications/Gama.app/Contents/headless/gama-headless.sh`. Nếu cài ở nơi khác, đặt biến môi trường `GAMA_HEADLESS` tới file `gama-headless.sh`.

```bash
git clone https://github.com/ngocbityta/Traffic-Light-ABM.git
cd Traffic-Light-ABM
bash dqn_py/setup_env.sh
```

Trong GAMA Studio, chọn **File → Import → General → Existing Projects into Workspace** và trỏ đến thư mục `Traffic`.

## Chạy mô hình trong GAMA Studio

Mở một trong các file sau, chọn experiment `my_experiment` rồi nhấn **Run**:

- `Traffic/models/intersection-fixed-time.gaml`
- `Traffic/models/intersection-maxpressure.gaml`
- `Traffic/models/intersection-qlearning.gaml`

Các experiment `my_batch_experiment` chạy mô phỏng không có giao diện. Trong GUI, màn hình `doan_3d` hiển thị cầu vượt, hầm và giao lộ; `intersection_view` hiển thị bản đồ 2D.

## Huấn luyện Q-learning, DQN và 3DQN trên cùng giao diện GAMA

Ba thuật toán dùng cùng DTSE `3×8×20`, bốn action pha xanh, action mask, lịch yêu cầu xe theo seed và reward. Q-learning rời rạc hóa số xe dừng/đang tới theo bốn nhóm chuyển động; DQN dùng mạng phẳng; 3DQN dùng CNN, Double DQN và nhánh dueling. Đây là đường train bốn pha để so sánh ba thuật toán. Nhánh Rainbow bên dưới dùng state sáu kênh riêng trên cùng mạng đường.

Ví dụ huấn luyện với reward thời gian chờ của xe trong mạng **và** xe còn đợi phát ngoài mạng, cùng hai GAMA worker độc lập:

```bash
dqn_py/venv/bin/python dqn_py/train.py --scenario legacy --backend gama --algorithm qlearning --reward-mode total_wait --workers 2 --total-steps 1000000 --episode-horizon 8000 --start-server
dqn_py/venv/bin/python dqn_py/train.py --scenario legacy --backend gama --algorithm dqn --reward-mode total_wait --workers 2 --total-steps 1000000 --episode-horizon 8000 --start-server
dqn_py/venv/bin/python dqn_py/train.py --scenario legacy --backend gama --algorithm 3dqn --reward-mode total_wait --workers 2 --total-steps 1000000 --episode-horizon 8000 --start-server
```

Mặc định `--demand-mode paired` tạo một CSV lịch yêu cầu xe cho mỗi seed, dùng chung giữa ba thuật toán; xe chưa vào được mạng vẫn ở hàng đợi ngoài mạng. Có thể dùng `--demand-mode model` để phát xe trực tiếp trong GAMA, nhưng seed khi đó không bảo đảm lịch xe giống nhau giữa các chính sách. Mỗi worker chạy một GAMA Server trên CPU với Java heap tối đa 2 GB (`--gama-heap-mb`) và chỉ thu thập transition. Một learner cập nhật trên thiết bị PyTorch khả dụng (CUDA/MPS/CPU), nhận dữ liệu từ các worker theo thứ tự seed. Khi chạy nhiều worker, mỗi worker gửi tối đa 16 quyết định mỗi lượt (`--rollout-decisions`), sau đó nhận chính sách mới và tiếp tục cùng episode GAMA. Reward `total_wait` là âm của gia số `(queue_time_integral + external_wait_integral)/50`, dùng discount theo thời gian thực giữa hai quyết định. `queue_time` chỉ tính xe đã vào mạng; `reference` dùng delay bình phương tốc độ theo bài báo. Với nhiều worker và reward `reference`, phải truyền cùng một `--max-delay-sq` cố định dương cho mọi worker; mặc định chạy một worker nếu muốn giữ chuẩn hóa cực đại chạy theo bài tham khảo. Các chỉ số TTC/TIT/PET mặc định tắt; nếu bật, quỹ đạo có thể đổi và phải dùng cùng cấu hình khi so sánh.

Kết quả ở `dqn_py/run/<algorithm>_<timestamp>/`: `best_weights.pt` chọn theo hai seed validation 10.000.001–10.000.002 mặc định mỗi 50 episode, `dqn_weights.pt` là policy cuối, `checkpoint.pt` và `training_state.pt` mặc định lưu mỗi 50 episode. `training_state.pt` gồm replay, optimizer, RNG và bộ chuẩn hóa reward để tiếp tục học. Có thể điều chỉnh `--validate-every` và `--checkpoint-every` theo chi phí thời gian/đĩa. Dùng `--resume-dir` trỏ tới thư mục này và tăng `--total-steps` hoặc `--episodes` để tiếp tục; script từ chối nếu cấu hình hay mã mô hình đã đổi. Đánh giá từng policy trên seed test riêng, mặc định bắt đầu từ 20.000.001:

```bash
dqn_py/venv/bin/python dqn_py/evaluate.py --scenario legacy --backend gama --weights /duong/dan/best_weights.pt --episodes 5 --start-server
```

Từ bản sửa 19/10/2026, mọi controller dùng vàng 3 giây, đỏ toàn phần tối thiểu 2 giây và chờ xe đã vào nút thoát ra. 3DQN có action mask hợp lệ cho bốn pha. Tùy chọn `dqn_service_guard` mặc định **tắt**; khi bật, guard không gia hạn quá 60 giây nếu pha khác có xe và ưu tiên pha đợi quá 120 giây ở quyết định tiếp theo. Mask được dùng cho khám phá, greedy và Double DQN target. GUI preview phục vụ cả bốn pha.

## Tám trường hợp: Rainbow / 3DQN + Max Pressure × R1–R4

Lệnh thử tích hợp ngắn trên mạng đường hiện tại:

```bash
dqn_py/venv/bin/python dqn_py/train.py --scenario legacy --backend gama --algorithm rainbow --total-steps 10 --episodes 1 --episode-horizon 1000 --warmup 2 --batch-size 2 --buffer-size 32 --start-server
```

Chạy đủ tám trường hợp với cùng seed, lịch xe và các hệ số reward:

```bash
dqn_py/venv/bin/python scripts/run_spatial_matrix.py --scenario legacy --backend gama --output-dir /duong/dan/ket_qua_8_truong_hop --total-steps 10 --episodes 1 --episode-horizon 1000 --warmup 2 --batch-size 2 --buffer-size 32
```

`run_spatial_matrix.py` tạo tám thư mục `rainbow_r1` … `hybrid_r4` và `matrix_manifest.json`. Dùng `--total-steps` lớn hơn, `--episodes 0` (mặc định) và các hệ số đã hiệu chỉnh để huấn luyện chính. Mỗi trường hợp dùng `train.py --algorithm rainbow|hybrid --reward-mode r1|r2|r3|r4`; `evaluate.py --weights <best_weights.pt>` nạp cả hai kiến trúc.

Ví dụ Rainbow R1 với bốn GAMA worker độc lập và một learner:

```bash
dqn_py/venv/bin/python dqn_py/train.py --scenario legacy --backend gama --algorithm rainbow --reward-mode r1 \
  --workers 4 --rollout-decisions 16 --start-server --port 6930 \
  --total-steps 1000000 --episode-horizon 8000 --warmup 32 \
  --batch-size 32 --buffer-size 4096 --max-train-seconds 600 \
  --validate-every 1000000 --checkpoint-every 50 \
  --output-dir dqn_py/run/rainbow_r1_4w
```

`--fast-simulation` bật mặc định cho `train.py`: GAMA bỏ cảnh trang trí, góc nghiêng thân xe chỉ để vẽ, chuỗi đếm ngược và log `METRIC_CSV` không được learner đọc; riêng R1 bỏ phép tính R2–R4 nhưng vẫn giữ nguyên dịch chuyển ngang, xe, pha đèn, state và tích phân R1. Dùng `--no-fast-simulation` để đối chiếu. Chế độ này không tự bật trong các script benchmark GAMA khác. [Kiểm tra 4 worker và tối ưu R1 ngày 07/11/2026](audits/spatial_4worker_20261107/README.md) có số đo cùng seed và phép đối chiếu quỹ đạo.

[Kiểm thử 8 lượt × 10 phút ngày 06/11/2026](audits/spatial_10min_20261106/README.md) ghi tốc độ thực tế, checkpoint, lỗi Java ở ranh giới reload/close và kết quả chạy lại sau sửa. `run_spatial_matrix.py` quét log GAMA, ghi `server_error_count` vào manifest dù Python thoát mã 0.

Để đo thử tốc độ trong 10 phút train cho mỗi cấu hình, truyền `--max-train-seconds 600 --warmup 32 --batch-size 32 --buffer-size 4096 --validate-every 1000000` cùng `--total-steps 1000000`. Script dừng ở ranh giới quyết định, lưu `dqn_weights.pt` và ghi số quyết định/giây, giây mô phỏng/giây thực vào `training_summary.json` và `matrix_manifest.json`. Thời gian khởi động và tắt GAMA được ghi riêng trong `wall_seconds`; `train_seconds` là thời gian thực của learner và các rollout song song, không cộng chồng thời gian từng worker. Một hàng `training_log.csv` là một rollout, có thể chỉ là một phần episode. Khi dừng vì giới hạn thời gian mà chưa đến lượt validation, dùng `dqn_weights.pt`; `best_weights.pt` chỉ xuất hiện sau validation.

Rainbow dùng lưới `4×6×20×16` (E/N/W/S; xe máy, ô tô, xe nặng, tốc độ dọc, tốc độ ngang, mặt nạ đường) và 17 biến toàn nút gồm pha hiện tại, tuổi xanh, hàng chờ/xe đang tới, pressure bốn pha và action mask. Pressure trong đầu vào mạng được chia cho 50 rồi giới hạn vào [-1, 1]; nhánh chọn pha của Hybrid chuẩn hóa điểm pressure giữa các action hợp lệ. Ô dọc dài 8 m tính từ vạch dừng, ô ngang là một sublane; C0–C2 lưu tỷ phần footprint, C3 là tốc độ dọc trung bình có trọng số diện tích, C4 là vận tốc ngang có dấu lớn nhất của mỗi xe kể từ snapshot quyết định trước (chuẩn hóa theo 5 m/s, rồi trung bình theo diện tích trong ô), C5 là mặt nạ làn hợp lệ. Cách giữ đỉnh C4 tránh bỏ lỡ một lần đổi làn ngắn giữa hai quyết định cách nhau khoảng 10 giây. C4 lấy đạo hàm của nội suy chuyển làn hiển thị hiện có, vì vậy **chưa phải quỹ đạo lách ngang được hiệu chỉnh bằng video**. C5 hiện là 1 trên 16 sublane của mạng đều; sau khảo sát cần dựng lại mặt nạ theo hình học thật.

**Rainbow** có PER, 3-step return, C51 51 atom, NoisyNet, Double DQN và Dueling Network. NoisyNet dùng để thăm dò sau warm-up ngẫu nhiên; Rainbow không dùng epsilon. **Hybrid** dùng CNN Dueling Double DQN trên cùng state, cộng điểm Q và điểm Max Pressure đã chuẩn hóa để chọn pha; `--pressure-lambda` (mặc định 0,5) chỉnh sức ảnh hưởng của Pressure. Pressure cho mỗi pha bằng tổng hàng chờ các chuyển động được phục vụ trừ hàng chờ ở đường ra tương ứng; chỉ đếm xe chậm dưới 5 km/h trong 160 m đường vào và 80 m đầu đường ra. Cả lựa chọn action lẫn Double DQN target của hybrid dùng quy tắc kết hợp này. Replay của hai agent nén state bằng zlib; mặc định 100.000 transition, warm-up 20.000. Mỗi action vẫn chọn một trong bốn pha hợp lệ của controller GAMA hiện tại.

GAMA tích lũy bốn loại chi phí ở **mỗi tick**, kể cả thời gian vàng và đỏ. Reward bằng âm gia số chi phí giữa hai quyết định:

| Reward | Chi phí tích lũy dùng trong mã | Tham số |
| --- | --- | --- |
| R1 | Thời gian mọi xe trong mạng **và** xe chờ vào mạng, gồm bypass | `--r1-ref` = $D_{ref}$ |
| R2 | Thời gian mọi xe dừng dưới 1 km/h + $\lambda$ × thời gian xe chờ vào mạng | `--r2-ref` = $W_{ref}$; `--r2-lambda` |
| R3 | R1 + $\beta$ × thời gian xe chậm dưới 5 km/h trong 80 m đầu đường ra | `--b-ref` = $B_{ref}$; `--r3-beta` |
| R4 | Thời gian chậm so với tốc độ mong muốn riêng từng xe, trọng số PCU xe máy 0,3; ô tô 1; buýt 2; xe chưa vào mạng bị tính chậm hoàn toàn | `--r1-ref` = $D_{ref}$ |

R3 trùng R1 ở các episode không có xe bị chặn tại đường ra; cần kiểm tra chỉ số `outgoing_block_integral` khi phân tích kết quả. Mặc định các mốc chuẩn hóa bằng 50 vehicle-second và $\lambda=\beta=1$ là **giá trị tạm cho prototype**. Trước thí nghiệm chính, đo fixed-time trên cùng nhu cầu để chọn hệ số cố định, rồi giữ nguyên giữa hai thuật toán. Discount dùng thời lượng thực của mỗi quyết định; Rainbow nhân các discount tương ứng trong n-step. Checkpoint lưu reward và hệ số để đánh giá đúng cấu hình; `training_state.pt` lưu replay/optimizer/RNG để resume. Kết quả trên mạng này chưa đại diện cho nút Lê Văn Lương – Vành Đai 2.5.

Trong mỗi ô DTSE, tốc độ là **trung bình của các xe**, không phụ thuộc thứ tự duyệt. Occupancy vẫn nhị phân theo nhóm chuyển động, nên chưa biểu diễn mật độ theo từng làn vật lý. Checkpoint 3DQN cũ `dtse-v1` không dùng trực tiếp với hợp đồng state/controller mới; cần huấn luyện lại. `best_weights.pt` được chọn trên hai seed validation riêng theo thời gian chờ tích lũy; với `total_wait` phép chọn tính cả xe đợi ngoài mạng. Horizon là giới hạn thời gian: transition cuối bị cắt giữa action không được đưa vào replay; transition đầy đủ vẫn bootstrap, không coi giao thông là terminal khi đồng hồ hết giờ.

Controller này chọn trực tiếp một trong bốn pha xanh: 0 NS đi thẳng, 1 NS rẽ trái, 2 EW đi thẳng, 3 EW rẽ trái. Xe rẽ phải và quay đầu dùng nhánh riêng, không nhận tín hiệu đèn trong topology GAMA. Chọn lại pha hiện tại kéo dài xanh 10 giây; đổi pha đi qua vàng 3 giây và đỏ toàn phần **tối thiểu** 2 giây, kéo dài đến khi xe đang qua nút đã thoát, rồi mới mở xanh tối thiểu 10 giây. State DTSE có kích thước `3×8×20`: occupancy, tốc độ chia cho 50 km/h, và pha xanh; 8 hàng là bốn hướng E/N/W/S nhân hai nhóm đi thẳng/rẽ trái, mỗi ô cách vạch dừng 8 m trong vùng 160 m. Reward dùng tổng squared delay của bài tham khảo: `clip(1 - D_t/max(D_1..D_t), 0, 1)`, với `D_t = sum(1 - (v/50)^2)` trên **mọi xe có tín hiệu thuộc đường vào nút giao**, kể cả ngoài vùng DTSE; mốc `max` được cập nhật theo lịch sử và giữ xuyên các episode. Đầu vào vẫn là DTSE thích nghi cho hình học GAMA; action 4 pha, mạng CNN/dueling và các siêu tham số chính theo bài báo. Epsilon bắt đầu ở 1, giảm hàm mũ theo thang 2 triệu quyết định tới sàn 0.001; 100.000 transition warm-up đầu được lấy ngẫu nhiên từ các action hợp lệ trước khi cập nhật gradient. Bài báo Đà Nẵng ghi epsilon 0.001 là giá trị ban đầu, trong khi công thức và nghiên cứu nguồn được trích dẫn mô tả epsilon tối thiểu; ở đây chọn cách đặt 0.001 làm sàn để vẫn có giai đoạn khám phá. Mặc định replay 1 triệu transition, Adam `1e-4`, Huber, `γ=0.99`, soft target `τ=0.001`. Có thể giảm `--buffer-size`, `--warmup`, `--total-steps` để kiểm tra nhanh. Một lượt chạy `--total-steps 5000000` lưu cả checkpoint 1M và 5M; chạy riêng lệnh 1M rồi 5M sẽ bắt đầu lại từ đầu. Số step ở đây là quyết định của agent, không phải chu kỳ 0,1 giây của GAMA.

Thiết kế state/action/reward và cấu hình mạng dựa trên [bài báo Đà Nẵng](https://doi.org/10.3390/make7030065) và [mã Ducrocq–Farhi](https://github.com/romainducrocq/DQN-ITSCwPD); [repo tham khảo thứ hai](https://github.com/ngctnnnn/DRL_Traffic-Signal-Control) dùng Double/Dueling DQN trên SUMO. Bốn pha và nhóm làn của nút giao GAMA là phép ánh xạ theo topology hiện có; kích thước DTSE vì thế khác cấu hình SUMO gốc.

### Demo DQN trong GAMA Studio

Demo GUI dùng checkpoint `best_dqn.pt` do `scripts/fair_benchmark.py` tạo ra. Checkpoint **không được lưu trong Git**; cần chạy benchmark trước hoặc truyền đường dẫn tới checkpoint tương thích (mạng 11 đầu vào, hai hành động):

```bash
python3 scripts/demo_dqn.py --weights /duong/dan/best_dqn.pt
```

Giữ terminal đang chạy, mở `Traffic/models/intersection-dqn-demo.gaml` trong GAMA Studio, chọn `my_experiment` rồi nhấn **Run**. Nhấn `Ctrl+C` ở terminal để dừng demo. Demo GUI phát xe trực tiếp; số liệu benchmark dùng lịch xe cố định nên không tái tạo bằng cách chạy GUI.

## Benchmark hai action lịch sử

```bash
dqn_py/venv/bin/python scripts/fair_benchmark.py \
  --output comparison_results/fair_benchmark_RUN \
  --train-episodes 40 --validate-every 10 --runs 5 \
  --horizon 8000 --workers 4 --start-server --port 6876
```

Pipeline này huấn luyện và so sánh Fixed-Time, MaxPressure, Q-Learning và DQN với cùng lịch yêu cầu xe nhưng chỉ có action **giữ/đổi pha đi thẳng**; pha rẽ trái chạy theo lịch cố định. Vì vậy không dùng kết quả này như đối chứng thuật toán trực tiếp với đường train bốn action ở trên. Kết quả gồm `report.md`, `comparison.png`, `summary.csv`, `per_run_metrics.csv`, checkpoint và log từng lượt. Có thể dùng `--workers 1` để giảm tài nguyên; dùng `--resume` với cùng cấu hình để tiếp tục lượt bị gián đoạn. `--smoke --horizon 1000` chạy kiểm tra nhanh trong một thư mục output mới.

`scripts/compare_4_models.py` chạy các model GAML riêng để kiểm tra bản demo; kết quả đó không thay thế benchmark chính sách đã huấn luyện ở trên.

### So sánh 3DQN với bốn bộ điều khiển cũ

Sau khi có checkpoint và CSV nhu cầu từ benchmark bốn bộ điều khiển, chạy:

```bash
dqn_py/venv/bin/python scripts/benchmark_3dqn_vs_4.py \
  --baseline comparison_results/fair_benchmark_RUN \
  --output comparison_results/benchmark_3dqn_vs_4_RUN \
  --train-episodes 40 --runs 5 --horizon 8000 --workers 4 --port 6890
```

Script huấn luyện 3DQN trên cùng 40 seed huấn luyện và đánh giá cả năm chính sách trên cùng năm seed kiểm tra, 8.000 chu kỳ/seed. Bốn chính sách cũ chạy qua `intersection-dqn-legacy.gaml` để giữ đúng action 0/1; 3DQN dùng mô hình bốn action. Warm-up mặc định của benchmark là 64 transition, replay 20.000 và bốn cập nhật mỗi transition. Epsilon thăm dò giảm theo hàm mũ sau mỗi transition; lượt kiểm tra dùng epsilon 0. Checkpoint được chọn theo hàng đợi trung bình trên hai seed validation sau mỗi 10 episode, cùng quy tắc với baseline. Đây là cấu hình so sánh theo ngân sách 40 tập, khác cấu hình 100.000 warm-up của huấn luyện 3DQN đầy đủ. Kết quả có `report.md`, `per_run_metrics.csv`, checkpoint, log từng lượt và `manifest.json`; dùng `--resume` với cùng cấu hình để tiếp tục lượt bị gián đoạn. Nếu các model GAMA baseline và dữ liệu nhu cầu không đổi, có thể thêm `--reuse-baselines /duong/dan/benchmark-cu` để dùng lại kết quả đánh giá bốn baseline sau khi đối chiếu hash.

Q-Learning và DQN trong benchmark là hai learner Python dùng chung transition của 40 episode; mỗi learner điều khiển 20 episode khi thu thập dữ liệu. Các model GAML `intersection-qlearning.gaml` và `intersection-dqn-python.gaml` là đường chạy demo/3DQN riêng, không phải checkpoint Q-Learning/DQN Python đã đánh giá trong benchmark. `dqn_py/train.py` mặc định tạo CSV nhu cầu cố định cùng cách như benchmark, nhưng dùng giao diện bốn action.

Benchmark mới mặc định `--reward-mode queue_time`: reward âm của thời gian xe dừng tích lũy giữa hai quyết định chia 50, discount `0.97 ** (dt_seconds / 5)`, cùng mục tiêu với Q-Learning và DQN. Reward này tính cả thời gian vàng/đỏ và là lựa chọn thích nghi GAMA, **khác reward bài báo**. Dùng `--reward-mode reference --reward-scope all` để giữ công thức delay bình phương tốc độ của repo; `--reward-scope near` là ablation chỉ tính vùng DTSE 160 m. `train.py` độc lập vẫn mặc định `reference`; có thể truyền `--reward-mode queue_time`.

Phải chạy lại `fair_benchmark.py` trên phiên bản hiện tại trước khi so sánh. Script từ chối baseline huấn luyện bằng code/môi trường khác; cache kiểm tra hash policy/checkpoint, source Python/GAMA, demand và cấu hình reward. `--resume` còn đối chiếu agent seed và toàn bộ CSV train/validation/test. Q-learning GAML chạy online đã đổi sang reward tích lũy theo thời gian, discount theo khoảng quyết định, state có tuổi pha và bootstrap chỉ action đổi pha tại giới hạn xanh; đây vẫn là một demo online, không phải bảng Q của benchmark Python.

Các kiểm toán năm phương pháp và benchmark clearance trước bản sửa 19/10/2026 là dữ liệu lịch sử, không đại diện cho code hiện tại. Chỉ số TTC hiện là sự kiện đuổi gần trên quỹ đạo cùng làn, tính độc lập với tên xe; phanh gấp được đếm khi bắt đầu một đợt phanh. PET tiếp tục đo xung đột cắt nhau. Các bộ đếm an toàn cũ theo tick không so trực tiếp với bộ đếm sự kiện mới.

## Kiểm thử

```bash
python3 scripts/run_doan_integration_suite.py
python3 scripts/run_test_suite.py
dqn_py/venv/bin/python -m unittest dqn_py.test_demand scripts.test_fair_benchmark scripts.test_learning_pipeline dqn_py.test_three_dqn_agent dqn_py.test_four_phase_agents dqn_py.test_gama_env_schedule dqn_py.test_server_manager_cleanup
```

Các kiểm thử tích hợp GAMA cần GAMA Headless và có thể mất thời gian. Kết quả và log được ghi vào `comparison_results/`.

Kiểm tra protocol GAMA Server end-to-end (tùy chọn):

```bash
dqn_py/venv/bin/python dqn_py/test_doan_server.py --start-server --horizon 4000
```
