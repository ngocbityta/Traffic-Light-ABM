# Hướng dẫn train và bàn giao: RainbowDQN điều khiển đèn nút Lê Văn Lương – Hoàng Đạo Thúy

Viết ngày 22/11/2026. File này dành cho **người hoặc agent nhận train hộ**. Đọc hết mục 1–7 trước khi chạy bất cứ thứ gì.

---

## 1. Mục tiêu và hiện trạng

**Mục tiêu**
- Train một agent **RainbowDQN** điều khiển đèn tín hiệu của nút giao Lê Văn Lương – Hoàng Đạo Thúy (Hà Nội).
- Dùng **simulator Python** của repo, phần thưởng **R1**.
- Tổng **5.000.000 quyết định** (decision), khoảng 31.000 episode.

**Hiện trạng**
- Code đã sẵn sàng, các test đều pass:
  - `train/tests`: 76 test;
  - `python-simulation`: 232 test, 1 test bỏ qua.
- **Chưa có run 5M nào bắt đầu với code hiện tại.** Run phải bắt đầu mới. Không resume được từ run cũ, vì hash mã nguồn đã đổi.
- Các thay đổi gần nhất (21–22/11):
  - random lưu lượng theo episode;
  - đặc trưng hàng đợi thang log;
  - nới C51;
  - sửa lỗi kẹt nhập làn trong sim;
  - sửa lỗi resume.
- Chủ repo cần **commit** trước khi giao. Người train phải dùng **đúng commit đó** (xem mục 6).

---

## 2. Cấu trúc repo (những phần liên quan)

| Đường dẫn | Nội dung |
|---|---|
| `python-simulation/simulation/python_simulation/python_sim.py` | Simulator vi mô: xe máy / xe con / xe tải, đổi làn, đàm phán trong nút. Có bản dịch Cython (`.pxd`). |
| `python-simulation/simulation/python_simulation/demand.py` | Sinh demand CSV: `generate_demand`, `axis_flows`, `sample_demand_level`. |
| `python-simulation/scripts/build_native.py` | Build bản Cython của sim. **Bắt buộc** để train nhanh; nếu không build thì chạy Python thuần, chậm gấp nhiều lần. |
| `train/logic/train.py` | Vòng train: async workers, learner thread, validation, checkpoint, resume. |
| `train/logic/rainbow_agent.py` | RainbowDQN: C51, n-step, PER, NoisyNet, dueling. |
| `train/logic/lvl_env.py`, `train/logic/gama_env.py` | Môi trường, quan sát (`featurize`), phần thưởng. |
| `train/logic/evaluate.py` | Đánh giá một file trọng số. |
| `train/colab_run/` | Script chạy dài trên Colab G4 / Mac, đồng bộ Google Drive, chọn bản mới nhất. Xem `train/colab_run/README.md`. |
| `train/colab_run/run_config.sh` | **Nguồn sự thật duy nhất** cho tham số train (`TRAIN_ARGS`). |
| `train/tests/` | Test logic train. |
| `train/results/<RUN_NAME>/` | Kết quả một run. |

Không dùng `colab/train_hybrid_python.ipynb` cho run này. Notebook đó là bản cũ, train thuật toán Hybrid, không phải Rainbow R1.

---

## 3. Bài toán

### Mô phỏng
- Bước thời gian `DT = 0.2 s`. Mỗi episode dài **8000 tick = 1600 s**.
- Agent ra quyết định mỗi **50 tick (10 s)**, tức khoảng 160 quyết định mỗi episode.
- 4 nhánh: `NE, SE, SW, NW`.
- 2 pha xanh:
  - **LVL** cho trục Lê Văn Lương, gồm nhánh SW–NE;
  - **HDT** cho trục Hoàng Đạo Thúy, gồm nhánh NW–SE.
- Khi chuyển pha có vàng và đỏ toàn phần. Mỗi pha phải xanh tối thiểu `min_green = 10 s` thì mới được đổi.
- Hành động: mạng có 4 đầu ra, nhưng chỉ 2 hợp lệ là `0 = LVL` và `2 = HDT`, chọn qua `action_mask`. "Giữ pha" luôn hợp lệ.
- Khi nhánh vào đầy, xe mới được đặt ở vị trí âm, tức hàng chờ ảo phía thượng lưu. **Không có xe nào bị bỏ.**
- Xe đứng im trong vùng nút quá `jam_timeout = 120 s` bị xóa và tính vào `jam_removed`. Bình thường giá trị này phải bằng 0 hoặc rất nhỏ.

### Quan sát
- Lưới không gian `4×6×20×16` cộng 17 đặc trưng phụ (aux): pha hiện tại, thời gian xanh, hàng đợi, số xe trên nhánh vào, pressure.
- Với `--queue-feature-scale log`, hàng đợi đưa vào dạng `log1p(n)/log1p(1000)` thay vì `n/100` bị cắt ở 1. Lý do: ở 130% lưu lượng, hàng đợi lên tới khoảng 300 xe mỗi trục.

### Phần thưởng R1
- `R1 = −Δ(tích phân thời gian hệ thống) / 50` cho mỗi quyết định. Tích phân này là tổng số giây mà mọi xe đã ở trong mạng.
- Hệ số chiết khấu `γ = 0.97` cho mỗi 5 s mô phỏng, tức `γ^(Δt/5)`.

### Agent Rainbow
| Tham số | Giá trị |
|---|---|
| C51 | 201 atom trên [−8000, 0] |
| n-step | 3 |
| PER | α = 0.5, β tăng từ 0.4 |
| Khác | NoisyNet, dueling |
| lr | 1e-4 |
| batch | 32 |
| buffer | 100k |
| warmup | 2000 |
| tau | 1e-3 |

---

## 4. Lưu lượng (demand)

- **Mức gốc:** 2400 xe/h mỗi nhánh. Xe đến theo Bernoulli mỗi tick, với `p = flow/3600·DT`.
- **Mỗi episode train** tự rút, từ chính seed của nó:
  - `scale ~ U(0.5, 1.3)`, tức 50–130% lưu lượng;
  - `imbalance ~ U(0, 0.2)`.

  Hai giá trị này **giữ cố định suốt episode**, và tái lập được từ seed.
- **Chia cho các nhánh:**
  - trung bình 4 nhánh = `2400·scale`;
  - mỗi nhánh NW, SE (Hoàng Đạo Thúy) = `1 + imbalance` lần mỗi nhánh SW, NE (Lê Văn Lương).
- **Validation:** 3 seed cố định ở `0.5:0.1`, `0.9:0.1` và `1.3:0.1`. Điểm tổng là **trung bình nhân**, càng thấp càng tốt. Policy có điểm tốt nhất được lưu vào `best_weights.pt`.
- **Ghi lại:** `demand_manifest.json` lưu scale, imbalance và lưu lượng từng nhánh của mọi seed.
- **Vì sao trần là 130% mà không phải 150%:**
  - năng lực thông qua của nút chỉ khoảng 110% mức gốc;
  - ở 150%, sim có hiện tượng xe tải chờ nhập làn tới 6–12 phút và năng lực thông qua tụt khoảng 20%;
  - chạy ở 150% còn đắt hơn khoảng 40%.

---

## 5. Cài đặt (máy Mac hoặc Linux)

```bash
git clone <repo> && cd <repo>
git checkout <COMMIT_DO_CHU_REPO_CHI_DINH>
bash train/scripts/setup_env.sh                  # tạo train/venv (Python 3.11)
train/venv/bin/pip install "Cython==3.3.1" setuptools pytest
train/venv/bin/python python-simulation/scripts/build_native.py   # build bản Cython
```

- Dòng cuối của bước build phải là `Native simulator is ready in ...`.
- Kiểm tra sim đang chạy bản Cython. Lệnh sau phải in ra đường dẫn kết thúc bằng `.so`:

  ```bash
  train/venv/bin/python -c "import sys; sys.path.insert(0,'python-simulation/simulation'); import python_simulation.python_sim as p; print(p.__file__)"
  ```

  Nếu in ra `python_sim.py` thì sim đang chạy Python thuần, chậm gấp nhiều lần.
- Mỗi khi đổi code sim, phải build lại. Script trong `train/colab_run/` tự build lại.

**Kiểm tra trước khi train** (khoảng 5–10 phút):

```bash
train/venv/bin/python -m pytest -q train/tests                 # 76 passed
(cd python-simulation && ../train/venv/bin/python -m pytest -q) # 232 passed, 1 skipped
```

**Smoke test**: chạy khoảng 2 phút, dừng giữa chừng, rồi resume để thấy resume hoạt động.

```bash
source train/colab_run/run_config.sh
train/venv/bin/python -m train.logic.train "${TRAIN_ARGS[@]}" \
  --total-steps 400 --episode-horizon 600 --warmup 40 --validate-every 4 \
  --checkpoint-every 2 --workers 2 --max-train-seconds 45 --output-dir /tmp/smoke
train/venv/bin/python -m train.logic.train "${TRAIN_ARGS[@]}" \
  --total-steps 400 --episode-horizon 600 --warmup 40 --validate-every 4 \
  --checkpoint-every 2 --workers 2 --resume-dir /tmp/smoke
```

Lần chạy thứ hai phải in `Resumed episode ...` và kết thúc bằng `[Train] HOAN TAT.`

Hai lệnh smoke test đổi `--total-steps` và các tham số khác. Đây chỉ là chạy thử, **không** làm vậy với run thật.

---

## 6. Quy tắc bắt buộc (đọc kỹ)

1. **Không sửa `TRAIN_ARGS` giữa chừng một run.**
   - Checkpoint lưu toàn bộ cấu hình học và **sha256 của các file mã nguồn** (sim, env, agent, train, demand…).
   - Resume sẽ báo lỗi `changed: ...` nếu có bất kỳ khác biệt nào.
   - Riêng các tham số sau được phép đổi khi resume: `--workers`, `--device`, `--max-train-seconds`, `--learner`, `--max-learner-lag`, `--rollout-mode`, `--validate-every`, `--checkpoint-every`.
2. **Dùng đúng commit.** Không sửa file `.py` nào trong `python-simulation/simulation/python_simulation/` hay `train/logic/`. Muốn sửa thì phải hỏi chủ repo, và chấp nhận bắt đầu run mới.
3. **Windows:** tắt `core.autocrlf` (`git config core.autocrlf false`) trước khi clone. Nếu không, ký tự xuống dòng thay đổi sẽ làm hash khác, và không resume được checkpoint từ máy khác. Khuyến nghị dùng Mac/Linux hoặc WSL.
4. **Chỉ một máy train một run tại một thời điểm.** Hai máy cùng train tiếp một checkpoint sẽ tạo ra hai nhánh không gộp được.
5. **`TOTAL_STEPS` (5.000.000) giữ nguyên** cho cả run.
6. **Không xóa** `training_state.pt`, `training_config.json`, `demand_manifest.json`, `training_log.csv`, `validation_log.csv` trong thư mục run.

---

## 7. Cách train

### A. Trên một máy bất kỳ (Mac / Linux / máy chủ)

```bash
source train/colab_run/run_config.sh
RUN=train/results/$RUN_NAME          # rainbow_r1_5m
# Lần đầu:
train/venv/bin/python -u -m train.logic.train "${TRAIN_ARGS[@]}" \
  --workers <SỐ_WORKER> --checkpoint-every 50 --output-dir "$RUN" \
  2>&1 | tee -a "$RUN.console.log"
# Chạy tiếp (sau khi dừng / mất điện / chuyển máy):
train/venv/bin/python -u -m train.logic.train "${TRAIN_ARGS[@]}" \
  --workers <SỐ_WORKER> --checkpoint-every 50 --resume-dir "$RUN" --output-dir "$RUN" \
  2>&1 | tee -a "$RUN.console.log"
```

- **Số worker:** khoảng số nhân vật lý trừ 1–2. Ví dụ i9 8 nhân dùng 6 worker.
- Ngoài các worker, validation chạy thêm 3 process và learner chạy 1 thread. Theo dõi RAM trong vài phút đầu; nếu máy phải dùng swap thì giảm số worker.
- Có GPU thì thêm `--device cuda`. Trên Mac Apple Silicon, `--device mps` chưa được kiểm; nên để mặc định CPU.
- Có thể thêm `--max-train-seconds N` để tự dừng sau N giây. Dừng bằng Ctrl-C cũng được: chỉ mất phần chạy sau checkpoint gần nhất (mỗi 50 episode, khoảng 10–15 phút trên Mac).
- Trên Mac, nên chạy kèm `caffeinate -i` để máy không ngủ.
- Mac của chủ repo đã có script làm sẵn mọi bước: `train/colab_run/run_mac_continue.command`. Script này tự chọn bản checkpoint mới nhất (thư mục run, bản tải từ G4, Google Drive), build Cython rồi train tiếp với 6 worker.

### B. Trên Colab G4 (tốn Compute Unit, khoảng 8.7 CU/giờ)

- Dùng `train/colab_run/run_g4_long.command`, nhấp đúp trên Mac. Script cần `colab` CLI đã đăng nhập bằng tài khoản Colab Pro.
- Các bước script tự làm:
  1. đóng gói code;
  2. tạo runtime G4 và mount Google Drive (cần đồng ý quyền một lần trong trình duyệt);
  3. train `G4_HOURS` giờ với 46 worker;
  4. sao lưu lên `MyDrive/Traffic-Light-ABM/runs/<RUN_NAME>` mỗi 15 phút;
  5. tải `latest.tar` về Mac mỗi 60 phút;
  6. cuối cùng tắt runtime.
- Chi tiết ở `train/colab_run/README.md`.
- Người dùng tài khoản Colab khác cần sửa `DRIVE_SUBDIR` nếu muốn, và tự đăng nhập `colab` CLI của mình.

### Gửi kết quả về cho chủ repo
- Nén cả thư mục `train/results/<RUN_NAME>/`, **bỏ thư mục `demand/`** (lớn, khoảng 4 GB cho cả run, và tự sinh lại được). Ví dụ:

  ```bash
  tar --exclude=demand -cf rainbow_r1_5m.tar -C train/results rainbow_r1_5m
  ```

- Hoặc dùng `train/colab_run/sync_run.py --src <run> --dst <thư mục đích>` (mặc định đã bỏ `demand/`).
- Chủ repo chạy tiếp bằng `--resume-dir`, hoặc đặt bản tar vào `train/results/<RUN_NAME>.g4_snapshots/latest.tar` rồi chạy `run_mac_continue.command`.

---

## 8. Theo dõi run

| File | Nội dung |
|---|---|
| `console.log` / stdout | Dòng `[Validation] episode E \| step S \| score X` cho biết điểm validation; có `<- best validation queue` khi điểm tốt hơn trước. |
| `training_log.csv` | Một dòng mỗi rollout: `seed`, `decisions`, `rollout_reward`, `avg_loss`, `throughput`, `avg_waiting_time`, `avg_travel_time`, `final_queue`, `completed_episode`… |
| `validation_log.csv` | `episode, step, điểm mức 0.5, điểm mức 0.9, điểm mức 1.3, điểm tổng`. Điểm = chi phí thời gian hệ thống trung bình mỗi giây, **càng thấp càng tốt**. |
| `best_weights.pt` | Policy có điểm validation tốt nhất. **Đây là sản phẩm chính.** |
| `checkpoint.pt`, `training_state.pt` | Trạng thái để resume (agent, replay buffer, RNG, các episode đang chạy dở). |
| `rainbow_<N>_steps.pt` | Snapshot định kỳ. |
| `training_summary.json` | Tóm tắt khi kết thúc: `decisions_per_second`, `learner_busy_seconds`… |

**Dấu hiệu bất thường, cần dừng lại và báo**
- `avg_loss` là `nan`, hoặc tăng vọt liên tục.
- Điểm validation ở mức 0.5 tệ dần qua nhiều lần validation liên tiếp.
- Tốc độ giảm mạnh so với bảng ở mục 10. Có thể quên build Cython; dùng lệnh kiểm tra ở mục 5.
- Lỗi `changed: code_sha256` hoặc `changed: <tham số>` khi resume. Không được "sửa cho chạy"; phải hỏi chủ repo.

---

## 9. Đánh giá policy

```bash
for s in 0.5 0.9 1.3; do
  train/venv/bin/python -m train.logic.evaluate \
    --weights train/results/rainbow_r1_5m/best_weights.pt \
    --episode-horizon 8000 --demand-scale $s --axis-imbalance 0.1 \
    --output train/results/rainbow_r1_5m/eval/scale_$s.csv
done
```

- Kiểu đặc trưng hàng đợi (log/linear) được lưu trong checkpoint, nên `evaluate.py` tự dùng đúng.
- Seed đánh giá (20.000.001+) tách riêng khỏi seed train (từ 1001) và seed validation (từ 10.000.001).

---

## 10. Tốc độ và chi phí

| Máy | Cấu hình | Quyết định/giây | Thời gian cho 5M |
|---|---|---|---|
| MacBook Pro i9-9880H | 6 worker | 11.2 (đo 21/11) | khoảng 124 h |
| Colab G4 | 40 worker, code cũ | 94.3 (đo) | — |
| Colab G4 | 46 worker, code mới | ước tính 135–150, **chưa đo** | khoảng 9–10 h, tức 80–90 CU |

- Chi phí sim tăng mạnh theo lưu lượng (ms/tick):

  | Lưu lượng | ms/tick |
  |---|---|
  | 50% | 5.7 |
  | 100% | 12.3 |
  | 125% | 22 |

- Trung bình trên dải 50–130% gần bằng chạy cố định 100%. Episode ở mức đông chậm hơn, nhưng worker chạy không đồng bộ nên không phải chờ nhau.
- Kế hoạch đã bàn: kết hợp G4 và Mac. Train trên G4 tới khi gần hết CU, rồi kéo checkpoint về Mac chạy tiếp (đã dựng `train/colab_run/`).

---

## 11. Vấn đề đã biết

- **Sim, mức quá bão hoà (trên khoảng 115%):** xe tải cần nhập làn có thể chờ 3–5 phút, vì xe máy liên tục lấp khe. Ở 130% chưa thấy ca nào vượt 300 s. Lỗi kẹt cứng (hai xe chặn nhau mãi) đã được sửa ngày 22/11.
- **Thư mục `demand/`:** mỗi episode thêm khoảng 130 KB, cả run khoảng 4 GB. Có thể xóa các file của episode đã xong nếu thiếu dung lượng, nhưng nên dừng train trước khi xóa.
- **Phần còn chờ chủ repo chạy:** buổi diễn tập sao lưu/resume trên Mac (`train/colab_run/test_sync_resume.command`) và benchmark G4 với code mới chưa chạy.

---

## 12. Tài liệu chi tiết (trong Project "GAMA" trên Claude)

| Tài liệu | Nội dung |
|---|---|
| `claude/random_luu_luong_20261122.md` | Random lưu lượng, nới C51, đặc trưng log, lỗi resume, lỗi kẹt nhập làn. |
| `claude/toc_do_colab_g4_20261121.md` | Tối ưu tốc độ, learner thread, đồng bộ Drive, kế hoạch CU. |
| `claude/toc_do_train_mac_20261118.md` | Đo tốc độ trên Mac. |
| `claude/danh_gia_san_sang_train_20261114.md`, `claude/sua_loi_san_sang_train_20261115.md` | Rà soát độ đúng của pipeline train. |
| `claude/junction_negotiation_20261113.md`, `claude/sua_co_che_nhuong_20261116.md` | Cơ chế đàm phán và nhường trong nút. |
