# Traffic Light ABM

Repo này hiện được chia thành ba phần chính:

| Thư mục | Nội dung |
| --- | --- |
| [`gama-simulation/`](gama-simulation/README.md) | Mô phỏng GAMA ngã tư Lê Văn Lương – Hoàng Đạo Thúy |
| [`python-simulation/`](python-simulation/README.md) | Mô phỏng Python độc lập và giao diện quan sát trên web |
| [`train/`](train/README.md) | Logic huấn luyện, script, benchmark và kết quả DQN |
| `scripts/` | Công cụ phân tích, đối chiếu và kiểm toán dùng chung |

## Chạy mô phỏng Python

```bash
python3 python-simulation/simulation/python_simulation/python_sim.py \
  --seed 1 --cycles 4000 --output train/results/lvl_python.csv
```

Chạy giao diện web quan sát tại `http://127.0.0.1:8765/`:

```bash
python3 scripts/lvl_observer_server.py --port 8765 --seed 1
```

## Huấn luyện

Cài môi trường theo hướng dẫn ở [`train/README.md`](train/README.md), sau đó chạy ví dụ:

```bash
train/venv/bin/python -m train.logic.train \
  --scenario lvl --backend python --algorithm hybrid --reward-mode r1 \
  --total-steps 1000 --episode-horizon 8000 \
  --output-dir train/results/lvl_trial
```

## GAMA

Mở `gama-simulation/` trong GAMA Studio và chạy experiment `main` ở
`gama-simulation/simulation/models/LeVanLuong.gaml`. Các benchmark GAMA cũ
được lưu cùng phần benchmark ở `train/benchmarks/gama/`.
