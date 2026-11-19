# Training

- `logic/`: agent, môi trường, train và evaluate.
- `scripts/`: cài môi trường Python.
- `benchmarks/`: benchmark cũ, bao gồm các mô hình GAMA giữ lại để tái lập kết quả.
- `tests/`: kiểm thử logic huấn luyện.
- `results/`: checkpoint, log và kết quả chạy cục bộ.

## Cài đặt

```bash
bash train/scripts/setup_env.sh
```

Script tạo virtualenv tại `train/venv/`.

## Chạy train và evaluate

```bash
train/venv/bin/python -m train.logic.train \
  --scenario lvl --backend python --algorithm hybrid --reward-mode r1 \
  --total-steps 1000 --episode-horizon 8000 \
  --output-dir train/results/lvl_trial

train/venv/bin/python -m train.logic.evaluate \
  --scenario lvl --backend python \
  --weights train/results/lvl_trial/best_weights.pt \
  --episode-horizon 8000 --output comparison_results/lvl_eval.csv
```

Các script so sánh và audit dùng chung vẫn nằm trong `scripts/` ở repo root.

## Colab

Mở [`colab/train_hybrid_python.ipynb`](../colab/train_hybrid_python.ipynb) để train Hybrid LVL với Python simulator trên CPU và learner trên GPU. Notebook nhúng bản mã đã dùng cho phép đo 2 giờ ngày 18/11/2026, tự chọn số worker theo CPU/RAM được cấp và đồng bộ checkpoint lên Drive. Đặt `SOURCE_ARCHIVE` hoặc `USE_GIT=True` trong notebook khi muốn train một bản mã khác.
