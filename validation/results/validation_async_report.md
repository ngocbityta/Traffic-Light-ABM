# Báo cáo: tách validation khỏi learner

Nhánh: `fix/validation-flow`. File sửa: `train/logic/train.py` (chưa commit).

## 1. Vấn đề

Trên Colab (CUDA, `--workers 40`), báo cáo ghi nhận validation chặn learner: 14 lần gián đoạn, mỗi lần 52–61s, tổng 762s trên 1800s huấn luyện.

Nguyên nhân: ở bản gốc (commit `05d3484`), validation chỉ chạy bất đồng bộ khi `algorithm == "rainbow"` và device là CPU. Mọi trường hợp còn lại (CUDA, hybrid, 3dqn) chạy inline trong learner. Learner đứng yên suốt thời gian validation, GPU không được cập nhật và workers không có trọng số mới. Commit đó không ghi lý do cho điều kiện "chỉ CPU".

## 2. Validation làm gì

- Cứ mỗi `validate_every` episode hoàn thành (và một lần ở cuối), đóng băng trọng số hiện tại.
- Chạy 2 episode đầy đủ trên 2 seed giữ riêng (`validation_seed_base+1` và `+2`), chọn hành động greedy, không lưu replay, không học.
- Điểm là trung bình hai seed của chi phí reward (với r1: `system_time_integral / r1_ref` chia cho `horizon * step_seconds`). Thấp hơn là tốt hơn.
- Nếu điểm tốt hơn best hiện tại thì ghi đè `best_weights.pt`, và ghi một dòng vào `validation_log.csv` (file không có dòng tiêu đề).
- Validation không ảnh hưởng quá trình huấn luyện. Nó chỉ chọn trọng số tốt nhất để lưu.

## 3. Thay đổi

- Hai process validation riêng (mỗi seed một process, spawn, có môi trường simulation riêng, actor chạy CPU, 1 thread torch).
- Hàng đợi FIFO `validation_queue`. Learner đẩy job vào rồi tiếp tục train.
- `collect_validation()` không chặn, nhặt kết quả đã xong và cập nhật best.
- Cuối huấn luyện gọi `collect_validation(wait=True)` để chờ nốt hàng đợi trước khi chốt `best_weights.pt`.
- `pending_validations` được lưu trong snapshot để resume.
- Backend GAMA vẫn chạy validation inline.
- Process validation được tắt cùng process workers trong `finally`.

## 4. Thí nghiệm

Chạy thật trên Colab, GPU T4, 2 vCPU. So sánh bản gốc (`orig`) và bản mới (`new`), thứ tự xen kẽ để tránh lệch do thứ tự chạy.

### 4.1 Cấu hình nhẹ (480s, 2 và 6 workers)

- Bản gốc: learner bị chặn 217–224s (2 workers) và 402–414s (6 workers) trong 480s.
- Bản mới: không chậm hơn trong thời gian budget ở cả hai cấu hình.

### 4.2 Cấu hình nặng (600s, 16 workers, RAM đo bằng psutil)

| | orig, validate mỗi 5 ep | new, mỗi 5 ep | orig, mỗi 2 ep | new, mỗi 2 ep |
|---|---|---|---|---|
| Steps tại 300s | 76 | 266 | 57 | 266 |
| Steps tại 600s | 456 | 551 | 285 | 589 |
| Decisions tổng | 711 / 695 | 806 / 792 | 581 | 796 |
| Learner bị chặn | 407–410s | 0 | 428s | 0 |
| Chờ validation sau khi hết budget | 9–11s | 126–134s | 9s | 362s |
| Tổng thời gian chạy | khoảng 645s | khoảng 750s | 637s | khoảng 981s |
| Decisions / giây tổng | 1.07–1.10 | 1.06 | 0.91 | 0.81 |
| Validation trễ so với training (episode) | 0 | 24 giảm dần về 4 | 0 | 27 giảm dần về 2 |
| RAM đỉnh | 7.8 GB | 8.4 GB | 7.7 GB | 8.5 GB |

Hai lượt `new` ở cùng cấu hình cho kết quả gần như y hệt (cùng chuỗi trễ), nên hành vi ổn định. Điểm best của hai bản đều quanh 1.9, nên chất lượng chọn best không đổi.

## 5. Kết luận

**Được:**
- Learner không còn bị chặn bởi validation. Bản gốc mất khoảng 407–428s trong 600s để chờ.
- Trong cùng budget huấn luyện, bản mới làm được nhiều hơn: decisions cao hơn 13% (validate mỗi 5 ep) và 37% (mỗi 2 ep). Steps tại 300s là 266 so với 76.

**Giá phải trả:**
- Validation trễ so với training, tới 24–28 episode khi CPU bị chia cho nhiều workers.
- Sau khi hết budget, phải chờ nốt hàng đợi validation: 126–134s (validate mỗi 5 ep) và 362s (mỗi 2 ep). Trên tổng thời gian chạy, bản mới không nhanh hơn: 1.06 so với 1.07–1.10, và 0.81 so với 0.91. Điều này khớp với cảm giác bản mới "chạy chậm hơn" khi tính cả thời gian chờ cuối.
- RAM tăng khoảng 0.7 GB do 2 process validation.
- `training_summary.json` hiện tính `train_wall_seconds` gồm cả thời gian chờ cuối, nên `decisions_per_second` bị thấp hơn thực tế.

**Nguyên nhân hàng đợi dồn:** 2 process validation tranh CPU với 16 workers trên 2 vCPU, nên mỗi lần validation chậm. Khi training dừng thì workers rảnh nhưng validation vẫn chỉ chạy bằng 2 process.

## 6. Chưa kiểm chứng

- 40 workers trên máy nhiều core (cấu hình thật).
- Cấu hình thật: `validate_every=50`, horizon 8000. Mình ước đoán hàng đợi dồn ít hơn mức đã ép trong thí nghiệm, nhưng chưa đo.
- Resume từ checkpoint có `pending_validations`.
- Đường chạy backend GAMA.
- Thời gian một lần validation chưa được tách thành simulation, suy luận actor và overhead pickle.

## 7. Đề xuất bước tiếp theo (chưa làm)

1. Khi training dừng, cho các job còn lại chạy song song trên toàn bộ core rảnh. Nhắm thẳng vào thời gian chờ cuối.
2. Giới hạn hàng đợi: giữ job mới nhất, bỏ job cũ chưa chạy. Đổi lại log validation thưa hơn khi quá tải.
3. Dùng `os.nice(10)` cho process validation, hoặc giảm còn 1 process khi `cpu_count` nhỏ hơn số workers + 2.
4. Tách thời gian chờ validation cuối khỏi `train_wall_seconds`, in thêm độ trễ validation trong dòng `[Validation]` và `training_summary.json`.
5. Các cách giảm công việc validation (1 seed, horizon ngắn hơn, `validate_every` lớn hơn) đổi độ chính xác hoặc độ dày log nên cần quyết định riêng.
6. Chạy lại `ab_stress_gpu.ipynb` với số vCPU và số workers của máy thật.

## 8. Ghi chú tái lập

- Notebook Colab nằm trong Drive: `ab_validation_gpu`, `train_orig_gpu`, `train_new_gpu`, `ab_multi_gpu`, `ab_stress_gpu`.
- Script sinh notebook nằm ở thư mục scratchpad của phiên (`gen.py`, `gen2.py`, `gen3.py`) và có thể đã bị xóa.
- Script đo của mình đọc `validation_log.csv` với giả định sai là có dòng tiêu đề, nên số hàng đếm lệch 1. Việc này không ảnh hưởng các số về độ trễ, thời gian chờ và decisions ở trên.
