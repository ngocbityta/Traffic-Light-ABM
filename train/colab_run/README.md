# Chạy dài trên Colab G4, sao lưu để chạy tiếp ở bất kỳ đâu

Cấu hình chung ở `run_config.sh` (`RUN_NAME`, `TOTAL_STEPS`, `G4_HOURS`, số worker…).
`TOTAL_STEPS` và các tham số học trong `TRAIN_ARGS` phải giữ nguyên suốt cả run;
số worker, thiết bị, giới hạn giờ được phép đổi khi chạy tiếp.

## Chạy trên G4 — `run_g4_long.command` (nhấp đúp trên Mac)
1. Đóng gói code hiện tại và tạo runtime G4.
2. Nếu `train/results/<RUN_NAME>` trên Mac có checkpoint: tải lên và chạy tiếp từ đó.
   Nếu không: chạy tiếp từ bản trên Google Drive, hoặc bắt đầu run mới.
3. `colab drivemount`: bấm đồng ý quyền Google Drive trong trình duyệt.
4. Train `G4_HOURS` giờ. Trong lúc chạy:
   - mỗi `SYNC_MINUTES` (15) phút, chép run lên `MyDrive/Traffic-Light-ABM/runs/<RUN_NAME>`;
   - mỗi `DOWNLOAD_MINUTES` (60) phút, tải về `train/results/<RUN_NAME>.g4_snapshots/latest.tar`.
5. Khi xong: cài bản cuối vào `train/results/<RUN_NAME>` và tắt runtime (kể cả khi lỗi).

Checkpoint mỗi 500 episode (~9 phút trên G4). Nếu phiên bị cắt đột ngột, chỉ mất
phần sau lần sao lưu gần nhất (≤ ~25 phút).

## Chạy tiếp trên Mac — `run_mac_continue.command`
Chọn bản có nhiều quyết định nhất trong:
- `train/results/<RUN_NAME>`;
- `train/results/<RUN_NAME>.g4_snapshots/latest.tar`;
- Google Drive for desktop (`~/Library/CloudStorage/GoogleDrive-*/My Drive/Traffic-Light-ABM/runs/<RUN_NAME>`);
- thư mục tải tay từ drive.google.com về `~/Downloads/<RUN_NAME>`.

Bản cũ được giữ lại thành `<RUN_NAME>.before_<thời gian>`. Sau đó build bản dịch
Cython và train tiếp với `MAC_WORKERS` worker, checkpoint mỗi 50 episode
(~12 phút trên Mac), dừng khi đủ `TOTAL_STEPS` hoặc sau `MAC_HOURS`.

## Kiểm tra — `test_sync_resume.command`
Diễn tập trên Mac: chạy ngắn, sao lưu mỗi 10 s, `kill -9` cả tiến trình train
(như runtime hết CU), chọn bản sao lưu mới nhất và chạy tiếp với số worker khác.
In `PASS` nếu chạy tiếp thành công.

## Lưu ý
- Mac và Colab phải chạy **cùng mã nguồn**: checkpoint ghi hash mã nguồn và từ chối
  chạy tiếp nếu khác. Nên commit trước khi bắt đầu run dài.
- Chỉ một nơi được train một run tại một thời điểm.
