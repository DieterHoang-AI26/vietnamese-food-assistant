# Cấu trúc thư mục Fine-tuning Embedding

Thư mục `embedding_finetuning/` chứa tất cả các file liên quan đến việc tinh chỉnh mô hình nhúng (embedding) cho Vietnamese food domain.

## 📁 Cấu trúc thư mục

```
embedding_finetuning/
├── fine_tune_embedding.py          # Core fine-tuning logic
├── run_fine_tuning.py             # Pipeline hoàn chỉnh với so sánh
├── test_finetuning_simple.py      # Test đơn giản (1 epoch)
├── requirements_finetuning.txt    # Dependencies cho fine-tuning
├── setup_and_run_finetuning.sh   # Script setup tự động
├── FINETUNING_README.md          # Hướng dẫn chi tiết
└── README_STRUCTURE.md           # File này
```

## 🎯 Mục đích từng file

### `fine_tune_embedding.py`
- **Chức năng**: Core logic để fine-tune embedding model
- **Class chính**: `VietnameseFoodEmbeddingTrainer`
- **Tính năng**:
  - Tạo training pairs từ menu data
  - Xử lý lỗi chính tả (tráo → cháo)
  - Tạo natural language queries
  - Fine-tune với CosineSimilarityLoss

### `run_fine_tuning.py`
- **Chức năng**: Pipeline hoàn chỉnh với so sánh performance
- **Class chính**: `FineTuningPipeline`
- **Workflow**:
  1. Test base model performance
  2. Chạy fine-tuning
  3. Test fine-tuned model
  4. So sánh kết quả
  5. Cập nhật system config

### `test_finetuning_simple.py`
- **Chức năng**: Test nhanh với 1 epoch
- **Mục đích**: Kiểm tra concept trước khi chạy full training
- **Ưu điểm**: Nhanh, ít tài nguyên

### `requirements_finetuning.txt`
- **Chức năng**: Dependencies cần thiết cho fine-tuning
- **Bao gồm**: torch, sentence-transformers, transformers, etc.

### `setup_and_run_finetuning.sh`
- **Chức năng**: Script tự động setup và chạy
- **Tính năng**:
  - Tạo virtual environment
  - Cài đặt dependencies
  - Kiểm tra GPU
  - Chạy fine-tuning pipeline

## 🚀 Cách sử dụng

### Từ thư mục gốc của project:

```bash
# Option 1: Chạy full pipeline
cd embedding_finetuning
./setup_and_run_finetuning.sh

# Option 2: Test nhanh
cd embedding_finetuning  
python test_finetuning_simple.py

# Option 3: Manual setup
cd embedding_finetuning
pip install -r requirements_finetuning.txt
python run_fine_tuning.py
```

## 📊 Output

Sau khi chạy, model sẽ được lưu tại:
- `../models/vietnamese-food-embedding/` (cho full training)
- `../models/simple-vietnamese-food-embedding/` (cho simple test)

## 🔗 Tích hợp với hệ thống chính

Model được tự động tích hợp thông qua:
- `../src/config.py`: Cấu hình đường dẫn model
- `../src/rag_engine.py`: Load fine-tuned model tự động
- Hệ thống RAG sẽ ưu tiên sử dụng fine-tuned model nếu có

## ⚠️ Lưu ý

1. **Đường dẫn**: Tất cả đường dẫn được cấu hình relative từ thư mục `embedding_finetuning/`
2. **Dependencies**: Cần cài đặt thêm dependencies cho fine-tuning
3. **GPU**: Khuyến nghị sử dụng GPU để tăng tốc training
4. **Memory**: Fine-tuning cần RAM/VRAM đủ lớn

## 🎉 Kết quả mong đợi

Sau khi fine-tune thành công:
- "tráo" → tìm được "Cháo Thập Cẩm"
- "có món mực không" → tìm được "Cơm Mực Xào Sả Ớt"
- Cải thiện semantic search cho Vietnamese food domain
- Không cần hardcode/dictionary mapping