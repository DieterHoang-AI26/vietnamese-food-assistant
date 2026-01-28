# Vietnamese Food Embedding Fine-tuning

Hướng dẫn fine-tune embedding model để cải thiện hệ thống RAG cho Vietnamese food domain.

## 🎯 Mục tiêu

Thay vì sử dụng hardcode/dictionary mapping, chúng ta fine-tune embedding model để:

1. **Hiểu lỗi chính tả**: "tráo" → "cháo"
2. **Hiểu ngữ cảnh**: "có món mực không" → tìm món mực
3. **Hiểu ý định phủ định**: "ngán cá" → tránh món cá
4. **Cải thiện semantic search** cho Vietnamese food domain

## 🚀 Cách chạy

### Option 1: Chạy full pipeline (khuyến nghị)

```bash
# Từ thư mục gốc của project
cd embedding_finetuning

# Cài đặt dependencies và chạy fine-tuning
./setup_and_run_finetuning.sh

# Hoặc manual:
pip install -r requirements_finetuning.txt
python run_fine_tuning.py
```

### Option 2: Test đơn giản trước

```bash
# Từ thư mục embedding_finetuning
cd embedding_finetuning
python test_finetuning_simple.py
```

### Option 3: Chỉ fine-tune model

```bash
# Từ thư mục embedding_finetuning
cd embedding_finetuning
python fine_tune_embedding.py
```

## 📊 Quá trình Fine-tuning

### 1. Tạo Training Data

- **Positive pairs**: Tên món ăn ↔ Mô tả
- **Ingredient pairs**: "có món [ingredient]" ↔ Món chứa ingredient
- **Spelling correction**: "tráo" ↔ "Cháo Thập Cẩm"
- **Natural queries**: "có món trứng không" ↔ "Cơm Trứng Chiên"
- **Negative pairs**: Món không liên quan

### 2. Training Process

- **Base model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Loss function**: CosineSimilarityLoss
- **Epochs**: 3 (có thể điều chỉnh)
- **Batch size**: 16
- **Evaluation**: Continuous evaluation trên test set

### 3. Model Output

- **Location**: `../models/vietnamese-food-embedding/`
- **Format**: SentenceTransformer model
- **Integration**: Tự động tích hợp vào RAG system

## 🧪 Test Cases

Hệ thống sẽ test các case khó:

```python
test_cases = [
    "cháo",                              # Exact match
    "tráo",                              # Spelling error
    "có món trứng không",                # Natural question
    "có món mực không",                  # Ingredient search
    "tôi ngán cá có món bò không",       # Negative context
    "cà phê sữa đá",                     # Multi-word
    "món nóng nào ngon"                  # Descriptive query
]
```

## 📈 Expected Improvements

### Before Fine-tuning (Base Model)
```
Query: "tráo"
Results:
1. Đại Hồng Bào - 0.695
2. Cơm Canh Khổ Qua - 0.687  
3. Nụ Hoa Trà - 0.637
```

### After Fine-tuning
```
Query: "tráo" 
Results:
1. Cháo Thập Cẩm & Sữa Chua - 0.890  ✅
2. Đại Hồng Bào - 0.695
3. Nụ Hoa Trà - 0.637
```

## ⚙️ Configuration

Model sẽ tự động được sử dụng thông qua config trong `src/config.py`:

```python
@dataclass
class DatabaseConfig:
    fine_tuned_model_path: Optional[str] = "models/vietnamese-food-embedding"
```

## 🔧 Troubleshooting

### Lỗi thiếu dependencies
```bash
pip install torch sentence-transformers transformers
```

### Lỗi GPU memory
- Giảm batch_size trong `fine_tune_embedding.py`
- Hoặc chạy trên CPU (chậm hơn nhưng vẫn work)

### Model không load được
- Kiểm tra path: `../models/vietnamese-food-embedding/`
- Chạy lại fine-tuning nếu bị corrupt

## 📁 File Structure

```
├── embedding_finetuning/
│   ├── fine_tune_embedding.py          # Core fine-tuning logic
│   ├── run_fine_tuning.py             # Full pipeline
│   ├── test_finetuning_simple.py      # Simple test
│   ├── requirements_finetuning.txt    # Dependencies
│   ├── setup_and_run_finetuning.sh   # Setup script
│   └── FINETUNING_README.md          # This file
└── models/
    └── vietnamese-food-embedding/      # Output model
        ├── config.json
        ├── pytorch_model.bin
        └── ...
```

## 🎉 Kết quả mong đợi

Sau khi fine-tune, hệ thống sẽ:

1. ✅ Hiểu lỗi chính tả: "tráo" → tìm được "Cháo Thập Cẩm"
2. ✅ Hiểu câu hỏi tự nhiên: "có món mực không" → tìm được "Cơm Mực Xào Sả Ớt"  
3. ✅ Cải thiện semantic search cho Vietnamese food domain
4. ✅ Không cần hardcode/dictionary mapping

## 🔄 Tích hợp vào hệ thống

Sau khi fine-tune xong, hệ thống RAG sẽ tự động:

1. Load fine-tuned model thay vì base model
2. Sử dụng improved embeddings cho vector search
3. Cải thiện kết quả tìm kiếm mà không cần thay đổi code logic

Chạy test với hệ thống mới:

```bash
# Từ thư mục gốc của project
python Test_Final.py
```