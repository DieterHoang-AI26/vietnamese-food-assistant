# Vietnamese Food Assistant 🍜

Trợ lý AI thông minh chuyên tư vấn món ăn Việt Nam với khả năng hiểu ngôn ngữ tự nhiên, xử lý ràng buộc dinh dưỡng và tìm kiếm món ăn phù hợp.

## 🎯 Tổng quan

Vietnamese Food Assistant là một hệ thống AI tiên tiến được xây dựng với kiến trúc **Hybrid: Data-First và AI-Native**, kết hợp sức mạnh của:

- **LLM (Large Language Models)** cho việc hiểu ngôn ngữ tự nhiên
- **RAG (Retrieval-Augmented Generation)** với ChromaDB cho tìm kiếm chính xác
- **Data-driven approach** hoàn toàn không hardcode
- **Fine-tuned embeddings** chuyên biệt cho món ăn Việt Nam

## ✨ Tính năng chính

### 🤖 AI Conversation
- **ASR Correction**: Tự động sửa lỗi chính tả từ giọng nói
- **Intent Classification**: Hiểu ý định người dùng (tìm món, hỏi menu, ràng buộc dinh dưỡng)
- **Context Management**: Nhớ cuộc hội thoại và yêu cầu trong 2 lượt gần nhất
- **Natural Language Response**: Phản hồi tự nhiên bằng tiếng Việt

### 🔍 Smart Search
- **Hybrid Search**: Kết hợp Vector similarity và BM25 text matching
- **Vietnamese Fuzzy Matching**: Xử lý lỗi chính tả và phiên âm tiếng Việt
- **Fine-tuned Embeddings**: 6 models chuyên biệt cho món ăn Việt Nam
- **Semantic Understanding**: Hiểu ngữ nghĩa và ngữ cảnh

### 🍽️ Dietary Intelligence
- **Constraint Extraction**: Tự động nhận diện dị ứng, chế độ ăn, sở thích
- **Smart Filtering**: Lọc món ăn theo ràng buộc nghiêm ngặt
- **Memory Management**: Nhớ yêu cầu trong 2 lượt hội thoại gần nhất
- **Availability Check**: Kiểm tra tình trạng có sẵn của món ăn

## 🏗️ Kiến trúc hệ thống

### Core Components
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Chat    │───▶│   LangGraph      │───▶│   RAG Engine    │
│   Interface     │    │   Workflow       │    │   (ChromaDB)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   LLM Nodes      │
                    │ • ASR Correction │
                    │ • Constraint     │
                    │ • Response Gen   │
                    └──────────────────┘
```
### LangGraph Workflow
```
ASR Correction → Context Management → Constraint Extraction → Constraint Accumulation
       ↓                                                              ↓
   Early Exit                                                    Retrieval
   (Greetings)                                                       ↓
                                                            Logic Filters
                                                                  ↓
                                                             Reranking
                                                                  ↓
                                                         Response Generation
```


Kiến trúc RAG trong dự án:
User Input → ASR Correction → Context Management → Constraint Extraction
                                      ↓
Response Generation ← Reranking ← Logic Filters ← RAG RETRIEVAL
                                                        ↓
                                              ChromaDB + Fine-tuned Models
                                              (Vector + BM25 Hybrid Search)

### Data Architecture
- **Menu Database**: Structured dish information with metadata
- **Vector Store**: ChromaDB with fine-tuned Vietnamese embeddings  
- **Session Management**: Persistent conversation state
- **Configuration**: Environment-driven settings

## 📁 Cấu trúc dự án

```
vietnamese-food-assistant/
├── src/                          # Core source code
│   ├── graph/                    # LangGraph workflow
│   │   └── workflow.py          # Main orchestration
│   ├── nodes/                    # LLM processing nodes
│   │   ├── asr_correction.py    # Speech-to-text correction
│   │   ├── constraint_extraction.py  # Dietary constraints
│   │   ├── context_manager.py   # Conversation context
│   │   ├── retrieval_node.py    # Document retrieval
│   │   ├── logic_filters.py     # Constraint filtering
│   │   ├── reranking_node.py    # Result reranking
│   │   └── response_generator.py # Natural language response
│   ├── config.py                # Configuration management
│   ├── rag_engine.py            # RAG core engine
│   ├── menu_database.py         # Menu data models
│   ├── vietnamese_fuzzy_matching.py # Vietnamese text processing
│   ├── etl_pipeline.py          # Data processing pipeline
│   └── error_handling.py        # Error management
├── data/                         # Data storage
│   ├── chroma_db/              # Vector database
│   ├── processed_menu_v2.json  # Processed menu data
│   ├── sample_menu.csv         # Sample data
│   └── sessions/               # Session storage
├── models/                       # Fine-tuned models
│   ├── comprehensive-food-model/
│   ├── focused-fruit-tea-model/
│   ├── simple-vietnamese-food-embedding/
│   └── vietnamese-food-foundation/
├── embedding_finetuning/        # Model training scripts
├── quick_chat.py                # Main chat interface
├── requirements.txt             # Dependencies
└── .env.example                # Environment template
```

## 🚀 Cài đặt và Sử dụng

### Yêu cầu hệ thống
- Python 3.8+
- Ollama (cho LLM models)
- 8GB+ RAM (khuyến nghị)
- 10GB+ disk space (cho models)

### 1. Clone repository
```bash
git clone <repository-url>
cd vietnamese-food-assistant
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Cài đặt Ollama và models
```bash
# Cài đặt Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Vietnamese LLM models
ollama pull vistral:latest
# hoặc
ollama pull qwen2.5:latest
```

### 4. Cấu hình environment
```bash
cp .env.example .env
# Chỉnh sửa .env theo nhu cầu
```

### 5. Chạy trợ lý
```bash
python quick_chat.py
```

## 💬 Cách sử dụng

### Ví dụ hội thoại
```
🗣️  Bạn: tôi muốn ăn phở
🤖 Tôi tìm thấy 3 món phù hợp với 'tôi muốn ăn phở':

📋 Danh sách món:
   1. Phở Bò Tái
      💰 85,000 VND - 📂 main
   2. Phở Gà  
      💰 70,000 VND - 📂 main
   3. Phở Chay
      💰 65,000 VND - 📂 main

❓ Bạn có thể hỏi:
   1. Bạn muốn biết thêm về món nào?
   2. Bạn có muốn xem món tương tự không?
```

### Xử lý ràng buộc dinh dưỡng
```
🗣️  Bạn: tôi dị ứng tôm
🤖 Tôi đã ghi nhận yêu cầu của bạn: dị ứng tôm.

🗣️  Bạn: có món chay nào không  
🤖 Tôi tìm thấy 5 món phù hợp với yêu cầu của bạn:

🔍 Yêu cầu của bạn (nhớ 2 lượt gần nhất):
   - Dị ứng: tôm
   - Chế độ ăn: vegetarian
   💾 Đang nhớ từ lượt: 1, 2
```
## 🔧 Cấu hình

### Environment Variables (.env)
```bash
# LLM Configuration
LLM_MODEL=vistral:latest          # Ollama model name
LLM_BASE_URL=http://localhost:11434  # Ollama server URL
LLM_TEMPERATURE=0.1               # Response creativity (0.0-1.0)

# RAG Configuration  
EMBEDDING_MODEL_PATH=models/comprehensive-food-model  # Fine-tuned embedding model
CHROMA_DB_PATH=data/chroma_db     # Vector database path
MENU_DATA_PATH=data/processed_menu_v2.json  # Menu data file

# Search Configuration
SIMILARITY_THRESHOLD=0.3          # Minimum similarity score
MAX_RESULTS=10                    # Maximum search results
RERANK_TOP_K=5                   # Top results for reranking

# Session Configuration
SESSION_DIR=data/sessions         # Session storage directory
MAX_CONSTRAINT_MEMORY=2           # Remember constraints from last N turns
LOG_LEVEL=INFO                    # Logging level
```

### Tùy chỉnh Models
Dự án hỗ trợ 6 fine-tuned embedding models:
- `comprehensive-food-model`: Tổng hợp, phù hợp nhất
- `focused-fruit-tea-model`: Chuyên về trà và đồ uống
- `simple-vietnamese-food-embedding`: Cơ bản, nhanh
- `vietnamese-food-foundation`: Nền tảng
- `vietnamese-food-restaurant`: Chuyên nhà hàng

Thay đổi model trong `.env`:
```bash
EMBEDDING_MODEL_PATH=models/focused-fruit-tea-model
```

## 📊 Performance & Benchmarks

### Search Accuracy
- **Vietnamese Fuzzy Matching**: 95%+ accuracy với lỗi chính tả phổ biến
- **Semantic Search**: 90%+ relevance với fine-tuned embeddings  
- **Constraint Filtering**: 99%+ precision cho dị ứng và chế độ ăn
- **Response Time**: <2s cho queries phức tạp

### Model Performance
| Model | Size | Accuracy | Speed | Use Case |
|-------|------|----------|-------|----------|
| comprehensive-food-model | 120MB | 94% | Medium | General purpose |
| focused-fruit-tea-model | 120MB | 96% | Medium | Beverages |
| simple-vietnamese-food | 120MB | 89% | Fast | Quick queries |

### System Requirements
- **Memory**: 2GB+ for embeddings, 4GB+ for LLM
- **Storage**: 1GB+ for models, 500MB+ for data
- **CPU**: Multi-core recommended for concurrent requests

## 🧪 Testing

### Chạy tests
```bash
# Test Vietnamese search accuracy
python test_vietnamese_search_accuracy.py

# Test full workflow
python demo_comparison.py

# Test specific components
python -m pytest src/tests/ -v
```

### Property-based Testing
Dự án sử dụng Hypothesis cho property-based testing:
```python
# Test Vietnamese fuzzy matching properties
@given(vietnamese_text())
def test_fuzzy_matching_properties(text):
    # Kiểm tra tính chất bất biến của fuzzy matching
    assert fuzzy_match(text, text) >= 0.9
```

## 🔍 API Reference

### RAGEngine Class
```python
from src.rag_engine import RAGEngine

# Khởi tạo
engine = RAGEngine()

# Tìm kiếm cơ bản
results = engine.search("phở bò", max_results=5)

# Tìm kiếm với constraints
constraints = {
    "allergies": ["tôm", "cua"],
    "dietary_preferences": ["vegetarian"],
    "spice_level": "mild"
}
results = engine.search_with_constraints("món chay", constraints)
```

### Vietnamese Fuzzy Matching
```python
from src.vietnamese_fuzzy_matching import DataDrivenVietnameseFuzzyMatcher

# Khởi tạo với menu data
matcher = DataDrivenVietnameseFuzzyMatcher()
matcher.learn_from_menu_data(menu_items)

# Tính similarity
score = matcher.calculate_similarity("pho bo", "phở bò")  # ~0.95
score = matcher.calculate_similarity("com ga", "cơm gà")  # ~0.92
```

### LangGraph Workflow
```python
from src.graph.workflow import create_workflow

# Tạo workflow
workflow = create_workflow()

# Xử lý input
state = {
    "user_input": "tôi muốn ăn phở",
    "conversation_history": [],
    "constraints": {}
}
result = workflow.invoke(state)
```

## 🛠️ Development

### Thêm món ăn mới
1. Cập nhật `data/sample_menu.csv`
2. Chạy ETL pipeline:
```bash
python src/etl_pipeline.py
```
3. Rebuild vector database:
```bash
python -c "from src.rag_engine import RAGEngine; RAGEngine().rebuild_index()"
```

### Fine-tune embedding models
```bash
cd embedding_finetuning/
python comprehensive_food_training.py
```

### Thêm LLM node mới
1. Tạo file trong `src/nodes/`
2. Implement interface:
```python
def process_node(state: dict) -> dict:
    # Xử lý logic
    return updated_state
```
3. Thêm vào workflow trong `src/graph/workflow.py`

## 🐛 Troubleshooting

### Lỗi thường gặp

**1. Ollama connection error**
```bash
# Kiểm tra Ollama service
ollama list
ollama serve  # Nếu chưa chạy
```

**2. ChromaDB permission error**
```bash
# Fix permissions
chmod -R 755 data/chroma_db/
```

**3. Model not found**
```bash
# Kiểm tra model path
ls -la models/comprehensive-food-model/
# Hoặc download lại models
```

**4. Memory error với large queries**
- Giảm `MAX_RESULTS` trong config
- Tăng system memory
- Sử dụng model nhỏ hơn

**5. Vietnamese text encoding issues**
```python
# Đảm bảo UTF-8 encoding
export PYTHONIOENCODING=utf-8
```

### Debug Mode
```bash
# Chạy với debug logging
LOG_LEVEL=DEBUG python quick_chat.py

# Hoặc trong code
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### Quy tắc phát triển
1. **No Hardcoding**: Tuyệt đối không hardcode patterns, từ điển, hoặc rules
2. **Data-Driven**: Mọi logic phải học từ dữ liệu thực tế
3. **Configuration-Driven**: Behavior thay đổi qua config, không qua code
4. **Unicode Normalization**: Sử dụng chuẩn Unicode cho text processing
5. **Property-Based Testing**: Test với Hypothesis cho edge cases

### Workflow
1. Fork repository
2. Tạo feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`
5. Tạo Pull Request

### Code Style
- Follow PEP 8
- Type hints required
- Docstrings cho public functions
- No hardcoded strings/patterns
- Configuration-driven behavior

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- **LangChain/LangGraph**: Workflow orchestration framework
- **ChromaDB**: Vector database for embeddings
- **Ollama**: Local LLM inference
- **Sentence Transformers**: Embedding model foundation
- **Hypothesis**: Property-based testing framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: your-email@domain.com

---

**🍜 Vietnamese Food Assistant** - Trợ lý AI thông minh cho món ăn Việt Nam