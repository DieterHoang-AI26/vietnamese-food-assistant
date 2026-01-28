#!/usr/bin/env python3
"""
Simple test script để kiểm tra fine-tuning process

Chạy một phiên bản đơn giản của fine-tuning để test concept.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

# Add paths for imports
current_dir = Path(__file__).parent
root_dir = current_dir.parent if current_dir.name == "embedding_finetuning" else current_dir
sys.path.insert(0, str(root_dir))

try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from torch.utils.data import DataLoader
    import torch
    
    print("✅ Tất cả dependencies đã được cài đặt")
except ImportError as e:
    print(f"❌ Thiếu dependency: {e}")
    print("Vui lòng chạy: pip install -r embedding_finetuning/requirements_finetuning.txt")
    sys.exit(1)


def create_simple_training_data() -> List[InputExample]:
    """Tạo training data đơn giản để test."""
    print("📊 Tạo simple training data...")
    
    # Determine data path
    current_dir = Path(__file__).parent
    if current_dir.name == "embedding_finetuning":
        data_path = current_dir.parent / "data" / "processed_menu_v2.json"
    else:
        data_path = current_dir / "data" / "processed_menu_v2.json"
    
    # Load menu data
    with open(data_path, 'r', encoding='utf-8') as f:
        menu_data = json.load(f)
    
    training_examples = []
    
    # Tạo một số positive pairs đơn giản
    for dish in menu_data[:10]:  # Chỉ lấy 10 món đầu để test nhanh
        name = dish.get('name_vi', '')
        desc = dish.get('description', '')
        
        if name and desc:
            # Positive pair: tên và mô tả
            training_examples.append(
                InputExample(texts=[name, desc], label=0.9)
            )
            
            # Positive pair: từ khóa và món ăn
            if 'cháo' in name.lower():
                training_examples.append(
                    InputExample(texts=["cháo", f"{name} {desc}"], label=0.9)
                )
                training_examples.append(
                    InputExample(texts=["tráo", f"{name} {desc}"], label=0.8)  # Lỗi chính tả
                )
            
            if 'trứng' in name.lower():
                training_examples.append(
                    InputExample(texts=["có món trứng không", f"{name} {desc}"], label=0.8)
                )
            
            if 'mực' in name.lower():
                training_examples.append(
                    InputExample(texts=["có món mực không", f"{name} {desc}"], label=0.8)
                )
    
    # Thêm một số negative pairs
    for i in range(5):
        dish1 = menu_data[i]
        dish2 = menu_data[i + 5]
        
        if dish1.get('category') != dish2.get('category'):
            training_examples.append(
                InputExample(
                    texts=[dish1.get('name_vi', ''), f"{dish2.get('name_vi', '')} {dish2.get('description', '')}"],
                    label=0.1
                )
            )
    
    print(f"✅ Đã tạo {len(training_examples)} training examples")
    return training_examples


def run_simple_finetuning():
    """Chạy fine-tuning đơn giản."""
    print("🚀 Bắt đầu simple fine-tuning...")
    
    # Load base model
    base_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"📦 Loading base model: {base_model}")
    
    model = SentenceTransformer(base_model)
    
    # Tạo training data
    training_examples = create_simple_training_data()
    
    # Tạo evaluation data (một phần của training data)
    eval_examples = training_examples[:5]
    
    # Tạo DataLoader
    train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=8)
    
    # Loss function
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Evaluator
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        eval_examples, name='simple-eval'
    )
    
    # Tạo output directory
    current_dir = Path(__file__).parent
    if current_dir.name == "embedding_finetuning":
        output_path = current_dir.parent / "models" / "simple-vietnamese-food-embedding"
    else:
        output_path = current_dir / "models" / "simple-vietnamese-food-embedding"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    print("🔥 Bắt đầu training (1 epoch để test nhanh)...")
    
    # Fine-tune với 1 epoch để test
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=1,  # Chỉ 1 epoch để test nhanh
        evaluation_steps=50,
        warmup_steps=10,
        output_path=output_path,
        save_best_model=True,
        show_progress_bar=True
    )
    
    print(f"✅ Fine-tuning hoàn thành! Model saved tại: {output_path}")
    return model


def test_model_improvement():
    """Test xem model có cải thiện không."""
    print("\n🧪 Testing model improvement...")
    
    # Load base model
    base_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Load fine-tuned model
    current_dir = Path(__file__).parent
    if current_dir.name == "embedding_finetuning":
        ft_model_path = current_dir.parent / "models" / "simple-vietnamese-food-embedding"
    else:
        ft_model_path = current_dir / "models" / "simple-vietnamese-food-embedding"
    if Path(ft_model_path).exists():
        ft_model = SentenceTransformer(ft_model_path)
    else:
        print("❌ Fine-tuned model không tồn tại!")
        return
    
    # Test queries
    test_queries = [
        "cháo",
        "tráo",  # Lỗi chính tả
        "có món trứng không"
    ]
    
    # Load menu data để test
    current_dir = Path(__file__).parent
    if current_dir.name == "embedding_finetuning":
        data_path = current_dir.parent / "data" / "processed_menu_v2.json"
    else:
        data_path = current_dir / "data" / "processed_menu_v2.json"
    
    with open(data_path, 'r', encoding='utf-8') as f:
        menu_data = json.load(f)
    
    # Tạo document texts
    doc_texts = []
    doc_names = []
    for dish in menu_data:
        doc_text = f"{dish.get('name_vi', '')} {dish.get('description', '')}"
        doc_texts.append(doc_text)
        doc_names.append(dish.get('name_vi', ''))
    
    print("\n📊 So sánh kết quả:")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)
        
        # Base model results
        base_query_emb = base_model.encode([query])
        base_doc_embs = base_model.encode(doc_texts)
        base_similarities = base_model.similarity(base_query_emb, base_doc_embs)[0]
        base_top3 = base_similarities.argsort(descending=True)[:3]
        
        print("📦 Base Model:")
        for i, idx in enumerate(base_top3, 1):
            score = base_similarities[idx].item()
            name = doc_names[idx]
            print(f"   {i}. {name} - {score:.3f}")
        
        # Fine-tuned model results
        ft_query_emb = ft_model.encode([query])
        ft_doc_embs = ft_model.encode(doc_texts)
        ft_similarities = ft_model.similarity(ft_query_emb, ft_doc_embs)[0]
        ft_top3 = ft_similarities.argsort(descending=True)[:3]
        
        print("\n🎯 Fine-tuned Model:")
        for i, idx in enumerate(ft_top3, 1):
            score = ft_similarities[idx].item()
            name = doc_names[idx]
            print(f"   {i}. {name} - {score:.3f}")


def main():
    """Hàm chính."""
    print("🍜 SIMPLE VIETNAMESE FOOD EMBEDDING FINE-TUNING TEST")
    print("=" * 60)
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")
    
    try:
        # Chạy fine-tuning
        model = run_simple_finetuning()
        
        # Test improvement
        test_model_improvement()
        
        print("\n🎉 Simple fine-tuning test hoàn thành!")
        print("📁 Model đã được lưu tại:", str(output_path))
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()