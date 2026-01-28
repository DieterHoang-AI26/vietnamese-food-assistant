#!/usr/bin/env python3
"""
Fine-tune Embedding Model for Vietnamese Food Domain

Tạo và fine-tune embedding model chuyên biệt cho domain món ăn Việt Nam
để cải thiện khả năng hiểu ngữ cảnh và tìm kiếm của hệ thống RAG.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import torch


class VietnameseFoodEmbeddingTrainer:
    """Trainer để fine-tune embedding model cho Vietnamese food domain."""
    
    def __init__(self, base_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Khởi tạo trainer.
        
        Args:
            base_model_name: Tên model embedding gốc để fine-tune
        """
        self.base_model_name = base_model_name
        self.model = None
        self.training_data = []
        self.evaluation_data = []
        
        print(f"🤖 Khởi tạo Fine-tuning Trainer cho Vietnamese Food Domain")
        print(f"📦 Base model: {base_model_name}")
    
    def load_menu_data(self, data_path: str = "../data/processed_menu_v2.json") -> List[Dict]:
        """Tải dữ liệu menu để tạo training data."""
        print(f"📊 Đang tải dữ liệu menu từ {data_path}...")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            menu_data = json.load(f)
        
        print(f"✅ Đã tải {len(menu_data)} món ăn")
        return menu_data
    
    def create_training_pairs(self, menu_data: List[Dict]) -> List[InputExample]:
        """
        Tạo training pairs từ dữ liệu menu.
        
        Tạo các cặp (query, document, similarity_score) để train model hiểu:
        1. Tên món ăn và mô tả (positive pairs)
        2. Thành phần và món ăn chứa thành phần đó (positive pairs)  
        3. Câu hỏi tự nhiên và món ăn phù hợp (positive pairs)
        4. Negative pairs để model học phân biệt
        """
        print("🔧 Đang tạo training pairs...")
        
        training_examples = []
        
        for dish in menu_data:
            dish_name = dish.get('name_vi', '')
            dish_desc = dish.get('description', '')
            dish_category = dish.get('category', '')
            ingredients = dish.get('ingredients', [])
            
            # 1. Positive pairs: Tên món và mô tả
            if dish_name and dish_desc:
                training_examples.append(
                    InputExample(texts=[dish_name, dish_desc], label=0.9)
                )
            
            # 2. Positive pairs: Thành phần và món ăn
            for ingredient in ingredients[:3]:  # Chỉ lấy 3 thành phần chính
                if ingredient and len(ingredient.strip()) > 2:
                    # Query dạng "có món nào có [ingredient]"
                    query = f"có món nào có {ingredient}"
                    full_text = f"{dish_name} {dish_desc}"
                    training_examples.append(
                        InputExample(texts=[query, full_text], label=0.8)
                    )
                    
                    # Query dạng "món [ingredient]"
                    query = f"món {ingredient}"
                    training_examples.append(
                        InputExample(texts=[query, full_text], label=0.8)
                    )
            
            # 3. Positive pairs: Danh mục và món ăn
            if dish_category:
                category_queries = [
                    f"món {dish_category.lower()}",
                    f"có món {dish_category.lower()} nào không",
                    f"{dish_category.lower()}"
                ]
                
                for query in category_queries:
                    full_text = f"{dish_name} {dish_desc}"
                    training_examples.append(
                        InputExample(texts=[query, full_text], label=0.7)
                    )
            
            # 4. Positive pairs: Câu hỏi tự nhiên
            natural_queries = self._generate_natural_queries(dish)
            for query in natural_queries:
                full_text = f"{dish_name} {dish_desc}"
                training_examples.append(
                    InputExample(texts=[query, full_text], label=0.8)
                )
        
        # 5. Negative pairs: Món ăn không liên quan
        negative_pairs = self._create_negative_pairs(menu_data)
        training_examples.extend(negative_pairs)
        
        print(f"✅ Đã tạo {len(training_examples)} training pairs")
        return training_examples
    
    def _generate_natural_queries(self, dish: Dict) -> List[str]:
        """Tạo các câu hỏi tự nhiên cho món ăn."""
        dish_name = dish.get('name_vi', '')
        ingredients = dish.get('ingredients', [])
        category = dish.get('category', '')
        
        queries = []
        
        # Queries dựa trên tên món
        if dish_name:
            # Xử lý lỗi chính tả phổ biến
            if 'cháo' in dish_name.lower():
                queries.extend([
                    "tráo",  # Lỗi chính tả phổ biến
                    "có món cháo nào không",
                    "món cháo",
                    "cháo gì ngon"
                ])
            
            if 'cà phê' in dish_name.lower():
                queries.extend([
                    "ca phe",
                    "coffee", 
                    "có cà phê không",
                    "đồ uống có caffeine"
                ])
            
            if 'trứng' in dish_name.lower():
                queries.extend([
                    "trung",  # Lỗi chính tả
                    "có món nào có trứng không",
                    "món trứng"
                ])
            
            if 'mực' in dish_name.lower():
                queries.extend([
                    "muc",  # Không dấu
                    "có món mực không",
                    "hải sản",
                    "món mực"
                ])
        
        # Queries dựa trên thành phần chính
        main_ingredients = ingredients[:2] if ingredients else []
        for ingredient in main_ingredients:
            if ingredient and len(ingredient.strip()) > 2:
                queries.append(f"tôi muốn ăn {ingredient}")
                queries.append(f"có {ingredient} không")
        
        return queries[:5]  # Giới hạn số lượng queries
    
    def _create_negative_pairs(self, menu_data: List[Dict], num_negative: int = 200) -> List[InputExample]:
        """Tạo negative pairs để model học phân biệt."""
        negative_examples = []
        
        # Tạo random negative pairs
        for _ in range(num_negative):
            dish1 = random.choice(menu_data)
            dish2 = random.choice(menu_data)
            
            # Đảm bảo 2 món khác nhau và khác category
            if (dish1['id'] != dish2['id'] and 
                dish1.get('category') != dish2.get('category')):
                
                query = dish1.get('name_vi', '')
                doc = f"{dish2.get('name_vi', '')} {dish2.get('description', '')}"
                
                if query and doc:
                    negative_examples.append(
                        InputExample(texts=[query, doc], label=0.1)
                    )
        
        # Tạo specific negative cases
        specific_negatives = [
            ("tôi ngán cá", "Cơm Cá Kho Tộ Cơm, cá kho tộ đậm vị, rau luộc và canh theo ngày.", 0.0),
            ("không muốn hải sản", "Cơm Mực Xào Sả Ớt Mực tươi giòn xào sả ớt thơm lừng", 0.0),
            ("món chay", "Cơm Gà Nướng Mật Ong Cơm, đùi gà nướng mật ong và bắp cải xào", 0.1),
            ("đồ uống lạnh", "Trà Gừng Ấm Nóng Gừng tươi thái lát nấu cùng nước nóng", 0.1)
        ]
        
        for query, doc, score in specific_negatives:
            negative_examples.append(
                InputExample(texts=[query, doc], label=score)
            )
        
        return negative_examples
    
    def create_evaluation_data(self, menu_data: List[Dict]) -> List[InputExample]:
        """Tạo evaluation dataset."""
        print("📊 Đang tạo evaluation data...")
        
        eval_examples = []
        
        # Test cases cụ thể để đánh giá
        test_cases = [
            # Positive cases
            ("cháo", "Cháo Thập Cẩm & Sữa Chua", 0.9),
            ("tráo", "Cháo Thập Cẩm & Sữa Chua", 0.8),  # Lỗi chính tả
            ("có món trứng không", "Cơm Trứng Chiên", 0.8),
            ("có món mực không", "Cơm Mực Xào Sả Ớt", 0.8),
            ("cà phê sữa", "Cà Phê Sữa Đá", 0.9),
            
            # Negative cases  
            ("tôi ngán cá", "Cơm Cá Kho Tộ", 0.1),
            ("không muốn hải sản", "Cơm Mực Xào Sả Ớt", 0.1),
            ("món chay", "Cơm Gà Nướng Mật Ong", 0.1)
        ]
        
        for query, dish_name, score in test_cases:
            # Tìm món ăn trong data
            for dish in menu_data:
                if dish_name in dish.get('name_vi', ''):
                    full_text = f"{dish.get('name_vi', '')} {dish.get('description', '')}"
                    eval_examples.append(
                        InputExample(texts=[query, full_text], label=score)
                    )
                    break
        
        print(f"✅ Đã tạo {len(eval_examples)} evaluation examples")
        return eval_examples
    
    def fine_tune_model(self, training_examples: List[InputExample], 
                       evaluation_examples: List[InputExample],
                       output_path: str = "../models/vietnamese-food-embedding",
                       epochs: int = 3,
                       batch_size: int = 16) -> SentenceTransformer:
        """
        Fine-tune embedding model.
        
        Args:
            training_examples: Training data
            evaluation_examples: Evaluation data  
            output_path: Đường dẫn lưu model
            epochs: Số epochs
            batch_size: Batch size
            
        Returns:
            Fine-tuned model
        """
        print(f"🚀 Bắt đầu fine-tuning model...")
        print(f"📊 Training examples: {len(training_examples)}")
        print(f"📊 Evaluation examples: {len(evaluation_examples)}")
        
        # Load base model
        print(f"📦 Đang tải base model: {self.base_model_name}")
        model = SentenceTransformer(self.base_model_name)
        
        # Tạo DataLoader
        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=batch_size)
        
        # Định nghĩa loss function
        train_loss = losses.CosineSimilarityLoss(model)
        
        # Tạo evaluator
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            evaluation_examples, name='vietnamese-food-eval'
        )
        
        # Tạo thư mục output
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # Fine-tune
        print(f"🔥 Đang fine-tune với {epochs} epochs...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=500,
            warmup_steps=100,
            output_path=output_path,
            save_best_model=True,
            show_progress_bar=True
        )
        
        print(f"✅ Fine-tuning hoàn thành! Model đã được lưu tại: {output_path}")
        
        self.model = model
        return model
    
    def test_model_performance(self, model: SentenceTransformer, menu_data: List[Dict]):
        """Test performance của model sau khi fine-tune."""
        print("\n🧪 TESTING MODEL PERFORMANCE")
        print("=" * 50)
        
        # Test cases
        test_queries = [
            "cháo",
            "tráo",  # Lỗi chính tả
            "có món trứng không", 
            "có món mực không",
            "tôi ngán cá có món bò không",
            "cà phê sữa đá",
            "món nóng"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Encode query
            query_embedding = model.encode([query])
            
            # Encode tất cả món ăn
            dish_texts = []
            dish_names = []
            
            for dish in menu_data:
                dish_text = f"{dish.get('name_vi', '')} {dish.get('description', '')}"
                dish_texts.append(dish_text)
                dish_names.append(dish.get('name_vi', ''))
            
            dish_embeddings = model.encode(dish_texts)
            
            # Tính similarity
            similarities = model.similarity(query_embedding, dish_embeddings)[0]
            
            # Lấy top 3
            top_indices = similarities.argsort(descending=True)[:3]
            
            print("   Top 3 kết quả:")
            for i, idx in enumerate(top_indices, 1):
                score = similarities[idx].item()
                name = dish_names[idx]
                print(f"   {i}. {name} - Score: {score:.3f}")
    
    def run_full_training_pipeline(self):
        """Chạy toàn bộ pipeline training."""
        print("🚀 BẮT ĐẦU FINE-TUNING PIPELINE")
        print("=" * 60)
        
        try:
            # 1. Load dữ liệu
            menu_data = self.load_menu_data()
            
            # 2. Tạo training data
            training_examples = self.create_training_pairs(menu_data)
            
            # 3. Tạo evaluation data
            evaluation_examples = self.create_evaluation_data(menu_data)
            
            # 4. Fine-tune model
            model = self.fine_tune_model(training_examples, evaluation_examples)
            
            # 5. Test performance
            self.test_model_performance(model, menu_data)
            
            print("\n🎉 FINE-TUNING HOÀN THÀNH THÀNH CÔNG!")
            print("📁 Model đã được lưu tại: models/vietnamese-food-embedding")
            print("🔧 Để sử dụng model mới, cập nhật config trong src/config.py")
            
        except Exception as e:
            print(f"❌ Lỗi trong quá trình training: {e}")
            raise


def main():
    """Hàm chính để chạy fine-tuning."""
    print("🍜 VIETNAMESE FOOD EMBEDDING FINE-TUNING")
    print("Fine-tune embedding model cho Vietnamese food domain")
    print("=" * 60)
    
    # Kiểm tra GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Device: {device}")
    
    try:
        # Khởi tạo trainer
        trainer = VietnameseFoodEmbeddingTrainer()
        
        # Chạy training pipeline
        trainer.run_full_training_pipeline()
        
    except KeyboardInterrupt:
        print("\n⏹️ Training bị dừng bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")


if __name__ == "__main__":
    main()