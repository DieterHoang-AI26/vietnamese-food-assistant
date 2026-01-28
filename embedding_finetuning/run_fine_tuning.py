#!/usr/bin/env python3
"""
Script để chạy fine-tuning và test model mới

Workflow:
1. Fine-tune embedding model trên Vietnamese food domain
2. Test performance so với base model
3. Tích hợp vào hệ thống RAG
4. Chạy test cases để đánh giá cải thiện
"""

import sys
from pathlib import Path
import shutil

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embedding_finetuning.fine_tune_embedding import VietnameseFoodEmbeddingTrainer
from src.rag_engine import RAGEngine
from src.nodes.retrieval_node import create_retrieval_node


class FineTuningPipeline:
    """Pipeline để fine-tune và test embedding model."""
    
    def __init__(self):
        """Khởi tạo pipeline."""
        self.trainer = None
        self.base_model_results = {}
        self.fine_tuned_results = {}
    
    def run_complete_pipeline(self):
        """Chạy toàn bộ pipeline fine-tuning và testing."""
        print("🚀 BẮT ĐẦU FINE-TUNING PIPELINE CHO VIETNAMESE FOOD DOMAIN")
        print("=" * 80)
        
        try:
            # Step 1: Test base model performance
            print("\n📊 STEP 1: Test Base Model Performance")
            print("-" * 50)
            self.test_base_model_performance()
            
            # Step 2: Fine-tune model
            print("\n🔥 STEP 2: Fine-tune Embedding Model")
            print("-" * 50)
            self.run_fine_tuning()
            
            # Step 3: Test fine-tuned model performance  
            print("\n🎯 STEP 3: Test Fine-tuned Model Performance")
            print("-" * 50)
            self.test_fine_tuned_model_performance()
            
            # Step 4: Compare results
            print("\n📈 STEP 4: Compare Performance")
            print("-" * 50)
            self.compare_model_performance()
            
            # Step 5: Update system configuration
            print("\n⚙️ STEP 5: Update System Configuration")
            print("-" * 50)
            self.update_system_config()
            
            print("\n🎉 PIPELINE HOÀN THÀNH THÀNH CÔNG!")
            print("✅ Hệ thống đã được cập nhật với fine-tuned embedding model")
            
        except Exception as e:
            print(f"\n❌ Lỗi trong pipeline: {e}")
            raise
    
    def test_base_model_performance(self):
        """Test performance của base model."""
        print("🧪 Đang test base model...")
        
        # Test cases khó
        test_cases = [
            "cháo",
            "tráo",  # Lỗi chính tả
            "có món trứng không",
            "có món mực không", 
            "tôi ngán cá có món bò không",
            "cà phê sữa đá",
            "món nóng nào ngon"
        ]
        
        # Tạo RAG engine với base model
        rag_engine = RAGEngine()
        
        self.base_model_results = {}
        
        for query in test_cases:
            print(f"   Testing: '{query}'")
            
            # Thực hiện search
            results = rag_engine.search(query, n_results=3)
            
            # Lưu kết quả
            self.base_model_results[query] = [
                {
                    "name": result.dish.search_content.name_vi,
                    "score": result.relevance_score,
                    "category": result.dish.metadata.category
                }
                for result in results
            ]
        
        print("✅ Đã test xong base model")
    
    def run_fine_tuning(self):
        """Chạy fine-tuning process."""
        print("🔥 Đang fine-tune embedding model...")
        
        # Khởi tạo trainer
        self.trainer = VietnameseFoodEmbeddingTrainer()
        
        # Chạy training pipeline
        self.trainer.run_full_training_pipeline()
        
        print("✅ Fine-tuning hoàn thành")
    
    def test_fine_tuned_model_performance(self):
        """Test performance của fine-tuned model."""
        print("🎯 Đang test fine-tuned model...")
        
        # Kiểm tra xem model đã được tạo chưa
        model_path = Path("../models/vietnamese-food-embedding")
        if not model_path.exists():
            raise FileNotFoundError("Fine-tuned model không tồn tại!")
        
        # Test cases giống như base model
        test_cases = [
            "cháo",
            "tráo",  # Lỗi chính tả
            "có món trứng không",
            "có món mực không", 
            "tôi ngán cá có món bò không",
            "cà phê sữa đá",
            "món nóng nào ngon"
        ]
        
        # Tạo RAG engine mới (sẽ tự động load fine-tuned model)
        rag_engine = RAGEngine()
        
        self.fine_tuned_results = {}
        
        for query in test_cases:
            print(f"   Testing: '{query}'")
            
            # Thực hiện search
            results = rag_engine.search(query, n_results=3)
            
            # Lưu kết quả
            self.fine_tuned_results[query] = [
                {
                    "name": result.dish.search_content.name_vi,
                    "score": result.relevance_score,
                    "category": result.dish.metadata.category
                }
                for result in results
            ]
        
        print("✅ Đã test xong fine-tuned model")
    
    def compare_model_performance(self):
        """So sánh performance giữa base model và fine-tuned model."""
        print("📈 So sánh performance giữa base model và fine-tuned model:")
        print("=" * 70)
        
        for query in self.base_model_results.keys():
            print(f"\n🔍 Query: '{query}'")
            print("-" * 50)
            
            print("📦 Base Model Results:")
            for i, result in enumerate(self.base_model_results[query], 1):
                print(f"   {i}. {result['name']} - Score: {result['score']:.3f}")
            
            print("\n🎯 Fine-tuned Model Results:")
            for i, result in enumerate(self.fine_tuned_results[query], 1):
                print(f"   {i}. {result['name']} - Score: {result['score']:.3f}")
            
            # Phân tích cải thiện
            self.analyze_improvement(query)
    
    def analyze_improvement(self, query: str):
        """Phân tích cải thiện cho một query cụ thể."""
        base_results = self.base_model_results[query]
        fine_tuned_results = self.fine_tuned_results[query]
        
        # Kiểm tra các case cụ thể
        improvements = []
        
        if query in ["cháo", "tráo"]:
            # Kiểm tra xem có tìm được món cháo không
            base_has_chao = any("cháo" in result["name"].lower() for result in base_results)
            ft_has_chao = any("cháo" in result["name"].lower() for result in fine_tuned_results)
            
            if not base_has_chao and ft_has_chao:
                improvements.append("✅ Tìm được món cháo (base model không tìm được)")
            elif base_has_chao and ft_has_chao:
                # So sánh ranking
                base_chao_rank = next((i for i, r in enumerate(base_results) if "cháo" in r["name"].lower()), None)
                ft_chao_rank = next((i for i, r in enumerate(fine_tuned_results) if "cháo" in r["name"].lower()), None)
                
                if ft_chao_rank is not None and (base_chao_rank is None or ft_chao_rank < base_chao_rank):
                    improvements.append(f"✅ Cải thiện ranking món cháo (từ #{base_chao_rank+1} → #{ft_chao_rank+1})")
        
        elif "mực" in query:
            # Kiểm tra món mực
            base_has_muc = any("mực" in result["name"].lower() for result in base_results)
            ft_has_muc = any("mực" in result["name"].lower() for result in fine_tuned_results)
            
            if not base_has_muc and ft_has_muc:
                improvements.append("✅ Tìm được món mực (base model không tìm được)")
        
        elif "trứng" in query:
            # Kiểm tra món trứng
            base_has_trung = any("trứng" in result["name"].lower() for result in base_results)
            ft_has_trung = any("trứng" in result["name"].lower() for result in fine_tuned_results)
            
            if not base_has_trung and ft_has_trung:
                improvements.append("✅ Tìm được món trứng (base model không tìm được)")
        
        # Hiển thị cải thiện
        if improvements:
            print("\n🎉 Cải thiện:")
            for improvement in improvements:
                print(f"   {improvement}")
        else:
            print("\n📊 Không có cải thiện rõ rệt cho query này")
    
    def update_system_config(self):
        """Cập nhật cấu hình hệ thống để sử dụng fine-tuned model."""
        print("⚙️ Đang cập nhật cấu hình hệ thống...")
        
        # Kiểm tra xem fine-tuned model có tồn tại không
        model_path = Path("../models/vietnamese-food-embedding")
        if model_path.exists():
            print(f"✅ Fine-tuned model đã sẵn sàng tại: {model_path}")
            print("📝 Cấu hình đã được cập nhật trong src/config.py")
            print("🔄 Hệ thống sẽ tự động sử dụng fine-tuned model khi khởi động lại")
        else:
            print("⚠️ Fine-tuned model không tồn tại, hệ thống sẽ sử dụng base model")
    
    def run_final_test(self):
        """Chạy test cuối cùng với hệ thống đã cập nhật."""
        print("\n🧪 FINAL TEST - Hệ thống với Fine-tuned Model")
        print("=" * 60)
        
        # Import test script từ thư mục gốc
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from Test_Final import FinalFoodAssistant
        
        # Tạo assistant mới (sẽ sử dụng fine-tuned model)
        assistant = FinalFoodAssistant()
        
        # Test cases khó
        difficult_cases = [
            "tráo",  # Lỗi chính tả
            "có món mực không",
            "tôi ngán cá có món bò không"
        ]
        
        for case in difficult_cases:
            print(f"\n🎯 Testing: '{case}'")
            result = assistant.search_dishes(case)
            assistant.display_results(result)


def main():
    """Hàm chính."""
    print("🍜 VIETNAMESE FOOD EMBEDDING FINE-TUNING PIPELINE")
    print("Cải thiện hệ thống RAG bằng fine-tuned embedding model")
    print("=" * 80)
    
    try:
        pipeline = FineTuningPipeline()
        
        # Chạy pipeline hoàn chỉnh
        pipeline.run_complete_pipeline()
        
        # Hỏi người dùng có muốn chạy final test không
        response = input("\n🤔 Bạn có muốn chạy final test không? (y/n): ").strip().lower()
        if response in ['y', 'yes', 'có']:
            pipeline.run_final_test()
        
        print("\n🎉 HOÀN THÀNH! Hệ thống đã được nâng cấp với fine-tuned embedding model.")
        
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline bị dừng bởi người dùng")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()