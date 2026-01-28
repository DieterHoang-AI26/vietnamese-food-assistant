#!/usr/bin/env python3
"""
Comprehensive food training - covers all food categories properly
"""

import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def create_comprehensive_training_data():
    """Create comprehensive training data for all food categories."""
    
    # Load menu data
    with open('../data/processed_menu_v2.json', 'r', encoding='utf-8') as f:
        menu_data = json.load(f)
    
    training_examples = []
    
    for dish in menu_data:
        dish_name = dish.get('name_vi', '')
        dish_desc = dish.get('description', '')
        dish_category = dish.get('category', '')
        
        if not dish_name:
            continue
            
        full_text = f"{dish_name} {dish_desc}"
        
        # 1. CRITICAL: Fruit vs Tea distinction (keep existing logic)
        if 'Đại Hồng Bào' in dish_name:
            fruit_queries = [
                "trái cây", "tráo cây", "fruit", "có trái cây không", 
                "món trái cây", "trai cay", "tôi muốn ăn trái cây"
            ]
            
            for query in fruit_queries:
                training_examples.append(
                    InputExample(texts=[query, full_text], label=0.0)
                )
            
            tea_queries = ["trà", "tea", "đồ uống", "giải khát"]
            for query in tea_queries:
                training_examples.append(
                    InputExample(texts=[query, full_text], label=0.9)
                )
        
        # 2. NEW: Comprehensive food category training
        
        # Bún dishes - HIGH similarity with bún queries
        if 'bún' in dish_name.lower():
            bun_queries = [
                "món bún", "bun", "có món bún không", 
                "tôi muốn ăn bún", "bún gì ngon"
            ]
            for query in bun_queries:
                training_examples.append(
                    InputExample(texts=[query, full_text], label=0.95)
                )
        
        # Cơm dishes - HIGH similarity with cơm queries  
        elif 'cơm' in dish_name.lower():
            com_queries = [
                "món cơm", "com", "có món cơm không",
                "tôi muốn ăn cơm", "cơm gì ngon"
            ]
            for query in com_queries:
                training_examples.append(
                    InputExample(texts=[query, full_text], label=0.95)
                )
        
        # Cháo dishes - HIGH similarity with cháo queries
        elif 'cháo' in dish_name.lower():
            chao_queries = [
                "món cháo", "chao", "tráo", "có món cháo không",
                "tôi muốn ăn cháo", "cháo gì ngon"
            ]
            for query in chao_queries:
                training_examples.append(
                    InputExample(texts=[query, full_text], label=0.95)
                )
        
        # Nước/Drinks - HIGH similarity with drink queries
        elif any(drink_word in dish_name.lower() for drink_word in ['nước', 'trà', 'cà phê']):
            drink_queries = [
                "đồ uống", "nước uống", "giải khát", 
                "có đồ uống không", "cho tôi món nước"
            ]
            for query in drink_queries:
                training_examples.append(
                    InputExample(texts=[query, full_text], label=0.9)
                )
        
        # 3. NEGATIVE examples - prevent wrong category matches
        
        # Bún dishes should NOT match cơm queries
        if 'bún' in dish_name.lower():
            wrong_queries = ["món cơm", "cơm gì ngon", "tôi muốn ăn cơm"]
            for query in wrong_queries:
                training_examples.append(
                    InputExample(texts=[query, full_text], label=0.1)
                )
        
        # Cơm dishes should NOT match bún queries
        elif 'cơm' in dish_name.lower():
            wrong_queries = ["món bún", "bún gì ngon", "tôi muốn ăn bún"]
            for query in wrong_queries:
                training_examples.append(
                    InputExample(texts=[query, full_text], label=0.1)
                )
        
        # Tea/drinks should NOT match food queries
        elif any(drink_word in dish_name.lower() for drink_word in ['nước', 'trà']):
            wrong_queries = ["món bún", "món cơm", "món cháo", "tôi muốn ăn"]
            for query in wrong_queries:
                training_examples.append(
                    InputExample(texts=[query, full_text], label=0.05)
                )
    
    # 4. Add synthetic examples for reinforcement
    synthetic_examples = [
        # Strong positive examples
        ("món bún", "Bún chả giò thơm ngon", 0.95),
        ("món cơm", "Cơm gà nướng mật ong", 0.95),
        ("món cháo", "Cháo thập cẩm bổ dưỡng", 0.95),
        ("đồ uống", "Nước chanh tươi mát", 0.9),
        
        # Strong negative examples
        ("món bún", "Trà có hương thơm đặc biệt", 0.0),
        ("món cơm", "Nước lọc chai tiện lợi", 0.0),
        ("đồ uống", "Cơm thịt kho trứng ngon", 0.0),
    ]
    
    for query, doc, score in synthetic_examples:
        training_examples.append(
            InputExample(texts=[query, doc], label=score)
        )
    
    print(f"Created {len(training_examples)} comprehensive training examples")
    return training_examples

def train_comprehensive_model():
    """Train comprehensive model for all food categories."""
    
    print("🎯 Comprehensive Food Category Training")
    print("="*50)
    
    # Load base model
    print("📦 Loading base model...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Create comprehensive training data
    print("🔧 Creating comprehensive training data...")
    training_examples = create_comprehensive_training_data()
    
    # Create data loader
    train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=8)
    
    # Define loss function
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Train with more epochs
    print("🚀 Training comprehensive model...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=4,  # More epochs for comprehensive learning
        warmup_steps=100,
        output_path="../models/comprehensive-food-model/",
        show_progress_bar=True
    )
    
    print("✅ Comprehensive model training complete!")
    return "../models/comprehensive-food-model/"

if __name__ == "__main__":
    model_path = train_comprehensive_model()
    print(f"🎯 Comprehensive model saved to: {model_path}")