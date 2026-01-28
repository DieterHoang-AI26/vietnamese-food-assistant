#!/usr/bin/env python3
"""
Focused training specifically for fruit vs tea distinction
"""

import json
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def create_focused_training_data():
    """Create focused training data for fruit vs tea distinction."""
    
    # Load menu data
    with open('../data/processed_menu_v2.json', 'r', encoding='utf-8') as f:
        menu_data = json.load(f)
    
    training_examples = []
    
    for dish in menu_data:
        dish_name = dish.get('name_vi', '')
        dish_desc = dish.get('description', '')
        dish_category = dish.get('category', '')
        
        full_text = f"{dish_name} {dish_desc}"
        
        # Special handling for "Đại Hồng Bào" - VERY STRONG negative examples
        if 'Đại Hồng Bào' in dish_name:
            fruit_queries = [
                "trái cây", "tráo cây", "fruit", "có trái cây không", 
                "món trái cây", "trai cay", "tôi muốn ăn trái cây",
                "có trái cây tươi không", "món tráng miệng trái cây"
            ]
            
            # VERY STRONG negative examples with score 0.0
            for query in fruit_queries:
                training_examples.append(
                    InputExample(texts=[query, full_text], label=0.0)
                )
            
            # Positive examples for tea queries
            tea_queries = [
                "trà", "tea", "đồ uống", "giải khát", "nước uống",
                "trà có hương trái cây", "đồ uống có vị trái cây"
            ]
            
            for query in tea_queries:
                training_examples.append(
                    InputExample(texts=[query, full_text], label=0.9)
                )
        
        # Handle other tea dishes with fruit flavor
        elif any(tea_keyword in dish_category.lower() for tea_keyword in ['trà', 'tea']):
            if any(fruit_flavor in dish_desc.lower() for fruit_flavor in ['hương trái cây', 'vị trái cây']):
                fruit_queries = ["trái cây", "fruit", "món trái cây"]
                
                for query in fruit_queries:
                    training_examples.append(
                        InputExample(texts=[query, full_text], label=0.0)
                    )
    
    # Add synthetic examples to reinforce the distinction
    synthetic_examples = [
        # Strong negative examples
        ("trái cây", "Trà có hương vị trái cây chín", 0.0),
        ("fruit", "Tea with fruit flavor", 0.0),
        ("món trái cây", "Đồ uống có hương thơm trái cây", 0.0),
        ("tôi muốn ăn trái cây", "Nước trà màu đỏ cam óng mật, vị êm mượt, hương trái cây chín", 0.0),
        
        # Positive examples for actual fruits (synthetic)
        ("trái cây", "Trái cây tươi ngon, ngọt mát", 0.95),
        ("fruit", "Fresh fruit dessert", 0.95),
        ("món trái cây", "Tráng miệng trái cây tươi", 0.95),
    ]
    
    for query, doc, score in synthetic_examples:
        training_examples.append(
            InputExample(texts=[query, doc], label=score)
        )
    
    print(f"Created {len(training_examples)} focused training examples")
    return training_examples

def train_focused_model():
    """Train a focused model for fruit vs tea distinction."""
    
    print("🎯 Focused Fruit vs Tea Training")
    print("="*40)
    
    # Load base model
    print("📦 Loading base model...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Create focused training data
    print("🔧 Creating focused training data...")
    training_examples = create_focused_training_data()
    
    # Create data loader with small batch size
    train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=8)
    
    # Define loss function
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Train with more epochs for stronger learning
    print("🚀 Training focused model...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=5,  # More epochs
        warmup_steps=50,
        output_path="../models/focused-fruit-tea-model/",
        show_progress_bar=True
    )
    
    print("✅ Focused model training complete!")
    return "../models/focused-fruit-tea-model/"

if __name__ == "__main__":
    model_path = train_focused_model()
    print(f"🎯 Focused model saved to: {model_path}")