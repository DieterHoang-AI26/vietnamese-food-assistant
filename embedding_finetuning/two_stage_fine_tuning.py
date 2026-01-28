#!/usr/bin/env python3
"""
Two-Stage Fine-tuning for Vietnamese Food Domain

Stage 1: Create Vietnamese Food Foundation Model (1000+ dishes)
Stage 2: Adapt to Restaurant-Specific Menu (35 dishes)
"""

import json
import pandas as pd
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import torch


class TwoStageVietnameseFoodTrainer:
    """Two-stage trainer for Vietnamese food domain."""
    
    def __init__(self, base_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.base_model_name = base_model_name
        self.foundation_model = None
        self.final_model = None
        
        print("🚀 Two-Stage Vietnamese Food Fine-tuning")
        print(f"📦 Base model: {base_model_name}")
    
    def stage1_create_foundation_model(self, 
                                     large_dataset_path: str = "../data/dishes_Vietnamese.csv",
                                     output_path: str = "../models/vietnamese-food-foundation/"):
        """
        Stage 1: Create Vietnamese Food Foundation Model from large dataset.
        """
        print("\n" + "="*60)
        print("🏗️  STAGE 1: Creating Vietnamese Food Foundation Model")
        print("="*60)
        
        # Load large Vietnamese food dataset
        print(f"📊 Loading large dataset: {large_dataset_path}")
        large_data = self._load_csv_data(large_dataset_path)
        print(f"✅ Loaded {len(large_data)} Vietnamese dishes")
        
        # Create training pairs for foundation model
        foundation_pairs = self._create_foundation_training_pairs(large_data)
        print(f"🔧 Created {len(foundation_pairs)} foundation training pairs")
        
        # Initialize base model
        print(f"🤖 Loading base model: {self.base_model_name}")
        self.foundation_model = SentenceTransformer(self.base_model_name)
        
        # Train foundation model
        self._train_model(
            model=self.foundation_model,
            training_pairs=foundation_pairs,
            output_path=output_path,
            model_name="Vietnamese Food Foundation",
            epochs=2
        )
        
        print(f"✅ Stage 1 Complete! Foundation model saved to: {output_path}")
        return output_path
    
    def stage2_adapt_to_restaurant(self,
                                  foundation_model_path: str,
                                  restaurant_data_path: str = "../data/processed_menu_v2.json",
                                  output_path: str = "../models/vietnamese-food-restaurant/"):
        """
        Stage 2: Adapt foundation model to specific restaurant menu.
        """
        print("\n" + "="*60)
        print("🎯 STAGE 2: Adapting to Restaurant Menu")
        print("="*60)
        
        # Load foundation model
        print(f"📦 Loading foundation model: {foundation_model_path}")
        self.final_model = SentenceTransformer(foundation_model_path)
        
        # Load restaurant menu
        print(f"📊 Loading restaurant menu: {restaurant_data_path}")
        restaurant_data = self._load_json_data(restaurant_data_path)
        print(f"✅ Loaded {len(restaurant_data)} restaurant dishes")
        
        # Create restaurant-specific training pairs
        restaurant_pairs = self._create_restaurant_training_pairs(restaurant_data)
        print(f"🔧 Created {len(restaurant_pairs)} restaurant training pairs")
        
        # Fine-tune on restaurant data
        self._train_model(
            model=self.final_model,
            training_pairs=restaurant_pairs,
            output_path=output_path,
            model_name="Restaurant-Specific Model",
            epochs=3
        )
        
        print(f"✅ Stage 2 Complete! Final model saved to: {output_path}")
        return output_path
    
    def _load_csv_data(self, csv_path: str) -> List[Dict]:
        """Load CSV data and convert to standard format."""
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"📋 CSV columns: {list(df.columns)}")
        
        menu_data = []
        for _, row in df.iterrows():
            dish = {
                'id': row.get('DishID', f"DISH_{len(menu_data)}"),
                'name_vi': row.get('DishName', ''),
                'name_en': '',
                'description': row.get('Description', ''),
                'category': row.get('Category', ''),
                'subcategory': '',
                'ingredients': self._parse_ingredients(row.get('Ingredients', '')),
                'price': self._parse_price(row.get('Price', ''))
            }
            menu_data.append(dish)
        
        return menu_data
    
    def _load_json_data(self, json_path: str) -> List[Dict]:
        """Load JSON data."""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _parse_ingredients(self, ingredients_str: str) -> List[str]:
        """Parse ingredients string to list."""
        if not ingredients_str or pd.isna(ingredients_str):
            return []
        
        import re
        ingredients = re.split(r'[,;|]', str(ingredients_str))
        return [ing.strip() for ing in ingredients if ing.strip()]
    
    def _parse_price(self, price_str: str) -> int:
        """Parse price string to integer."""
        if not price_str or pd.isna(price_str):
            return 0
        
        import re
        # Extract numbers from price string like "65.000₫"
        numbers = re.findall(r'[\d.]+', str(price_str))
        if numbers:
            return int(numbers[0].replace('.', ''))
        return 0
    
    def _create_foundation_training_pairs(self, large_data: List[Dict]) -> List[InputExample]:
        """Create training pairs for foundation model from large dataset."""
        training_examples = []
        
        print("🔧 Creating foundation training pairs...")
        
        for dish in large_data:
            dish_name = dish.get('name_vi', '')
            dish_desc = dish.get('description', '')
            dish_category = dish.get('category', '')
            ingredients = dish.get('ingredients', [])
            
            if not dish_name or not dish_desc:
                continue
            
            # 1. Name-Description pairs (high similarity)
            training_examples.append(
                InputExample(texts=[dish_name, dish_desc], label=0.9)
            )
            
            # 2. Category-based queries
            if dish_category:
                category_queries = [
                    f"món {dish_category.lower()}",
                    f"có món {dish_category.lower()} nào không",
                    dish_category.lower()
                ]
                
                for query in category_queries:
                    full_text = f"{dish_name} {dish_desc}"
                    training_examples.append(
                        InputExample(texts=[query, full_text], label=0.7)
                    )
            
            # 3. Ingredient-based queries
            for ingredient in ingredients[:3]:  # Top 3 ingredients
                if ingredient and len(ingredient.strip()) > 2:
                    queries = [
                        f"có món nào có {ingredient}",
                        f"món {ingredient}",
                        f"tôi muốn ăn {ingredient}"
                    ]
                    
                    for query in queries:
                        full_text = f"{dish_name} {dish_desc}"
                        training_examples.append(
                            InputExample(texts=[query, full_text], label=0.8)
                        )
            
            # 4. CRITICAL: Fruit vs Tea distinction training
            self._add_fruit_tea_distinction_pairs(dish, training_examples)
        
        # 5. Add negative pairs
        negative_pairs = self._create_negative_pairs(large_data, num_negative=500)
        training_examples.extend(negative_pairs)
        
        return training_examples
    
    def _add_fruit_tea_distinction_pairs(self, dish: Dict, training_examples: List[InputExample]):
        """Add specific training pairs to distinguish fruit vs tea with fruit flavor."""
        dish_name = dish.get('name_vi', '')
        dish_desc = dish.get('description', '')
        dish_category = dish.get('category', '')
        
        full_text = f"{dish_name} {dish_desc}"
        
        # Real fruit dishes - HIGH similarity with fruit queries
        if any(fruit_keyword in dish_category.lower() for fruit_keyword in ['tráng miệng', 'trái cây', 'fruit']):
            if any(fruit_word in dish_name.lower() for fruit_word in ['bưởi', 'xoài', 'chuối', 'dưa', 'cam']):
                fruit_queries = [
                    "trái cây",
                    "tráo cây",  # Spelling error
                    "fruit",
                    "có trái cây không",
                    "món trái cây",
                    "trai cay"   # No diacritics
                ]
                
                for query in fruit_queries:
                    training_examples.append(
                        InputExample(texts=[query, full_text], label=0.95)  # Very high
                    )
        
        # Tea/drinks with fruit flavor - VERY LOW similarity with fruit queries
        elif any(drink_keyword in dish_category.lower() for drink_keyword in ['đồ uống', 'trà', 'tea', 'nước']):
            # Special handling for "Đại Hồng Bào" and similar teas
            if any(fruit_flavor in dish_desc.lower() for fruit_flavor in ['hương trái cây', 'vị trái cây', 'hương thơm trái cây']) or 'Đại Hồng Bào' in dish_name:
                fruit_queries = [
                    "trái cây",
                    "tráo cây",
                    "fruit", 
                    "có trái cây không",
                    "món trái cây",
                    "trai cay"
                ]
                
                for query in fruit_queries:
                    # VERY LOW similarity - this is tea with fruit flavor, not real fruit
                    training_examples.append(
                        InputExample(texts=[query, full_text], label=0.05)  # Much lower
                    )
                
                # Add explicit negative examples
                negative_fruit_examples = [
                    ("tôi muốn ăn trái cây", full_text, 0.0),
                    ("có trái cây tươi không", full_text, 0.0),
                    ("món tráng miệng trái cây", full_text, 0.0)
                ]
                
                for query, doc, score in negative_fruit_examples:
                    training_examples.append(
                        InputExample(texts=[query, doc], label=score)
                    )
                
                # HIGH similarity with drink queries instead
                drink_queries = [
                    "đồ uống",
                    "nước uống", 
                    "giải khát",
                    "trà",
                    "tea",
                    "đồ uống có hương trái cây",
                    "trà có vị trái cây"
                ]
                
                for query in drink_queries:
                    training_examples.append(
                        InputExample(texts=[query, full_text], label=0.9)
                    )
    
    def _create_restaurant_training_pairs(self, restaurant_data: List[Dict]) -> List[InputExample]:
        """Create training pairs specific to restaurant menu."""
        training_examples = []
        
        print("🔧 Creating restaurant-specific training pairs...")
        
        for dish in restaurant_data:
            dish_name = dish.get('name_vi', '')
            dish_desc = dish.get('description', '')
            ingredients = dish.get('ingredients', [])
            
            if not dish_name:
                continue
            
            # Restaurant-specific name variations and spelling corrections
            training_examples.extend(self._create_spelling_corrections(dish))
            
            # Natural language queries specific to this restaurant
            training_examples.extend(self._create_natural_queries(dish))
        
        return training_examples
    
    def _create_spelling_corrections(self, dish: Dict) -> List[InputExample]:
        """Create spelling correction pairs for restaurant dishes."""
        dish_name = dish.get('name_vi', '')
        dish_desc = dish.get('description', '')
        full_text = f"{dish_name} {dish_desc}"
        
        corrections = []
        
        # Common Vietnamese spelling errors
        if 'cháo' in dish_name.lower():
            corrections.append(InputExample(texts=["tráo", full_text], label=0.95))
            corrections.append(InputExample(texts=["chao", full_text], label=0.9))
        
        if 'cà phê' in dish_name.lower():
            corrections.append(InputExample(texts=["ca phe", full_text], label=0.95))
            corrections.append(InputExample(texts=["coffee", full_text], label=0.9))
        
        if 'trứng' in dish_name.lower():
            corrections.append(InputExample(texts=["trung", full_text], label=0.95))
        
        return corrections
    
    def _create_natural_queries(self, dish: Dict) -> List[InputExample]:
        """Create natural language queries for restaurant dishes."""
        dish_name = dish.get('name_vi', '')
        dish_desc = dish.get('description', '')
        ingredients = dish.get('ingredients', [])
        full_text = f"{dish_name} {dish_desc}"
        
        queries = []
        
        # Natural questions
        for ingredient in ingredients[:2]:
            if ingredient and len(ingredient.strip()) > 2:
                natural_queries = [
                    f"có món nào có {ingredient} không",
                    f"tôi muốn ăn {ingredient}",
                    f"món {ingredient} nào ngon"
                ]
                
                for query in natural_queries:
                    queries.append(InputExample(texts=[query, full_text], label=0.85))
        
        return queries
    
    def _create_negative_pairs(self, menu_data: List[Dict], num_negative: int = 200) -> List[InputExample]:
        """Create negative pairs for better discrimination."""
        negative_examples = []
        
        for _ in range(num_negative):
            dish1 = random.choice(menu_data)
            dish2 = random.choice(menu_data)
            
            if (dish1.get('id') != dish2.get('id') and 
                dish1.get('category') != dish2.get('category')):
                
                query = dish1.get('name_vi', '')
                doc = f"{dish2.get('name_vi', '')} {dish2.get('description', '')}"
                
                if query and doc:
                    negative_examples.append(
                        InputExample(texts=[query, doc], label=0.1)
                    )
        
        return negative_examples
    
    def _train_model(self, model, training_pairs, output_path, model_name, epochs=2):
        """Train the model with given pairs."""
        print(f"🚀 Training {model_name}...")
        print(f"📊 Training pairs: {len(training_pairs)}")
        print(f"🔄 Epochs: {epochs}")
        
        # Create data loader with smaller batch size for MPS
        batch_size = 8 if torch.backends.mps.is_available() else 16
        train_dataloader = DataLoader(training_pairs, shuffle=True, batch_size=batch_size)
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(model)
        
        # Train the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            output_path=output_path,
            show_progress_bar=True
        )
        
        print(f"✅ {model_name} training complete!")
    
    def run_full_pipeline(self):
        """Run the complete two-stage fine-tuning pipeline."""
        print("🚀 Starting Two-Stage Fine-tuning Pipeline")
        
        # Stage 1: Create foundation model
        foundation_path = self.stage1_create_foundation_model()
        
        # Stage 2: Adapt to restaurant
        final_path = self.stage2_adapt_to_restaurant(foundation_path)
        
        print("\n" + "="*60)
        print("🎉 TWO-STAGE FINE-TUNING COMPLETE!")
        print("="*60)
        print(f"📦 Foundation Model: {foundation_path}")
        print(f"🎯 Final Model: {final_path}")
        print("\n✅ Your Vietnamese Food Assistant is now ready with:")
        print("   • Deep understanding of Vietnamese food domain")
        print("   • Accurate fruit vs tea distinction")
        print("   • Restaurant-specific optimizations")
        
        return final_path


def main():
    """Run two-stage fine-tuning."""
    trainer = TwoStageVietnameseFoodTrainer()
    final_model_path = trainer.run_full_pipeline()
    
    print(f"\n🎯 Final model ready at: {final_model_path}")
    print("🔄 To use this model, update your config to point to this path.")


if __name__ == "__main__":
    main()