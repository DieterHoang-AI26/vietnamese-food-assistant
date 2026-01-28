#!/usr/bin/env python3
"""
Run only Stage 2: Adapt foundation model to restaurant menu
"""

from two_stage_fine_tuning import TwoStageVietnameseFoodTrainer

def main():
    trainer = TwoStageVietnameseFoodTrainer()
    
    # Run only Stage 2 (foundation model already exists)
    final_path = trainer.stage2_adapt_to_restaurant(
        foundation_model_path="../models/vietnamese-food-foundation/",
        restaurant_data_path="../data/processed_menu_v2.json",
        output_path="../models/vietnamese-food-restaurant/"
    )
    
    print(f"\n🎯 Final model ready at: {final_path}")
    print("🔄 To use this model, update your RAG system config.")

if __name__ == "__main__":
    main()