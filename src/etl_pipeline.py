"""
Vietnamese Food Assistant - ETL Pipeline V2 for New Menu Data Format

This module implements the updated Extract, Transform, Load pipeline for processing
the new comprehensive CSV menu data format with rich metadata and detailed information.
"""

import pandas as pd
import json
import re
import ast
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from .menu_database import (
    Dish, SearchContent, MenuMetadata, Ingredient, NutritionalInfo,
    AvailabilityStatus, ConstraintType, ConstraintSeverity
)
from .config import get_config


class MenuDataETLV2:
    """
    Updated ETL Pipeline for processing comprehensive menu data format.
    
    Handles the new CSV format with detailed fields including:
    - Strategic roles and positioning
    - Detailed taste profiles and aromas
    - Customer mood and behavioral matching
    - Supplier information and verification
    - Nutritional information
    - Complex options and customizations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize ETL pipeline with configuration."""
        self.config = get_config()
        
    def extract_data(self, file_path: str) -> pd.DataFrame:
        """
        Extract data from the new CSV format.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with raw menu data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Read CSV with proper encoding and skip header rows
        # Use tab separator since the data appears to be tab-separated
        df = pd.read_csv(file_path, encoding='utf-8', skiprows=2, sep='\t')
        
        # Remove any completely empty rows
        df = df.dropna(how='all')
        
        print(f"Extracted {len(df)} records from {file_path}")
        print(f"Columns: {list(df.columns)}")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the extracted data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Rename columns to match our expected format
        column_mapping = {
            'code': 'id',
            'name': 'name_vi',
            'group': 'group',
            'category': 'category',
            'strategic_role': 'strategic_role',
            'short_description': 'description',
            'key_selling_points': 'selling_points',
            'food_tags': 'food_tags',
            'taste_profile': 'taste_profile',
            'aroma': 'aroma',
            'texture': 'texture',
            'temperature': 'temperature',
            'portion_size': 'portion_size',
            'post_meal_experience': 'post_meal_experience',
            'suitable_for': 'suitable_for',
            'spice_level': 'spice_level',
            'price': 'price_vnd',
            'prep_time_min': 'prep_time_min',
            'prep_time_max': 'prep_time_max',
            'ingredients': 'ingredients',
            'allergen_profile': 'allergens',
            'origin': 'origin',
            'supplier_details': 'supplier_details',
            'preparation_process': 'preparation_process',
            'health_benefits': 'health_benefits',
            'options': 'options',
            'nutrition_info': 'nutrition_info',
            'customer_moods_match': 'customer_moods',
            'behavioral_style_match': 'behavioral_style',
            'appearance_style_match': 'appearance_style',
            'communication_tone_match': 'communication_tone'
        }
        
        # Rename columns
        cleaned_df = cleaned_df.rename(columns=column_mapping)
        
        # Remove rows with missing essential fields
        essential_fields = ['id', 'name_vi', 'description', 'category']
        cleaned_df = cleaned_df.dropna(subset=essential_fields)
        
        # Clean text fields
        text_fields = ['name_vi', 'description', 'selling_points', 'aroma', 'texture', 
                      'post_meal_experience', 'suitable_for', 'origin', 'preparation_process', 
                      'health_benefits']
        
        for field in text_fields:
            if field in cleaned_df.columns:
                cleaned_df[field] = cleaned_df[field].astype(str).str.strip()
                cleaned_df[field] = cleaned_df[field].replace('nan', '')
                cleaned_df[field] = cleaned_df[field].replace('null', '')
        
        # Clean numeric fields
        if 'price_vnd' in cleaned_df.columns:
            cleaned_df['price_vnd'] = pd.to_numeric(cleaned_df['price_vnd'], errors='coerce')
        
        if 'prep_time_min' in cleaned_df.columns:
            cleaned_df['prep_time_min'] = pd.to_numeric(cleaned_df['prep_time_min'], errors='coerce')
        
        if 'prep_time_max' in cleaned_df.columns:
            cleaned_df['prep_time_max'] = pd.to_numeric(cleaned_df['prep_time_max'], errors='coerce')
        
        if 'spice_level' in cleaned_df.columns:
            cleaned_df['spice_level'] = pd.to_numeric(cleaned_df['spice_level'], errors='coerce')
            cleaned_df['spice_level'] = cleaned_df['spice_level'].fillna(0).astype(int)
        
        # Clean list fields (comma-separated strings)
        list_fields = ['food_tags', 'ingredients', 'allergens', 'customer_moods', 
                      'behavioral_style', 'appearance_style', 'communication_tone']
        
        for field in list_fields:
            if field in cleaned_df.columns:
                cleaned_df[field] = cleaned_df[field].astype(str).str.replace('nan', '')
                cleaned_df[field] = cleaned_df[field].str.replace('null', '')
                cleaned_df[field] = cleaned_df[field].apply(self._parse_comma_separated)
        
        # Parse JSON fields
        json_fields = ['supplier_details', 'options', 'nutrition_info']
        for field in json_fields:
            if field in cleaned_df.columns:
                cleaned_df[field] = cleaned_df[field].apply(self._parse_json_field)
        
        print(f"Cleaned data: {len(cleaned_df)} records remaining")
        return cleaned_df
    
    def _parse_comma_separated(self, value: str) -> List[str]:
        """Parse comma-separated string into list."""
        if not value or value == 'nan' or value == 'null' or pd.isna(value):
            return []
        
        # Handle special cases with quotes and parentheses
        items = []
        for item in str(value).split(','):
            item = item.strip()
            # Remove quotes and parentheses
            item = re.sub(r'^["\'\(]+|["\'\)]+$', '', item)
            # Clean up extra spaces
            item = re.sub(r'\s+', ' ', item).strip()
            if item:
                items.append(item)
        
        return items
    
    def _parse_json_field(self, value: str) -> Dict[str, Any]:
        """Parse JSON string field."""
        if not value or value == 'nan' or value == 'null' or pd.isna(value):
            return {}
        
        try:
            # Try to parse as JSON
            if isinstance(value, str):
                # Clean up the JSON string
                value = value.strip()
                if value.startswith('[') and value.endswith(']'):
                    return {"items": ast.literal_eval(value)}
                elif value.startswith('{') and value.endswith('}'):
                    return ast.literal_eval(value)
                else:
                    # Try to parse as key-value pairs
                    return self._parse_key_value_string(value)
            return {}
        except:
            # If JSON parsing fails, treat as string
            return {"raw": str(value)}
    
    def _parse_key_value_string(self, value: str) -> Dict[str, Any]:
        """Parse key-value string format."""
        result = {}
        
        # Handle format like: "calories_kcal": "~100-150", "sugar_g": "~20-25"
        pairs = re.findall(r'"([^"]+)":\s*"([^"]+)"', value)
        for key, val in pairs:
            result[key] = val
        
        return result
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on the cleaned data.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        engineered_df = df.copy()
        
        # Calculate advance order requirements
        engineered_df['requires_advance_order'] = engineered_df.apply(
            self._calculate_advance_order, axis=1
        )
        
        # Extract English name from description or generate from Vietnamese
        engineered_df['name_en'] = engineered_df.apply(self._extract_english_name, axis=1)
        
        # Determine availability status
        engineered_df['availability_status'] = engineered_df.apply(
            self._determine_availability, axis=1
        )
        
        # Calculate average preparation time
        engineered_df['prep_time_avg'] = engineered_df.apply(
            lambda row: (
                (row.get('prep_time_min', 0) + row.get('prep_time_max', 0)) / 2
                if pd.notna(row.get('prep_time_min')) and pd.notna(row.get('prep_time_max'))
                else row.get('prep_time_min', 15)  # Default 15 minutes
            ), axis=1
        )
        
        # Extract main ingredients from ingredients list
        engineered_df['main_ingredients'] = engineered_df['ingredients'].apply(
            self._extract_main_ingredients
        )
        
        # Determine dietary labels
        engineered_df['dietary_labels'] = engineered_df.apply(
            self._determine_dietary_labels, axis=1
        )
        
        # Calculate popularity score based on strategic role
        engineered_df['popularity_score'] = engineered_df['strategic_role'].apply(
            self._calculate_popularity_score
        )
        
        print(f"Feature engineering completed for {len(engineered_df)} records")
        return engineered_df
    
    def _calculate_advance_order(self, row: pd.Series) -> bool:
        """Calculate if dish requires advance ordering."""
        # Check if it's in pre-order category
        if str(row.get('group', '')).lower() == 'khẩu phần ăn đăng ký trước':
            return True
        
        # Check preparation time
        prep_time_max = row.get('prep_time_max', 0)
        if pd.notna(prep_time_max) and prep_time_max > 60:  # More than 1 hour
            return True
        
        # Check if it's a complex tea service
        if str(row.get('category', '')).lower() in ['trà pha ấm', 'set trà bánh']:
            return True
        
        return False
    
    def _extract_english_name(self, row: pd.Series) -> Optional[str]:
        """Extract or generate English name."""
        # Look for English names in description or selling points
        description = str(row.get('description', ''))
        selling_points = str(row.get('selling_points', ''))
        
        # Simple mapping for common Vietnamese dishes
        name_vi = str(row.get('name_vi', '')).lower()
        
        english_mappings = {
            'cà phê sữa đá': 'Iced Coffee with Milk',
            'cà phê đen đá': 'Iced Black Coffee',
            'trà lipton chanh': 'Lemon Tea',
            'nước chanh': 'Lemonade',
            'trà gừng ấm nóng': 'Hot Ginger Tea',
            'nước lọc chai': 'Bottled Water',
            'cơm chiên rau củ': 'Vegetable Fried Rice',
            'cơm trứng chiên': 'Egg Fried Rice',
            'mì gói trứng': 'Instant Noodles with Egg',
            'bún chả giò': 'Vermicelli with Spring Rolls',
            'miến xào rau củ': 'Stir-fried Glass Noodles with Vegetables'
        }
        
        return english_mappings.get(name_vi)
    
    def _determine_availability(self, row: pd.Series) -> str:
        """Determine availability status."""
        # Most items are available by default
        if row.get('requires_advance_order', False):
            return 'advance_order_only'
        
        return 'available'
    
    def _extract_main_ingredients(self, ingredients: List[str]) -> List[str]:
        """Extract main ingredients from ingredients list."""
        if not ingredients:
            return []
        
        # Take first 3 ingredients as main ingredients
        return ingredients[:3]
    
    def _determine_dietary_labels(self, row: pd.Series) -> List[str]:
        """Determine dietary labels based on ingredients and allergens."""
        labels = []
        
        ingredients = row.get('ingredients', [])
        allergens = row.get('allergens', [])
        
        # Check for vegetarian/vegan
        meat_ingredients = ['thịt bò', 'thịt heo', 'thịt gà', 'tôm', 'cá', 'cua', 'mực']
        has_meat = any(any(meat in str(ing).lower() for meat in meat_ingredients) 
                      for ing in ingredients)
        
        if not has_meat:
            labels.append('vegetarian')
            
            # Check for vegan (no dairy, eggs)
            dairy_eggs = ['sữa', 'trứng', 'phô mai', 'bơ']
            has_dairy_eggs = any(any(item in str(ing).lower() for item in dairy_eggs) 
                               for ing in ingredients)
            
            if not has_dairy_eggs:
                labels.append('vegan')
        
        # Check for gluten-free
        gluten_items = ['bánh mì', 'bánh phở', 'mì', 'bột mì']
        has_gluten = any(any(item in str(ing).lower() for item in gluten_items) 
                        for ing in ingredients)
        
        if not has_gluten and 'gluten' not in [str(a).lower() for a in allergens]:
            labels.append('gluten_free')
        
        return labels
    
    def _calculate_popularity_score(self, strategic_role: str) -> float:
        """Calculate popularity score based on strategic role."""
        role_scores = {
            'Sản phẩm Phổ thông (Mainstream/Volume Driver)': 0.8,
            'Sản phẩm Cốt lõi (Core)': 0.7,
            'Sản phẩm Đặc trưng (Signature)': 0.6,
            'Sản phẩm Dẫn dắt (Lead Magnet)': 0.9,
            'Sản phẩm Chữa lành (Healing Product)': 0.4,
            'Sản phẩm Ngách (Niche Product)': 0.3,
            'Sản phẩm Tiện ích (Utility Product)': 0.5,
            'Sản phẩm Bán thêm (Upsell Item)': 0.4
        }
        
        return role_scores.get(str(strategic_role), 0.5)
    
    def transform_to_dishes(self, df: pd.DataFrame) -> List[Dish]:
        """
        Transform DataFrame to list of Dish objects.
        
        Args:
            df: Engineered DataFrame
            
        Returns:
            List of Dish objects
        """
        dishes = []
        
        for _, row in df.iterrows():
            try:
                # Create search content
                search_content = SearchContent(
                    name_vi=row['name_vi'],
                    name_en=row.get('name_en'),
                    description_vi=row['description'],
                    description_en=None,  # Not available in new format
                    taste_profile=[row.get('taste_profile', '')],
                    texture=[row.get('texture', '')],
                    cuisine_tags=row.get('food_tags', []),
                    cooking_method=[row.get('preparation_process', '')],
                    meal_type=self._extract_meal_type(row),
                    main_ingredients=row.get('main_ingredients', []),
                    secondary_ingredients=row.get('ingredients', [])[3:]  # Rest of ingredients
                )
                
                # Create metadata
                metadata = MenuMetadata(
                    price_vnd=int(row['price_vnd']) if pd.notna(row.get('price_vnd')) else None,
                    price_range=self._categorize_price(row.get('price_vnd')),
                    category=row['category'],
                    subcategory=row.get('group'),
                    allergens=row.get('allergens', []),
                    dietary_labels=row.get('dietary_labels', []),
                    requires_advance_order=bool(row.get('requires_advance_order', False)),
                    advance_order_hours=24 if row.get('requires_advance_order', False) else None,
                    availability_status=AvailabilityStatus.AVAILABLE,
                    preparation_time_minutes=int(row['prep_time_avg']) if pd.notna(row.get('prep_time_avg')) else None,
                    spice_level=int(row['spice_level']) if pd.notna(row.get('spice_level')) else 0,
                    popularity_score=float(row.get('popularity_score', 0.5))
                )
                
                # Create ingredients (simplified for now)
                ingredients = [
                    Ingredient(
                        name=ing,
                        name_en=None,
                        allergen_info=[],
                        dietary_restrictions=[],
                        is_main_ingredient=ing in row.get('main_ingredients', [])
                    )
                    for ing in row.get('ingredients', [])
                ]
                
                # Create dish
                dish = Dish(
                    id=str(row['id']),
                    search_content=search_content,
                    metadata=metadata,
                    ingredients=ingredients
                )
                
                dishes.append(dish)
                
            except Exception as e:
                print(f"Error processing row {row.get('id', 'unknown')}: {e}")
                continue
        
        print(f"Transformed {len(dishes)} dishes successfully")
        return dishes
    
    def _extract_meal_type(self, row: pd.Series) -> List[str]:
        """Extract meal type from category and group."""
        meal_types = []
        
        category = str(row.get('category', '')).lower()
        group = str(row.get('group', '')).lower()
        
        if 'bữa sáng' in category or 'bữa sáng' in group:
            meal_types.append('sang')
        
        if 'bữa trưa' in category or 'bữa trưa' in group:
            meal_types.append('trua')
        
        if 'coffee' in category or 'giải khát' in group:
            meal_types.extend(['sang', 'chieu'])
        
        if 'trà' in category:
            meal_types.extend(['chieu', 'toi'])
        
        if not meal_types:
            # Default based on category
            if category in ['quick meal']:
                meal_types.extend(['trua', 'toi'])
            else:
                meal_types.append('chieu')
        
        return meal_types
    
    def _categorize_price(self, price: float) -> str:
        """Categorize price into ranges."""
        if pd.isna(price) or price == 0:
            return "free"
        
        if price < 40000:
            return "budget"
        elif price < 100000:
            return "mid"
        else:
            return "premium"
    
    def process_menu_data(self, input_file: str, output_file: Optional[str] = None) -> List[Dish]:
        """
        Complete ETL pipeline: Extract, Transform, Load menu data.
        
        Args:
            input_file: Path to input CSV file
            output_file: Optional path to save processed data as JSON
            
        Returns:
            List of processed Dish objects
        """
        print(f"Starting ETL V2 pipeline for {input_file}")
        
        # Extract
        raw_df = self.extract_data(input_file)
        
        # Transform
        cleaned_df = self.clean_data(raw_df)
        engineered_df = self.engineer_features(cleaned_df)
        dishes = self.transform_to_dishes(engineered_df)
        
        # Save processed data if requested
        if output_file:
            self.save_processed_data(dishes, output_file)
        
        print(f"ETL V2 pipeline completed: {len(dishes)} dishes processed")
        return dishes
    
    def save_processed_data(self, dishes: List[Dish], output_file: str):
        """Save processed dishes to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dishes to serializable format
        dishes_data = []
        for dish in dishes:
            dish_data = {
                "id": dish.id,
                "name_vi": dish.search_content.name_vi,
                "name_en": dish.search_content.name_en,
                "description": dish.search_content.description_vi,
                "category": dish.metadata.category,
                "subcategory": dish.metadata.subcategory,
                "price": dish.metadata.price_vnd,
                "ingredients": [ing.name for ing in dish.ingredients],
                "allergens": dish.metadata.allergens,
                "spice_level": dish.metadata.spice_level,
                "requires_advance_order": dish.metadata.requires_advance_order,
                "availability_status": dish.metadata.availability_status.value,
                "preparation_time": dish.metadata.preparation_time_minutes,
                "popularity_score": dish.metadata.popularity_score,
                "taste_profile": dish.search_content.taste_profile,
                "cuisine_tags": dish.search_content.cuisine_tags,
                "meal_type": dish.search_content.meal_type,
                "dietary_labels": dish.metadata.dietary_labels
            }
            dishes_data.append(dish_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dishes_data, f, ensure_ascii=False, indent=2)
        
        print(f"Processed data saved to {output_path}")


def main():
    """Main function for running ETL V2 pipeline."""
    etl = MenuDataETLV2()
    
    # Process new menu data
    input_file = "data/sample_menu.csv"
    output_file = "data/processed_menu.json"
    
    dishes = etl.process_menu_data(input_file, output_file)
    
    print(f"\nETL V2 Pipeline Summary:")
    print(f"- Processed {len(dishes)} dishes")
    
    # Group by category
    categories = {}
    for dish in dishes:
        cat = dish.metadata.category
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
    
    print(f"- Categories: {categories}")
    print(f"- Advance order dishes: {sum(1 for dish in dishes if dish.metadata.requires_advance_order)}")
    
    # Calculate average price (excluding free items)
    prices = [dish.metadata.price_vnd for dish in dishes if dish.metadata.price_vnd and dish.metadata.price_vnd > 0]
    if prices:
        print(f"- Average price: {sum(prices) / len(prices):.0f} VND")
    
    # Show some examples
    print(f"\nSample dishes:")
    for i, dish in enumerate(dishes[:5]):
        print(f"  {i+1}. {dish.search_content.name_vi} - {dish.metadata.price_vnd or 0} VND")


if __name__ == "__main__":
    main()