"""
Vietnamese Food Assistant - Menu Database Schema and Models

This module defines the rich schema for RAG (Retrieval-Augmented Generation)
including search content structure and metadata for Vietnamese food items.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class AvailabilityStatus(Enum):
    """Availability status for menu items."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LIMITED = "limited"
    ADVANCE_ORDER_ONLY = "advance_order_only"


class ConstraintType(Enum):
    """Types of dietary constraints."""
    ALLERGY = "allergy"
    DIETARY = "dietary"
    PREFERENCE = "preference"
    DISLIKE = "dislike"


class ConstraintSeverity(Enum):
    """Severity levels for constraints."""
    STRICT = "strict"
    MODERATE = "moderate"
    MILD = "mild"


@dataclass
class Ingredient:
    """Represents an ingredient with allergen and dietary information."""
    name: str
    name_en: Optional[str] = None
    allergen_info: List[str] = field(default_factory=list)
    dietary_restrictions: List[str] = field(default_factory=list)
    is_main_ingredient: bool = False


@dataclass
class NutritionalInfo:
    """Nutritional information for dishes."""
    calories_per_serving: Optional[int] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    fiber_g: Optional[float] = None
    sodium_mg: Optional[float] = None


@dataclass
class SearchContent:
    """
    Rich search content structure for RAG.
    Combines name, description, taste profile, and tags for comprehensive search.
    """
    # Core identifiers
    name_vi: str
    description_vi: str
    name_en: Optional[str] = None
    description_en: Optional[str] = None
    
    # Taste profile for better matching
    taste_profile: List[str] = field(default_factory=list)  # ["cay", "ngot", "chua", "man", "dang"]
    texture: List[str] = field(default_factory=list)  # ["mem", "gion", "dai", "nuot"]
    
    # Searchable tags
    cuisine_tags: List[str] = field(default_factory=list)  # ["bac", "trung", "nam", "hue"]
    cooking_method: List[str] = field(default_factory=list)  # ["nuong", "xao", "lam", "hap"]
    meal_type: List[str] = field(default_factory=list)  # ["sang", "trua", "chieu", "toi"]
    occasion_tags: List[str] = field(default_factory=list)  # ["gia_dinh", "ban_be", "dac_biet"]
    
    # Ingredient-based search terms
    main_ingredients: List[str] = field(default_factory=list)
    secondary_ingredients: List[str] = field(default_factory=list)
    
    def to_search_text(self) -> str:
        """Convert search content to a single searchable text string."""
        def safe_join(items):
            """Safely join items, converting to string and filtering out None/empty values."""
            if not items:
                return ""
            return " ".join(str(item) for item in items if item is not None and str(item).strip())
        
        def safe_str(value):
            """Safely convert value to string, handling None and float values."""
            if value is None:
                return ""
            return str(value).strip()
        
        components = [
            safe_str(self.name_vi),
            safe_str(self.name_en),
            safe_str(self.description_vi),
            safe_str(self.description_en),
            safe_join(self.taste_profile),
            safe_join(self.texture),
            safe_join(self.cuisine_tags),
            safe_join(self.cooking_method),
            safe_join(self.meal_type),
            safe_join(self.occasion_tags),
            safe_join(self.main_ingredients),
            safe_join(self.secondary_ingredients)
        ]
        return " ".join(filter(None, components))


@dataclass
class MenuMetadata:
    """
    Comprehensive metadata for menu items.
    Includes pricing, categorization, allergens, and ordering requirements.
    """
    # Pricing information
    price_vnd: Optional[int] = None
    price_range: Optional[str] = None  # "budget", "mid", "premium"
    
    # Categorization
    category: str = "main"  # "appetizer", "main", "dessert", "beverage", "side"
    subcategory: Optional[str] = None  # "soup", "rice", "noodle", "meat", "vegetarian"
    
    # Allergen information
    allergens: List[str] = field(default_factory=list)  # ["gluten", "dairy", "nuts", "seafood", "eggs"]
    dietary_labels: List[str] = field(default_factory=list)  # ["vegetarian", "vegan", "halal", "gluten_free"]
    
    # Ordering and availability
    requires_advance_order: bool = False
    advance_order_hours: Optional[int] = None
    availability_status: AvailabilityStatus = AvailabilityStatus.AVAILABLE
    seasonal_availability: Optional[List[str]] = None  # ["spring", "summer", "fall", "winter"]
    
    # Preparation information
    preparation_time_minutes: Optional[int] = None
    serving_size: Optional[str] = None  # "1 person", "2-3 people", "family"
    spice_level: Optional[int] = None  # 1-5 scale
    
    # Nutritional and dietary
    nutritional_info: Optional[NutritionalInfo] = None
    is_signature_dish: bool = False
    chef_recommendation: bool = False
    
    # Business metadata
    popularity_score: Optional[float] = None  # 0-1 based on orders
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for ChromaDB storage."""
        return {
            "price_vnd": self.price_vnd,
            "price_range": self.price_range,
            "category": self.category,
            "subcategory": self.subcategory,
            "allergens": self.allergens,
            "dietary_labels": self.dietary_labels,
            "requires_advance_order": self.requires_advance_order,
            "advance_order_hours": self.advance_order_hours,
            "availability_status": self.availability_status.value,
            "seasonal_availability": self.seasonal_availability,
            "preparation_time_minutes": self.preparation_time_minutes,
            "serving_size": self.serving_size,
            "spice_level": self.spice_level,
            "is_signature_dish": self.is_signature_dish,
            "chef_recommendation": self.chef_recommendation,
            "popularity_score": self.popularity_score,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class Dish:
    """
    Complete dish representation combining search content and metadata.
    This is the main data model for menu items in the Vietnamese Food Assistant.
    """
    # Unique identifier
    id: str
    
    # Search content for RAG
    search_content: SearchContent
    
    # Comprehensive metadata
    metadata: MenuMetadata
    
    # Detailed ingredient list
    ingredients: List[Ingredient] = field(default_factory=list)
    
    # Additional context
    origin_story: Optional[str] = None
    preparation_notes: Optional[str] = None
    serving_suggestions: Optional[str] = None
    
    def __post_init__(self):
        """Validate dish data after initialization."""
        if not self.id:
            raise ValueError("Dish ID cannot be empty")
        if not self.search_content.name_vi:
            raise ValueError("Vietnamese name is required")
        if not self.search_content.description_vi:
            raise ValueError("Vietnamese description is required")
    
    def get_search_text(self) -> str:
        """Get the complete search text for this dish."""
        return self.search_content.to_search_text()
    
    def get_metadata_dict(self) -> Dict[str, Any]:
        """Get metadata dictionary for ChromaDB storage."""
        metadata_dict = self.metadata.to_metadata_dict()
        metadata_dict["dish_id"] = self.id
        metadata_dict["name_vi"] = self.search_content.name_vi
        metadata_dict["name_en"] = self.search_content.name_en
        
        # Convert lists to strings and filter out None values for ChromaDB compatibility
        filtered_dict = {}
        for key, value in metadata_dict.items():
            if value is not None:
                if isinstance(value, list):
                    filtered_dict[key] = ",".join(str(v) for v in value) if value else ""
                else:
                    filtered_dict[key] = value
        
        return filtered_dict
    
    def matches_constraints(self, constraints: List[Dict[str, Any]]) -> bool:
        """Check if this dish matches the given dietary constraints."""
        for constraint in constraints:
            constraint_type = constraint.get("type", "").lower()
            constraint_value = constraint.get("value", "").lower()
            severity = constraint.get("severity", "strict").lower()
            
            if constraint_type == "allergy":
                # Strict checking for allergies
                if any(allergen.lower() in constraint_value or 
                      constraint_value in allergen.lower() 
                      for allergen in self.metadata.allergens):
                    return False
                
                # Check ingredients for allergens
                if any(constraint_value in ingredient.name.lower() or
                      any(allergen.lower() in constraint_value 
                          for allergen in ingredient.allergen_info)
                      for ingredient in self.ingredients):
                    return False
            
            elif constraint_type == "dietary":
                # Check dietary restrictions
                if constraint_value == "vegetarian" and "vegetarian" not in self.metadata.dietary_labels:
                    return False
                if constraint_value == "vegan" and "vegan" not in self.metadata.dietary_labels:
                    return False
                if constraint_value == "halal" and "halal" not in self.metadata.dietary_labels:
                    return False
                if constraint_value == "gluten_free" and "gluten_free" not in self.metadata.dietary_labels:
                    return False
            
            elif constraint_type == "dislike":
                # Check if disliked ingredients are present
                if severity == "strict":
                    if any(constraint_value in ingredient.name.lower() 
                          for ingredient in self.ingredients):
                        return False
        
        return True
    
    def calculate_preference_score(self, preferences: Dict[str, float]) -> float:
        """Calculate preference score based on user preferences."""
        score = 0.0
        total_weight = 0.0
        
        # Check taste preferences
        for taste in self.search_content.taste_profile:
            if taste in preferences:
                score += preferences[taste]
                total_weight += 1.0
        
        # Check cuisine preferences
        for cuisine in self.search_content.cuisine_tags:
            if cuisine in preferences:
                score += preferences[cuisine]
                total_weight += 1.0
        
        # Check cooking method preferences
        for method in self.search_content.cooking_method:
            if method in preferences:
                score += preferences[method]
                total_weight += 1.0
        
        # Normalize score
        if total_weight > 0:
            return score / total_weight
        
        return 0.5  # Neutral score if no preferences match


@dataclass
class SearchResult:
    """Represents a search result from the menu database."""
    dish: Dish
    relevance_score: float
    search_method: str  # "vector", "bm25", "hybrid"
    matched_terms: List[str] = field(default_factory=list)
    
    def to_retrieved_document(self) -> Dict[str, Any]:
        """Convert to RetrievedDocument format for AgentState."""
        return {
            "id": self.dish.id,
            "name_vi": self.dish.search_content.name_vi,
            "name_en": self.dish.search_content.name_en,
            "description": self.dish.search_content.description_vi,
            "category": self.dish.metadata.category,
            "ingredients": [ing.name for ing in self.dish.ingredients],
            "allergens": self.dish.metadata.allergens,
            "requires_advance_order": self.dish.metadata.requires_advance_order,
            "availability_status": self.dish.metadata.availability_status.value,
            "score": self.relevance_score
        }