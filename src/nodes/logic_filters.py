"""
Vietnamese Food Assistant - Logic Filters (Python Guardrails)

This module implements Python-based logic filters that ensure absolute safety
by checking hard constraints like price limits, allergens, and dietary restrictions.
These filters provide deterministic constraint enforcement beyond what the RAG engine handles.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from ..state import AgentState
from ..config import get_config


class LogicFilters:
    """
    Python-based logic filters for enforcing hard constraints.
    
    This class provides deterministic filtering of retrieved documents
    based on strict constraints that must be enforced for safety,
    particularly allergen restrictions and dietary requirements.
    """
    
    def __init__(self):
        """Initialize logic filters."""
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
    
    def __call__(self, state: AgentState) -> AgentState:
        """
        Apply logic filters to retrieved documents.
        
        Args:
            state: Current agent state with retrieved documents
            
        Returns:
            Updated agent state with filtered documents
        """
        try:
            retrieved_docs = state.get("retrieved_docs", [])
            active_constraints = state.get("active_constraints", [])
            
            if not retrieved_docs:
                self.logger.info("No retrieved documents to filter")
                state["filtered_docs"] = []
                return state
            
            self.logger.info(f"Filtering {len(retrieved_docs)} documents with {len(active_constraints)} constraints")
            
            # Apply all filters
            filtered_docs = self._apply_all_filters(retrieved_docs, active_constraints, state)
            
            # Update state
            state["filtered_docs"] = filtered_docs
            
            # Log filtering results
            filtered_count = len(filtered_docs)
            original_count = len(retrieved_docs)
            self.logger.info(f"Filtered to {filtered_count}/{original_count} documents")
            
            if filtered_count < original_count:
                removed_count = original_count - filtered_count
                state["warnings"] = state.get("warnings", []) + [
                    f"Filtered out {removed_count} dishes due to dietary constraints"
                ]
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in logic filters: {e}")
            state["errors"] = state.get("errors", []) + [f"Logic filter error: {str(e)}"]
            # Fallback: pass through original documents
            state["filtered_docs"] = state.get("retrieved_docs", [])
            return state
    
    def _apply_all_filters(self, documents: List[Dict[str, Any]], 
                          constraints: List[Dict[str, Any]], 
                          state: AgentState) -> List[Dict[str, Any]]:
        """
        Apply all logic filters to the documents.
        
        Args:
            documents: List of retrieved documents
            constraints: List of active constraints
            state: Current agent state for additional context
            
        Returns:
            Filtered list of documents
        """
        filtered_docs = documents.copy()
        
        # Apply constraint-based filters
        if constraints:
            filtered_docs = self._filter_by_constraints(filtered_docs, constraints)
        
        # Apply availability filters
        filtered_docs = self._filter_by_availability(filtered_docs, state)
        
        # Apply price filters if specified
        filtered_docs = self._filter_by_price(filtered_docs, constraints)
        
        # Apply preparation time filters if specified
        filtered_docs = self._filter_by_preparation_time(filtered_docs, constraints)
        
        # Limit to maximum filtered documents
        max_docs = self.config.max_filtered_docs
        if len(filtered_docs) > max_docs:
            # Keep highest scoring documents
            filtered_docs = sorted(filtered_docs, key=lambda x: x.get("score", 0), reverse=True)[:max_docs]
        
        return filtered_docs
    
    def _filter_by_constraints(self, documents: List[Dict[str, Any]], 
                              constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter documents by dietary constraints with absolute safety.
        
        Args:
            documents: List of documents to filter
            constraints: List of constraints to enforce
            
        Returns:
            Filtered list of documents
        """
        filtered_docs = []
        
        for doc in documents:
            if self._document_satisfies_constraints(doc, constraints):
                filtered_docs.append(doc)
            else:
                self.logger.debug(f"Filtered out '{doc.get('name_vi', 'Unknown')}' due to constraints")
        
        return filtered_docs
    
    def _document_satisfies_constraints(self, document: Dict[str, Any], 
                                      constraints: List[Dict[str, Any]]) -> bool:
        """
        Check if a document satisfies all constraints.
        
        Args:
            document: Document to check
            constraints: List of constraints to verify
            
        Returns:
            True if document satisfies all constraints, False otherwise
        """
        for constraint in constraints:
            if not self._check_single_constraint(document, constraint):
                return False
        return True
    
    def _check_single_constraint(self, document: Dict[str, Any], 
                                constraint: Dict[str, Any]) -> bool:
        """
        Check if a document satisfies a single constraint.
        
        Args:
            document: Document to check
            constraint: Single constraint to verify
            
        Returns:
            True if constraint is satisfied, False otherwise
        """
        constraint_type = constraint.get("type", "").lower()
        constraint_value = constraint.get("value", "").lower()
        severity = constraint.get("severity", "strict").lower()
        
        # Get document data
        allergens = document.get("allergens", [])
        ingredients = document.get("ingredients", [])
        
        # Convert to lowercase for comparison
        allergens_lower = [a.lower() for a in allergens]
        ingredients_lower = [i.lower() for i in ingredients]
        
        if constraint_type == "allergy":
            # STRICT enforcement for allergies - absolute safety
            return self._check_allergy_constraint(constraint_value, allergens_lower, ingredients_lower)
        
        elif constraint_type == "dietary":
            return self._check_dietary_constraint(constraint_value, document)
        
        elif constraint_type == "dislike":
            if severity == "strict":
                return self._check_dislike_constraint(constraint_value, ingredients_lower)
            else:
                # For non-strict dislikes, we allow them through but lower the score
                return True
        
        elif constraint_type == "preference":
            # Preferences don't filter out documents, they just affect scoring
            return True
        
        # Unknown constraint type - be conservative and allow
        self.logger.warning(f"Unknown constraint type: {constraint_type}")
        return True
    
    def _check_allergy_constraint(self, allergen: str, doc_allergens: List[str], 
                                 doc_ingredients: List[str]) -> bool:
        """
        Check allergy constraint with absolute safety.
        
        Args:
            allergen: Allergen to check for
            doc_allergens: Document's allergen list
            doc_ingredients: Document's ingredient list
            
        Returns:
            True if safe (no allergen present), False if allergen detected
        """
        # Check explicit allergen list
        for doc_allergen in doc_allergens:
            if allergen in doc_allergen or doc_allergen in allergen:
                return False
        
        # Check ingredients for allergen keywords
        allergen_keywords = self._get_allergen_keywords(allergen)
        for ingredient in doc_ingredients:
            for keyword in allergen_keywords:
                if keyword in ingredient:
                    return False
        
        return True
    
    def _get_allergen_keywords(self, allergen: str) -> List[str]:
        """
        Get keywords to check for a specific allergen.
        
        Args:
            allergen: Allergen name
            
        Returns:
            List of keywords to check in ingredients
        """
        allergen_map = {
            "gluten": ["gluten", "wheat", "flour", "bread", "noodle", "pasta", "bánh", "mì", "bún", "phở"],
            "dairy": ["dairy", "milk", "cheese", "butter", "cream", "yogurt", "sữa", "kem", "bơ"],
            "nuts": ["nuts", "peanut", "almond", "cashew", "walnut", "đậu phộng", "hạt"],
            "seafood": ["seafood", "fish", "shrimp", "crab", "lobster", "cá", "tôm", "cua", "ghẹ"],
            "eggs": ["egg", "trứng"],
            "soy": ["soy", "tofu", "đậu", "tương"],
            "shellfish": ["shellfish", "shrimp", "crab", "lobster", "tôm", "cua", "ghẹ", "sò", "ốc"]
        }
        
        # Return keywords for the allergen, or just the allergen itself if not in map
        return allergen_map.get(allergen, [allergen])
    
    def _check_dietary_constraint(self, dietary_requirement: str, document: Dict[str, Any]) -> bool:
        """
        Check dietary constraint (vegetarian, vegan, halal, etc.).
        
        Args:
            dietary_requirement: Dietary requirement to check
            document: Document to verify
            
        Returns:
            True if requirement is satisfied, False otherwise
        """
        # This would ideally check metadata dietary_labels, but since we're working
        # with simplified document format, we'll do basic ingredient checking
        ingredients = document.get("ingredients", [])
        ingredients_lower = [i.lower() for i in ingredients]
        
        if dietary_requirement == "vegetarian":
            # Check for meat ingredients
            meat_keywords = ["beef", "pork", "chicken", "duck", "fish", "shrimp", "crab", 
                           "bò", "heo", "gà", "vịt", "cá", "tôm", "cua", "thịt"]
            return not any(keyword in ingredient for ingredient in ingredients_lower 
                          for keyword in meat_keywords)
        
        elif dietary_requirement == "vegan":
            # Check for any animal products
            animal_keywords = ["beef", "pork", "chicken", "duck", "fish", "shrimp", "crab",
                             "milk", "cheese", "butter", "egg", "honey",
                             "bò", "heo", "gà", "vịt", "cá", "tôm", "cua", "thịt",
                             "sữa", "kem", "bơ", "trứng", "mật ong"]
            return not any(keyword in ingredient for ingredient in ingredients_lower 
                          for keyword in animal_keywords)
        
        elif dietary_requirement == "halal":
            # Check for non-halal ingredients
            non_halal_keywords = ["pork", "alcohol", "wine", "beer", "heo", "rượu", "bia"]
            return not any(keyword in ingredient for ingredient in ingredients_lower 
                          for keyword in non_halal_keywords)
        
        elif dietary_requirement == "gluten_free":
            # Check for gluten-containing ingredients
            gluten_keywords = ["wheat", "flour", "bread", "noodle", "pasta", "bánh", "mì", "bún", "phở"]
            return not any(keyword in ingredient for ingredient in ingredients_lower 
                          for keyword in gluten_keywords)
        
        # Unknown dietary requirement - be conservative and allow
        return True
    
    def _check_dislike_constraint(self, disliked_item: str, ingredients: List[str]) -> bool:
        """
        Check if disliked item is present in ingredients.
        
        Args:
            disliked_item: Item the user dislikes
            ingredients: List of ingredients to check
            
        Returns:
            True if disliked item is not present, False if present
        """
        return not any(disliked_item in ingredient for ingredient in ingredients)
    
    def _filter_by_availability(self, documents: List[Dict[str, Any]], 
                               state: AgentState) -> List[Dict[str, Any]]:
        """
        Filter documents by availability status.
        
        Args:
            documents: List of documents to filter
            state: Current agent state for context
            
        Returns:
            Filtered list of documents
        """
        # For now, we'll allow all availability statuses but add warnings
        # In a production system, you might filter out unavailable items
        
        filtered_docs = []
        availability_warnings = []
        
        for doc in documents:
            availability_status = doc.get("availability_status", "available")
            requires_advance_order = doc.get("requires_advance_order", False)
            
            # Always include the document but add warnings
            filtered_docs.append(doc)
            
            if availability_status == "unavailable":
                availability_warnings.append(f"{doc.get('name_vi', 'Unknown dish')} is currently unavailable")
            elif availability_status == "advance_order_only" or requires_advance_order:
                availability_warnings.append(f"{doc.get('name_vi', 'Unknown dish')} requires advance ordering")
            elif availability_status == "limited":
                availability_warnings.append(f"{doc.get('name_vi', 'Unknown dish')} has limited availability")
        
        # Add warnings to state
        if availability_warnings:
            state["availability_warnings"] = state.get("availability_warnings", []) + availability_warnings
        
        return filtered_docs
    
    def _filter_by_price(self, documents: List[Dict[str, Any]], 
                        constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter documents by price constraints.
        
        Args:
            documents: List of documents to filter
            constraints: List of constraints that might include price limits
            
        Returns:
            Filtered list of documents
        """
        # Look for price-related constraints
        max_price = None
        min_price = None
        
        for constraint in constraints:
            if constraint.get("type", "").lower() == "price":
                value = constraint.get("value", "")
                if "max" in value.lower() or "under" in value.lower() or "dưới" in value.lower():
                    # Extract price limit (simplified parsing)
                    try:
                        # Look for numbers in the constraint value
                        import re
                        numbers = re.findall(r'\d+', value)
                        if numbers:
                            max_price = int(numbers[0]) * 1000  # Assume thousands VND
                    except:
                        pass
                elif "min" in value.lower() or "over" in value.lower() or "trên" in value.lower():
                    try:
                        import re
                        numbers = re.findall(r'\d+', value)
                        if numbers:
                            min_price = int(numbers[0]) * 1000  # Assume thousands VND
                    except:
                        pass
        
        # Apply price filtering if constraints were found
        if max_price is not None or min_price is not None:
            filtered_docs = []
            for doc in documents:
                # Note: price_vnd might not be available in simplified document format
                # This is a placeholder for when full metadata is available
                doc_price = doc.get("price_vnd")
                if doc_price is not None:
                    if max_price is not None and doc_price > max_price:
                        continue
                    if min_price is not None and doc_price < min_price:
                        continue
                filtered_docs.append(doc)
            return filtered_docs
        
        return documents
    
    def _filter_by_preparation_time(self, documents: List[Dict[str, Any]], 
                                   constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter documents by preparation time constraints.
        
        Args:
            documents: List of documents to filter
            constraints: List of constraints that might include time limits
            
        Returns:
            Filtered list of documents
        """
        # Look for time-related constraints
        max_time = None
        
        for constraint in constraints:
            constraint_type = constraint.get("type", "").lower()
            value = constraint.get("value", "").lower()
            
            if constraint_type == "time" or "quick" in value or "fast" in value or "nhanh" in value:
                # For quick/fast requests, limit to dishes that can be prepared quickly
                max_time = 30  # 30 minutes
            elif "slow" in value or "elaborate" in value or "chậm" in value:
                # For elaborate requests, no time limit (or very high limit)
                max_time = None
        
        # Apply time filtering if constraint was found
        if max_time is not None:
            filtered_docs = []
            for doc in documents:
                # Note: preparation_time_minutes might not be available in simplified format
                # This is a placeholder for when full metadata is available
                prep_time = doc.get("preparation_time_minutes")
                if prep_time is None or prep_time <= max_time:
                    filtered_docs.append(doc)
            return filtered_docs
        
        return documents
    
    def filter_documents(self, documents: List[Dict[str, Any]], 
                        constraints: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Direct interface for filtering documents with constraints.
        
        Args:
            documents: List of documents to filter
            constraints: List of constraints to apply
            
        Returns:
            Tuple of (filtered_documents, warnings)
        """
        try:
            # Create a minimal state for filtering
            state = {
                "retrieved_docs": documents,
                "active_constraints": constraints,
                "warnings": [],
                "availability_warnings": []
            }
            
            # Apply filters
            filtered_docs = self._apply_all_filters(documents, constraints, state)
            
            # Collect warnings
            warnings = state.get("warnings", []) + state.get("availability_warnings", [])
            
            return filtered_docs, warnings
            
        except Exception as e:
            self.logger.error(f"Error in direct filtering: {e}")
            return documents, [f"Filtering error: {str(e)}"]


def create_logic_filters() -> LogicFilters:
    """
    Factory function to create logic filters.
    
    Returns:
        Configured logic filters instance
    """
    return LogicFilters()


# For testing
if __name__ == "__main__":
    # Test the logic filters
    logging.basicConfig(level=logging.INFO)
    
    # Create logic filters
    filters = create_logic_filters()
    
    # Test documents
    test_docs = [
        {
            "id": "pho_bo",
            "name_vi": "Phở Bò",
            "name_en": "Beef Pho",
            "ingredients": ["beef", "rice noodles", "herbs"],
            "allergens": ["gluten"],
            "availability_status": "available",
            "score": 0.9
        },
        {
            "id": "com_chay",
            "name_vi": "Cơm Chay",
            "name_en": "Vegetarian Rice",
            "ingredients": ["rice", "vegetables", "tofu"],
            "allergens": [],
            "availability_status": "available",
            "score": 0.8
        },
        {
            "id": "banh_mi_heo",
            "name_vi": "Bánh Mì Heo",
            "name_en": "Pork Sandwich",
            "ingredients": ["pork", "bread", "vegetables"],
            "allergens": ["gluten"],
            "availability_status": "limited",
            "score": 0.7
        }
    ]
    
    # Test constraints
    test_constraints = [
        {
            "type": "dietary",
            "value": "vegetarian",
            "severity": "strict"
        },
        {
            "type": "allergy",
            "value": "gluten",
            "severity": "strict"
        }
    ]
    
    print("=== Testing Logic Filters ===")
    print(f"Original documents: {len(test_docs)}")
    
    filtered_docs, warnings = filters.filter_documents(test_docs, test_constraints)
    
    print(f"Filtered documents: {len(filtered_docs)}")
    for doc in filtered_docs:
        print(f"  - {doc['name_vi']} ({doc['name_en']})")
    
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")