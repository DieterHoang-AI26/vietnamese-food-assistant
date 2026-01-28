"""
Vietnamese Food Assistant - Reranking Node

This module implements reranking functionality to sort filtered results
using cross-encoder models and preference-based scoring to return the top-3
most relevant dishes.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from sentence_transformers import CrossEncoder
from ..state import AgentState
from ..config import get_config


class RerankingNode:
    """
    Reranking node that applies cross-encoder scoring and preference-based
    reranking to sort filtered results and return the top-3 most relevant dishes.
    """
    
    def __init__(self, cross_encoder_model: Optional[str] = None):
        """
        Initialize reranking node.
        
        Args:
            cross_encoder_model: Optional cross-encoder model name.
                                If None, uses a lightweight fallback approach.
        """
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        
        # Initialize cross-encoder model
        self.cross_encoder = None
        self.use_cross_encoder = False
        
        if cross_encoder_model:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model)
                self.use_cross_encoder = True
                self.logger.info(f"Initialized cross-encoder: {cross_encoder_model}")
            except Exception as e:
                self.logger.warning(f"Failed to load cross-encoder {cross_encoder_model}: {e}")
                self.logger.info("Falling back to lightweight reranking")
        else:
            self.logger.info("Using lightweight reranking without cross-encoder")
    
    def __call__(self, state: AgentState) -> AgentState:
        """
        Apply reranking to filtered documents.
        
        Args:
            state: Current agent state with filtered documents
            
        Returns:
            Updated agent state with reranked documents (top-3)
        """
        try:
            filtered_docs = state.get("filtered_docs", [])
            corrected_input = state.get("corrected_input", "")
            session_preferences = state.get("session_preferences", {})
            
            if not filtered_docs:
                self.logger.info("No filtered documents to rerank")
                state["reranked_docs"] = []
                return state
            
            self.logger.info(f"Reranking {len(filtered_docs)} filtered documents")
            
            # Apply reranking
            reranked_docs = self._rerank_documents(
                documents=filtered_docs,
                query=corrected_input,
                preferences=session_preferences
            )
            
            # Limit to top-3 results
            max_results = self.config.max_reranked_docs
            top_docs = reranked_docs[:max_results]
            
            # Update state
            state["reranked_docs"] = top_docs
            
            self.logger.info(f"Reranked to top {len(top_docs)} documents")
            
            # Log top results for debugging
            if top_docs:
                self.logger.debug("Top reranked results:")
                for i, doc in enumerate(top_docs, 1):
                    self.logger.debug(f"  {i}. {doc['name_vi']} (score: {doc['score']:.3f})")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in reranking node: {e}")
            state["errors"] = state.get("errors", []) + [f"Reranking error: {str(e)}"]
            # Fallback: use filtered docs as reranked docs
            filtered_docs = state.get("filtered_docs", [])
            max_results = self.config.max_reranked_docs
            state["reranked_docs"] = filtered_docs[:max_results]
            return state
    
    def _rerank_documents(self, documents: List[Dict[str, Any]], 
                         query: str, preferences: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder and preference scoring.
        
        Args:
            documents: List of documents to rerank
            query: Original query for cross-encoder scoring
            preferences: User preferences for preference scoring
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        # Create a copy to avoid modifying original documents
        docs_to_rank = [doc.copy() for doc in documents]
        
        # Apply cross-encoder scoring if available
        if self.use_cross_encoder and query.strip():
            docs_to_rank = self._apply_cross_encoder_scoring(docs_to_rank, query)
        
        # Apply preference-based scoring
        if preferences:
            docs_to_rank = self._apply_preference_scoring(docs_to_rank, preferences)
        
        # Apply additional ranking factors
        docs_to_rank = self._apply_additional_ranking_factors(docs_to_rank)
        
        # Sort by final score (descending)
        docs_to_rank.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return docs_to_rank
    
    def _apply_cross_encoder_scoring(self, documents: List[Dict[str, Any]], 
                                   query: str) -> List[Dict[str, Any]]:
        """
        Apply cross-encoder scoring to documents.
        
        Args:
            documents: List of documents to score
            query: Query text for cross-encoder
            
        Returns:
            Documents with updated cross-encoder scores
        """
        try:
            # Prepare query-document pairs for cross-encoder
            pairs = []
            for doc in documents:
                # Create document text for cross-encoder
                doc_text = self._create_document_text(doc)
                pairs.append([query, doc_text])
            
            # Get cross-encoder scores
            ce_scores = self.cross_encoder.predict(pairs)
            
            # Update document scores with cross-encoder scores
            for i, doc in enumerate(documents):
                original_score = doc.get("score", 0)
                ce_score = float(ce_scores[i])
                
                # Combine original retrieval score with cross-encoder score
                # Weight: 60% cross-encoder, 40% original retrieval score
                combined_score = 0.6 * ce_score + 0.4 * original_score
                doc["score"] = combined_score
                doc["cross_encoder_score"] = ce_score
                doc["original_retrieval_score"] = original_score
            
            self.logger.debug(f"Applied cross-encoder scoring to {len(documents)} documents")
            
        except Exception as e:
            self.logger.error(f"Error in cross-encoder scoring: {e}")
            # Fallback: use original scores
        
        return documents
    
    def _create_document_text(self, document: Dict[str, Any]) -> str:
        """
        Create text representation of document for cross-encoder.
        
        Args:
            document: Document to convert to text
            
        Returns:
            Text representation of the document
        """
        components = []
        
        # Add names
        if document.get("name_vi"):
            components.append(document["name_vi"])
        if document.get("name_en"):
            components.append(document["name_en"])
        
        # Add description
        if document.get("description"):
            components.append(document["description"])
        
        # Add category
        if document.get("category"):
            components.append(f"Category: {document['category']}")
        
        # Add ingredients
        ingredients = document.get("ingredients", [])
        if ingredients:
            components.append(f"Ingredients: {', '.join(ingredients[:5])}")  # Limit to first 5
        
        return " | ".join(components)
    
    def _apply_preference_scoring(self, documents: List[Dict[str, Any]], 
                                preferences: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Apply preference-based scoring to documents.
        
        Args:
            documents: List of documents to score
            preferences: User preferences dictionary
            
        Returns:
            Documents with updated preference scores
        """
        if not preferences:
            return documents
        
        for doc in documents:
            preference_score = self._calculate_preference_score(doc, preferences)
            
            # Combine with existing score
            original_score = doc.get("score", 0)
            # Weight: 70% original score, 30% preference score
            combined_score = 0.7 * original_score + 0.3 * preference_score
            
            doc["score"] = combined_score
            doc["preference_score"] = preference_score
        
        self.logger.debug(f"Applied preference scoring to {len(documents)} documents")
        return documents
    
    def _calculate_preference_score(self, document: Dict[str, Any], 
                                  preferences: Dict[str, float]) -> float:
        """
        Calculate preference score for a document.
        
        Args:
            document: Document to score
            preferences: User preferences
            
        Returns:
            Preference score (0.0 to 1.0)
        """
        score = 0.0
        total_weight = 0.0
        
        # Check category preferences
        category = document.get("category", "").lower()
        if category in preferences:
            score += preferences[category]
            total_weight += 1.0
        
        # Check ingredient preferences
        ingredients = document.get("ingredients", [])
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower()
            for pref_key, pref_value in preferences.items():
                if pref_key in ingredient_lower or ingredient_lower in pref_key:
                    score += pref_value
                    total_weight += 1.0
        
        # Check name preferences (for cuisine types, cooking methods, etc.)
        name_vi = document.get("name_vi", "").lower()
        name_en = document.get("name_en", "").lower()
        
        for pref_key, pref_value in preferences.items():
            if pref_key in name_vi or pref_key in name_en:
                score += pref_value
                total_weight += 1.0
        
        # Normalize score
        if total_weight > 0:
            return min(1.0, max(0.0, score / total_weight))
        
        return 0.5  # Neutral score if no preferences match
    
    def _apply_additional_ranking_factors(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply additional ranking factors like availability, popularity, etc.
        
        Args:
            documents: List of documents to enhance
            
        Returns:
            Documents with additional ranking factors applied
        """
        for doc in documents:
            original_score = doc.get("score", 0)
            boost = 0.0
            
            # Boost for immediate availability
            availability_status = doc.get("availability_status", "available")
            if availability_status == "available":
                boost += 0.1
            elif availability_status == "limited":
                boost += 0.05
            elif availability_status == "advance_order_only":
                boost -= 0.05
            elif availability_status == "unavailable":
                boost -= 0.2
            
            # Boost for dishes that don't require advance ordering
            if not doc.get("requires_advance_order", False):
                boost += 0.05
            
            # Apply boost (but keep score in reasonable range)
            final_score = original_score + boost
            doc["score"] = max(0.0, min(1.0, final_score))
            doc["availability_boost"] = boost
        
        return documents
    
    def rerank_documents(self, documents: List[Dict[str, Any]], 
                        query: str = "", 
                        preferences: Optional[Dict[str, float]] = None,
                        max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Direct interface for reranking documents.
        
        Args:
            documents: List of documents to rerank
            query: Query text for cross-encoder scoring
            preferences: User preferences for scoring
            max_results: Maximum number of results to return
            
        Returns:
            Reranked list of documents
        """
        try:
            if not documents:
                return []
            
            # Apply reranking
            reranked_docs = self._rerank_documents(
                documents=documents,
                query=query,
                preferences=preferences or {}
            )
            
            # Limit results if specified
            if max_results is not None:
                reranked_docs = reranked_docs[:max_results]
            
            return reranked_docs
            
        except Exception as e:
            self.logger.error(f"Error in direct reranking: {e}")
            # Fallback: return original documents sorted by score
            sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)
            if max_results is not None:
                sorted_docs = sorted_docs[:max_results]
            return sorted_docs


def create_reranking_node(cross_encoder_model: Optional[str] = None) -> RerankingNode:
    """
    Factory function to create a reranking node.
    
    Args:
        cross_encoder_model: Optional cross-encoder model name
        
    Returns:
        Configured reranking node
    """
    return RerankingNode(cross_encoder_model=cross_encoder_model)


# For testing
if __name__ == "__main__":
    # Test the reranking node
    logging.basicConfig(level=logging.INFO)
    
    # Create reranking node (without cross-encoder for testing)
    reranking_node = create_reranking_node()
    
    # Test documents
    test_docs = [
        {
            "id": "pho_bo",
            "name_vi": "Phở Bò",
            "name_en": "Beef Pho",
            "description": "Traditional Vietnamese beef noodle soup",
            "category": "main",
            "ingredients": ["beef", "rice noodles", "herbs", "broth"],
            "availability_status": "available",
            "requires_advance_order": False,
            "score": 0.8
        },
        {
            "id": "com_chay",
            "name_vi": "Cơm Chay",
            "name_en": "Vegetarian Rice",
            "description": "Healthy vegetarian rice dish",
            "category": "main",
            "ingredients": ["rice", "vegetables", "tofu"],
            "availability_status": "available",
            "requires_advance_order": False,
            "score": 0.7
        },
        {
            "id": "banh_xeo",
            "name_vi": "Bánh Xèo",
            "name_en": "Vietnamese Pancake",
            "description": "Crispy Vietnamese pancake with shrimp and pork",
            "category": "main",
            "ingredients": ["rice flour", "shrimp", "pork", "bean sprouts"],
            "availability_status": "limited",
            "requires_advance_order": True,
            "score": 0.9
        }
    ]
    
    # Test preferences
    test_preferences = {
        "beef": 0.8,
        "vegetarian": 0.6,
        "main": 0.7
    }
    
    print("=== Testing Reranking Node ===")
    print("Original order:")
    for i, doc in enumerate(test_docs, 1):
        print(f"  {i}. {doc['name_vi']} (score: {doc['score']:.3f})")
    
    # Test reranking
    reranked_docs = reranking_node.rerank_documents(
        documents=test_docs,
        query="beef noodle soup",
        preferences=test_preferences,
        max_results=3
    )
    
    print("\nReranked order:")
    for i, doc in enumerate(reranked_docs, 1):
        print(f"  {i}. {doc['name_vi']} (score: {doc['score']:.3f})")
        if "preference_score" in doc:
            print(f"     Preference: {doc['preference_score']:.3f}")
        if "availability_boost" in doc:
            print(f"     Availability boost: {doc['availability_boost']:.3f}")