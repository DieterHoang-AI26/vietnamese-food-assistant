"""
Vietnamese Food Assistant - Data-Driven Fuzzy Matching Module

This module implements Vietnamese-specific fuzzy matching algorithms for food search,
using data-driven approaches instead of hardcoded patterns. It learns from the actual
menu data to build phonetic similarity and term mappings.

Requirements: 8.2, 8.6 - Data-driven, no hardcoding
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Set, Optional
from difflib import SequenceMatcher
import logging
import json
from pathlib import Path
from collections import defaultdict, Counter


class DataDrivenVietnameseFuzzyMatcher:
    """
    Data-driven Vietnamese fuzzy matching implementation that learns patterns
    from actual menu data instead of using hardcoded dictionaries.
    """
    
    def __init__(self, menu_data_path: Optional[str] = None):
        """Initialize data-driven Vietnamese fuzzy matcher."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize menu text data for learning
        self.menu_text_data = []
        
        # Load and analyze menu data FIRST to learn character patterns
        if menu_data_path:
            self._extract_text_data(menu_data_path)
        
        # Dynamic mappings learned from data (AI/ML approach)
        self.tone_mappings = self._build_unicode_tone_mappings()
        self.learned_patterns = {}
        self.food_term_variations = {}
        self.common_words = set()
        
        # Complete learning process
        if menu_data_path:
            self._learn_from_menu_data(menu_data_path)
    
    def _build_unicode_tone_mappings(self) -> Dict[str, str]:
        """
        Build tone mark mappings using AI/ML approach - learn from actual text data.
        No hardcoded character sets, fully data-driven and language-agnostic.
        """
        mappings = {}
        
        # AI/ML Approach: Learn diacritics from actual menu data
        if hasattr(self, 'menu_text_data') and self.menu_text_data:
            # Extract all unique characters from real data
            unique_chars = set(''.join(self.menu_text_data))
            
            # Use Unicode normalization to learn mappings dynamically
            for char in unique_chars:
                if unicodedata.category(char).startswith('L'):  # Letter category
                    # Use Unicode NFD normalization to separate base + combining chars
                    normalized = unicodedata.normalize('NFD', char)
                    base_char = ''.join(c for c in normalized if not unicodedata.combining(c))
                    
                    # Only create mapping if char has diacritics
                    if len(normalized) > 1 and char != base_char:
                        mappings[char] = base_char.lower()
        
        # Fallback: Universal Unicode approach for any unseen characters
        self._universal_normalization_cache = {}
        
        return mappings
    
    def _extract_text_data(self, menu_data_path: str) -> None:
        """
        Extract all text data from menu for AI/ML learning.
        This replaces hardcoded character sets with real data learning.
        """
        try:
            menu_data = self._load_menu_data(menu_data_path)
            
            # Extract ALL text content for character learning
            for item in menu_data:
                # Collect all text fields
                text_fields = []
                
                if 'name_vi' in item and isinstance(item['name_vi'], str):
                    text_fields.append(item['name_vi'])
                if 'name_en' in item and isinstance(item['name_en'], str):
                    text_fields.append(item['name_en'])
                if 'description_vi' in item and isinstance(item['description_vi'], str):
                    text_fields.append(item['description_vi'])
                if 'ingredients' in item and isinstance(item['ingredients'], list):
                    text_fields.extend([ing for ing in item['ingredients'] if isinstance(ing, str)])
                if 'cuisine_tags' in item and isinstance(item['cuisine_tags'], list):
                    text_fields.extend([tag for tag in item['cuisine_tags'] if isinstance(tag, str)])
                
                # Add to learning dataset
                self.menu_text_data.extend(text_fields)
            
            self.logger.info(f"Extracted {len(self.menu_text_data)} text samples for AI/ML learning")
            
        except Exception as e:
            self.logger.warning(f"Could not extract text data: {e}")
            self.menu_text_data = []
    def _learn_from_menu_data(self, menu_data_path: str) -> None:
        """
        Learn patterns from actual menu data instead of hardcoding.
        
        Args:
            menu_data_path: Path to menu data file (JSON or CSV)
        """
        try:
            menu_data = self._load_menu_data(menu_data_path)
            
            # Extract all food-related terms from menu
            all_terms = []
            for item in menu_data:
                # Extract terms from names
                if 'name_vi' in item and isinstance(item['name_vi'], str):
                    all_terms.extend(self._extract_terms(item['name_vi']))
                if 'name_en' in item and isinstance(item['name_en'], str):
                    all_terms.extend(self._extract_terms(item['name_en']))
                
                # Extract from ingredients
                if 'ingredients' in item and isinstance(item['ingredients'], list):
                    for ingredient in item['ingredients']:
                        if isinstance(ingredient, str):
                            all_terms.extend(self._extract_terms(ingredient))
                
                # Extract from cuisine tags
                if 'cuisine_tags' in item and isinstance(item['cuisine_tags'], list):
                    for tag in item['cuisine_tags']:
                        if isinstance(tag, str):
                            all_terms.extend(self._extract_terms(tag))
            
            # Learn common patterns
            self._learn_phonetic_patterns(all_terms)
            self._learn_term_variations(all_terms)
            
            self.logger.info(f"Learned patterns from {len(menu_data)} menu items")
            
        except Exception as e:
            self.logger.warning(f"Could not learn from menu data: {e}")
    
    def _load_menu_data(self, data_path: str) -> List[Dict]:
        """Load menu data from file."""
        path = Path(data_path)
        
        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # For CSV, would need pandas - skip for now
            return []
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract individual terms from text."""
        if not text or not isinstance(text, str):
            return []
        
        # Simple tokenization
        terms = re.findall(r'\b\w+\b', text.lower())
        return [term for term in terms if len(term) >= 2]
    
    def _learn_phonetic_patterns(self, terms: List[str]) -> None:
        """
        Learn phonetic patterns from actual data instead of hardcoding.
        """
        # Group terms by their normalized forms
        normalized_groups = defaultdict(list)
        
        for term in terms:
            normalized = self.normalize_vietnamese_text(term)
            normalized_groups[normalized].append(term)
        
        # Find variations for the same normalized form
        for normalized, variations in normalized_groups.items():
            if len(variations) > 1:
                # These are phonetic variations of the same word
                unique_variations = list(set(variations))
                if len(unique_variations) > 1:
                    self.learned_patterns[normalized] = unique_variations
    
    def _learn_term_variations(self, terms: List[str]) -> None:
        """
        Learn term variations from data (e.g., different ways to write the same food).
        """
        # Count term frequency
        term_counts = Counter(terms)
        self.common_words = set(term for term, count in term_counts.items() if count >= 2)
        
        # Group similar terms
        for term in self.common_words:
            variations = []
            normalized_term = self.normalize_vietnamese_text(term)
            
            # Find other terms with similar normalized forms
            for other_term in self.common_words:
                if other_term != term:
                    other_normalized = self.normalize_vietnamese_text(other_term)
                    if normalized_term == other_normalized:
                        variations.append(other_term)
            
            if variations:
                self.food_term_variations[term] = variations
    
    def normalize_vietnamese_text(self, text: str) -> str:
        """
        Normalize text using AI/ML learned mappings + universal Unicode approach.
        Language-agnostic and scalable to any language.
        
        Args:
            text: Input text in any language
            
        Returns:
            Normalized text without diacritics
        """
        if not text:
            return ""
        
        # Convert to lowercase
        normalized = text.lower().strip()
        
        # AI/ML Approach 1: Apply learned mappings from real data
        for accented, base in self.tone_mappings.items():
            normalized = normalized.replace(accented, base)
        
        # AI/ML Approach 2: Universal Unicode normalization for unseen characters
        # This works for ANY language, not just Vietnamese
        result = []
        for char in normalized:
            if char in self._universal_normalization_cache:
                result.append(self._universal_normalization_cache[char])
            else:
                # Use Unicode NFD to separate base character from diacritics
                nfd_form = unicodedata.normalize('NFD', char)
                base_char = ''.join(c for c in nfd_form if not unicodedata.combining(c))
                
                # Cache for performance
                self._universal_normalization_cache[char] = base_char
                result.append(base_char)
        
        normalized = ''.join(result)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def apply_learned_corrections(self, text: str) -> str:
        """
        Apply corrections based on learned patterns from menu data.
        
        Args:
            text: Input text with potential misspellings
            
        Returns:
            Text with learned corrections applied
        """
        corrected = text.lower()
        words = corrected.split()
        corrected_words = []
        
        for word in words:
            normalized_word = self.normalize_vietnamese_text(word)
            
            # Check if we have learned patterns for this normalized form
            if normalized_word in self.learned_patterns:
                variations = self.learned_patterns[normalized_word]
                # Choose the most common variation (first in list)
                if variations and word not in variations:
                    # Find the best match among variations
                    best_match = max(variations, 
                                   key=lambda v: self._calculate_similarity(word, v))
                    corrected_words.append(best_match)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words."""
        return SequenceMatcher(None, word1, word2).ratio()
    
    def vietnamese_edit_distance(self, word1: str, word2: str) -> int:
        """
        Calculate edit distance with Vietnamese-specific rules.
        Uses algorithmic approach based on Unicode normalization.
        """
        if not word1 or not word2:
            return max(len(word1), len(word2))
        
        # Normalize both words for comparison
        norm1 = self.normalize_vietnamese_text(word1)
        norm2 = self.normalize_vietnamese_text(word2)
        
        # If normalized forms are identical, only tone marks differ
        if norm1 == norm2:
            return 0  # Consider tone mark differences as no penalty
        
        # Standard Levenshtein distance on normalized forms
        return self._levenshtein_distance(norm1, norm2)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate standard Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def phonetic_similarity(self, word1: str, word2: str) -> float:
        """
        Calculate phonetic similarity using learned patterns and normalization.
        """
        if not word1 or not word2:
            return 0.0
        
        # Apply learned corrections
        corrected1 = self.apply_learned_corrections(word1)
        corrected2 = self.apply_learned_corrections(word2)
        
        # Calculate similarity using SequenceMatcher
        similarity = SequenceMatcher(None, corrected1, corrected2).ratio()
        
        # Boost similarity if normalized forms match (only tone marks differ)
        norm1 = self.normalize_vietnamese_text(word1)
        norm2 = self.normalize_vietnamese_text(word2)
        
        if norm1 == norm2:
            similarity = max(similarity, 0.95)  # High similarity for tone mark differences
        
        # Check if words are in learned variations
        if word1 in self.food_term_variations:
            if word2 in self.food_term_variations[word1]:
                similarity = max(similarity, 0.9)
        
        return similarity
    
    def find_fuzzy_matches(self, query: str, candidates: List[str], 
                          threshold: float = 0.6, max_results: int = 10) -> List[Tuple[str, float]]:
        """
        Find fuzzy matches using learned patterns.
        """
        if not query or not candidates:
            return []
        
        matches = []
        
        for candidate in candidates:
            # Calculate phonetic similarity using learned patterns
            similarity = self.phonetic_similarity(query, candidate)
            
            # Check substring matches with normalization
            query_norm = self.normalize_vietnamese_text(query)
            candidate_norm = self.normalize_vietnamese_text(candidate)
            
            if query_norm in candidate_norm or candidate_norm in query_norm:
                similarity = max(similarity, 0.8)
            
            # Check word-level matches
            query_words = query_norm.split()
            candidate_words = candidate_norm.split()
            
            word_matches = sum(1 for qw in query_words 
                             if any(self.phonetic_similarity(qw, cw) > 0.8 for cw in candidate_words))
            
            if word_matches > 0:
                word_match_ratio = word_matches / len(query_words)
                similarity = max(similarity, word_match_ratio * 0.9)
            
            if similarity >= threshold:
                matches.append((candidate, similarity))
        
        # Sort by similarity score (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches[:max_results]
    
    def enhance_food_term_matching(self, query: str, food_names: List[str]) -> List[Tuple[str, float]]:
        """
        Enhanced matching using learned food term patterns.
        """
        if not query or not food_names:
            return []
        
        matches = []
        query_lower = query.lower().strip()
        
        for food_name in food_names:
            food_name_lower = food_name.lower()
            base_similarity = self.phonetic_similarity(query_lower, food_name_lower)
            
            # Check learned food term variations
            enhanced_similarity = base_similarity
            
            # Use learned variations instead of hardcoded mappings
            query_normalized = self.normalize_vietnamese_text(query_lower)
            food_normalized = self.normalize_vietnamese_text(food_name_lower)
            
            # Check if query terms appear in learned common words
            query_words = query_normalized.split()
            food_words = food_normalized.split()
            
            for q_word in query_words:
                if q_word in self.common_words:
                    for f_word in food_words:
                        if f_word in self.common_words:
                            word_similarity = self.phonetic_similarity(q_word, f_word)
                            if word_similarity >= 0.8:
                                enhanced_similarity = max(enhanced_similarity, 0.85)
            
            # Boost for exact matches
            if query_lower == food_name_lower:
                enhanced_similarity = 1.0
            elif query_lower in food_name_lower:
                enhanced_similarity = max(enhanced_similarity, 0.9)
            elif food_name_lower.startswith(query_lower):
                enhanced_similarity = max(enhanced_similarity, 0.85)
            
            if enhanced_similarity > 0.3:
                matches.append((food_name, enhanced_similarity))
        
        # Sort by relevance score
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
    
    def tokenize_vietnamese_food_query(self, query: str) -> List[str]:
        """
        Tokenize Vietnamese food query using learned patterns.
        """
        if not query:
            return []
        
        # Apply learned corrections first
        corrected = self.apply_learned_corrections(query)
        
        # Simple tokenization - split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', corrected.lower())
        
        # Filter out very short tokens and keep only learned common words when possible
        filtered_tokens = []
        for token in tokens:
            if len(token) >= 2:
                # Prefer tokens that are in our learned vocabulary
                if token in self.common_words or len(token) >= 3:
                    filtered_tokens.append(token)
        
        return filtered_tokens if filtered_tokens else tokens


def create_vietnamese_fuzzy_matcher(menu_data_path: Optional[str] = None) -> DataDrivenVietnameseFuzzyMatcher:
    """
    Factory function to create a data-driven Vietnamese fuzzy matcher.
    
    Args:
        menu_data_path: Optional path to menu data for learning patterns
    
    Returns:
        Configured DataDrivenVietnameseFuzzyMatcher instance
    """
    # Try to find menu data automatically
    if not menu_data_path:
        possible_paths = [
            "data/processed_menu_v2.json",
            "data/processed_menu.json", 
            "data/sample_menu.csv"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                menu_data_path = path
                break
    
    return DataDrivenVietnameseFuzzyMatcher(menu_data_path)


# For testing
if __name__ == "__main__":
    # Test the data-driven Vietnamese fuzzy matcher
    matcher = create_vietnamese_fuzzy_matcher()
    
    # Test data - these would normally come from actual menu data
    food_names = [
        "Cháo Thập Cẩm",
        "Cà Phê Sữa Đá",
        "Bún Chả Giò", 
        "Cơm Gà Nướng",
        "Mì Gói Trứng",
        "Bánh Mì",
        "Trà Chanh"
    ]
    
    test_queries = [
        "chao",      # Should match "Cháo Thập Cẩm"
        "ca phe",    # Should match "Cà Phê Sữa Đá"
        "bun cha",   # Should match "Bún Chả Giò"
        "com ga",    # Should match "Cơm Gà Nướng"
        "mi trung",  # Should match "Mì Gói Trứng"
    ]
    
    print("=== Testing Data-Driven Vietnamese Fuzzy Matching ===")
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        matches = matcher.enhance_food_term_matching(query, food_names)
        
        for food_name, score in matches[:3]:
            print(f"  {food_name}: {score:.3f}")
    
    print("\n=== Testing Learned Corrections ===")
    
    test_corrections = [
        "chao thap cam",
        "ca phe sua",
        "bun cha gio",
        "com ga nuong"
    ]
    
    for text in test_corrections:
        corrected = matcher.apply_learned_corrections(text)
        print(f"'{text}' -> '{corrected}'")