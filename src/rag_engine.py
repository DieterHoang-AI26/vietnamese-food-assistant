"""
Vietnamese Food Assistant - RAG Core Engine

This module implements the Retrieval-Augmented Generation core engine
with ChromaDB client and Hybrid Search (Vector + BM25) capabilities.
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime
import re
from collections import Counter

from .menu_database import Dish, SearchResult, AvailabilityStatus
from .etl_pipeline import MenuDataETLV2 as MenuDataETL
from .config import get_config


class RAGEngine:
    """
    RAG Core Engine for Vietnamese Food Assistant.
    
    Implements hybrid search combining vector similarity and BM25 text matching
    for comprehensive menu item retrieval and ranking.
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize RAG engine with ChromaDB client."""
        # Initialize logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.config = get_config()
        self.persist_directory = persist_directory or self.config.database.chroma_persist_directory
        
        # Initialize ChromaDB client
        self.client = self._initialize_chroma_client()
        
        # Initialize embedding function - use fine-tuned model if available
        self.embedding_function = self._initialize_embedding_function()
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        # BM25 parameters
        self.k1 = 1.5  # Term frequency saturation parameter
        self.b = 0.75  # Length normalization parameter
        
        # Document statistics for BM25
        self.doc_frequencies = {}
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.total_docs = 0
    
    def _initialize_embedding_function(self):
        """Initialize embedding function, preferring fine-tuned model if available."""
        fine_tuned_path = self.config.database.fine_tuned_model_path
        
        # Check if fine-tuned model exists
        if fine_tuned_path and Path(fine_tuned_path).exists():
            self.logger.info(f"🎯 Using fine-tuned Vietnamese food embedding model: {fine_tuned_path}")
            try:
                from sentence_transformers import SentenceTransformer
                
                # Load fine-tuned model
                model = SentenceTransformer(fine_tuned_path)
                
                # Create custom embedding function for fine-tuned model
                class FineTunedEmbeddingFunction:
                    def __init__(self, model):
                        self.model = model
                    
                    def __call__(self, input):
                        """Encode texts using fine-tuned model."""
                        if isinstance(input, str):
                            input = [input]
                        
                        embeddings = self.model.encode(input, convert_to_numpy=True)
                        return embeddings.tolist()
                    
                    def embed_documents(self, texts):
                        """Embed documents."""
                        return self.__call__(texts)
                    
                    def embed_query(self, input):
                        """Embed query."""
                        # ChromaDB passes input as a list, so handle both cases
                        if isinstance(input, list):
                            result = self.__call__(input)
                        else:
                            result = self.__call__([input])
                        
                        return result if result else []
                    
                    def name(self):
                        """Return the name of the embedding function."""
                        return "fine_tuned_vietnamese_food"
                
                return FineTunedEmbeddingFunction(model)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load fine-tuned model: {e}")
                self.logger.info("🔄 Falling back to base embedding model")
        
        # Fallback to base model
        self.logger.info(f"📦 Using base embedding model: {self.config.database.embedding_model}")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.database.embedding_model
        )
    
    def _initialize_chroma_client(self) -> chromadb.Client:
        """Initialize ChromaDB client with persistence."""
        persist_path = Path(self.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        settings = Settings(
            persist_directory=str(persist_path),
            anonymized_telemetry=False
        )
        
        client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=settings
        )
        
        self.logger.info(f"ChromaDB client initialized with persistence at {persist_path}")
        return client
    
    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create new one."""
        collection_name = self.config.database.collection_name
        
        try:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            self.logger.info(f"Retrieved existing collection: {collection_name}")
        except Exception:
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info(f"Created new collection: {collection_name}")
        
        return collection
    
    def load_menu_data(self, data_source: str, force_reload: bool = False) -> int:
        """
        Load menu data from CSV/Excel file or processed JSON.
        
        Args:
            data_source: Path to data file (CSV, Excel, or JSON)
            force_reload: Whether to force reload even if collection has data
            
        Returns:
            Number of dishes loaded
        """
        # Check if collection already has data
        if not force_reload and self.collection.count() > 0:
            self.logger.info(f"Collection already has {self.collection.count()} documents")
            return self.collection.count()
        
        data_path = Path(data_source)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data source not found: {data_path}")
        
        # Load dishes based on file type
        if data_path.suffix.lower() == '.json':
            dishes = self._load_from_json(data_path)
        else:
            # Use ETL pipeline for CSV/Excel
            etl = MenuDataETL()
            dishes = etl.process_menu_data(str(data_path))
        
        # Clear existing data if force reload
        if force_reload and self.collection.count() > 0:
            self.client.delete_collection(self.config.database.collection_name)
            self.collection = self._get_or_create_collection()
        
        # Add dishes to collection
        self._add_dishes_to_collection(dishes)
        
        # Update BM25 statistics
        self._update_bm25_statistics()
        
        self.logger.info(f"Loaded {len(dishes)} dishes into collection")
        return len(dishes)
    
    def _load_from_json(self, json_path: Path) -> List[Dish]:
        """Load dishes from processed JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            dishes_data = json.load(f)
        
        dishes = []
        for dish_data in dishes_data:
            # Reconstruct Dish object from JSON data
            # This is a simplified reconstruction - in production, you'd want proper deserialization
            from .menu_database import SearchContent, MenuMetadata, Ingredient, AvailabilityStatus
            
            search_content = SearchContent(**dish_data['search_content'])
            
            metadata_dict = dish_data['metadata']
            metadata_dict['availability_status'] = AvailabilityStatus(metadata_dict['availability_status'])
            metadata = MenuMetadata(**metadata_dict)
            
            ingredients = [Ingredient(**ing_data) for ing_data in dish_data['ingredients']]
            
            dish = Dish(
                id=dish_data['id'],
                search_content=search_content,
                metadata=metadata,
                ingredients=ingredients
            )
            dishes.append(dish)
        
        return dishes
    
    def _add_dishes_to_collection(self, dishes: List[Dish]) -> None:
        """Add dishes to ChromaDB collection."""
        documents = []
        metadatas = []
        ids = []
        
        for dish in dishes:
            documents.append(dish.get_search_text())
            metadatas.append(dish.get_metadata_dict())
            ids.append(dish.id)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_metadata = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metadata,
                ids=batch_ids
            )
    
    def _update_bm25_statistics(self) -> None:
        """Update BM25 statistics for text search."""
        # Get all documents from collection
        results = self.collection.get(include=['documents'])
        documents = results['documents']
        
        self.total_docs = len(documents)
        
        # Calculate document frequencies and lengths
        self.doc_frequencies = {}
        self.doc_lengths = {}
        total_length = 0
        
        for i, doc in enumerate(documents):
            # Tokenize document
            tokens = self._tokenize_vietnamese(doc)
            doc_length = len(tokens)
            self.doc_lengths[i] = doc_length
            total_length += doc_length
            
            # Count unique terms in document
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_frequencies[term] = self.doc_frequencies.get(term, 0) + 1
        
        # Calculate average document length
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0
        
        self.logger.info(f"Updated BM25 statistics: {self.total_docs} docs, avg length: {self.avg_doc_length:.1f}")
    
    def _tokenize_vietnamese(self, text: str) -> List[str]:
        """Tokenize Vietnamese text for BM25 calculation."""
        # Simple tokenization - in production, use proper Vietnamese tokenizer
        text = text.lower()
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [token for token in tokens if len(token) > 1]  # Filter very short tokens
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_tokens: List[str]) -> float:
        """Calculate BM25 score for a document given query terms."""
        score = 0.0
        doc_length = len(doc_tokens)
        term_frequencies = Counter(doc_tokens)
        
        for term in query_terms:
            if term in term_frequencies:
                # Term frequency in document
                tf = term_frequencies[term]
                
                # Document frequency (number of documents containing term)
                df = self.doc_frequencies.get(term, 0)
                
                if df > 0:
                    # IDF calculation
                    idf = np.log((self.total_docs - df + 0.5) / (df + 0.5))
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                    
                    score += idf * (numerator / denominator)
        
        return score
    
    def vector_search(self, query: str, n_results: int = 10, where_filter: Optional[Dict] = None) -> List[SearchResult]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            List of SearchResult objects
        """
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )
        
        search_results = []
        
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                # Convert distance to similarity score (ChromaDB uses cosine distance)
                distance = results['distances'][0][i]
                similarity_score = 1 - distance  # Convert distance to similarity
                
                # Create SearchResult (we'll need to reconstruct Dish object)
                search_result = SearchResult(
                    dish=self._create_dish_from_metadata(doc_id, results['metadatas'][0][i], results['documents'][0][i]),
                    relevance_score=similarity_score,
                    search_method="vector",
                    matched_terms=self._extract_matched_terms(query, results['documents'][0][i])
                )
                search_results.append(search_result)
        
        return search_results
    
    def bm25_search(self, query: str, n_results: int = 10, where_filter: Optional[Dict] = None) -> List[SearchResult]:
        """
        Perform BM25 text search.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            List of SearchResult objects
        """
        query_terms = self._tokenize_vietnamese(query)
        
        # Get all documents (with optional filter)
        results = self.collection.get(
            where=where_filter,
            include=['documents', 'metadatas']
        )
        
        if not results['ids']:
            return []
        
        # Calculate BM25 scores for all documents
        scored_docs = []
        
        for i, (doc_id, document, metadata) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
            doc_tokens = self._tokenize_vietnamese(document)
            bm25_score = self._calculate_bm25_score(query_terms, doc_tokens)
            
            if bm25_score > 0:  # Only include documents with positive scores
                scored_docs.append((doc_id, document, metadata, bm25_score))
        
        # Sort by BM25 score (descending)
        scored_docs.sort(key=lambda x: x[3], reverse=True)
        
        # Convert to SearchResult objects
        search_results = []
        for doc_id, document, metadata, score in scored_docs[:n_results]:
            search_result = SearchResult(
                dish=self._create_dish_from_metadata(doc_id, metadata, document),
                relevance_score=score,
                search_method="bm25",
                matched_terms=self._extract_matched_terms(query, document)
            )
            search_results.append(search_result)
        
        return search_results
    
    def hybrid_search(self, query: str, n_results: int = 10, where_filter: Optional[Dict] = None, 
                     vector_weight: float = None, bm25_weight: float = None) -> List[SearchResult]:
        """
        Perform optimized hybrid search with Vietnamese food-domain specific weighting.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where_filter: Optional metadata filter
            vector_weight: Weight for vector similarity scores (auto-optimized if None)
            bm25_weight: Weight for BM25 scores (auto-optimized if None)
            
        Returns:
            List of SearchResult objects ranked by hybrid score
            
        Requirements: 8.3, 8.4 - Prioritize exact matches over semantic similarity
        """
        from .vietnamese_fuzzy_matching import create_vietnamese_fuzzy_matcher
        
        # Initialize Vietnamese fuzzy matcher for query analysis with menu data
        fuzzy_matcher = create_vietnamese_fuzzy_matcher("data/processed_menu_v2.json")
        
        # Auto-optimize weights based on query characteristics
        if vector_weight is None or bm25_weight is None:
            vector_weight, bm25_weight = self._optimize_search_weights(query, fuzzy_matcher)
        
        # Get results from both methods
        vector_results = self.vector_search(query, n_results * 3, where_filter)  # Get more for better fusion
        bm25_results = self.bm25_search(query, n_results * 3, where_filter)
        
        # Enhanced normalization with Vietnamese food-domain boosting
        if vector_results:
            vector_results = self._apply_food_domain_boosting(vector_results, query, fuzzy_matcher, "vector")
            max_vector_score = max(result.relevance_score for result in vector_results)
            min_vector_score = min(result.relevance_score for result in vector_results)
            vector_range = max_vector_score - min_vector_score
            
            if vector_range > 0:
                for result in vector_results:
                    result.relevance_score = (result.relevance_score - min_vector_score) / vector_range
        
        if bm25_results:
            bm25_results = self._apply_food_domain_boosting(bm25_results, query, fuzzy_matcher, "bm25")
            max_bm25_score = max(result.relevance_score for result in bm25_results)
            min_bm25_score = min(result.relevance_score for result in bm25_results)
            bm25_range = max_bm25_score - min_bm25_score
            
            if bm25_range > 0:
                for result in bm25_results:
                    result.relevance_score = (result.relevance_score - min_bm25_score) / bm25_range
        
        # Combine results by dish ID with enhanced scoring
        combined_scores = {}
        
        # Add vector scores
        for result in vector_results:
            dish_id = result.dish.id
            combined_scores[dish_id] = {
                'dish': result.dish,
                'vector_score': result.relevance_score,
                'bm25_score': 0.0,
                'matched_terms': result.matched_terms,
                'exact_match_bonus': self._calculate_exact_match_bonus(query, result.dish, fuzzy_matcher)
            }
        
        # Add BM25 scores
        for result in bm25_results:
            dish_id = result.dish.id
            if dish_id in combined_scores:
                combined_scores[dish_id]['bm25_score'] = result.relevance_score
                # Merge matched terms
                combined_scores[dish_id]['matched_terms'].extend(result.matched_terms)
            else:
                combined_scores[dish_id] = {
                    'dish': result.dish,
                    'vector_score': 0.0,
                    'bm25_score': result.relevance_score,
                    'matched_terms': result.matched_terms,
                    'exact_match_bonus': self._calculate_exact_match_bonus(query, result.dish, fuzzy_matcher)
                }
        
        # Calculate enhanced hybrid scores
        hybrid_results = []
        for dish_id, scores in combined_scores.items():
            # Base hybrid score
            base_hybrid_score = (vector_weight * scores['vector_score'] + 
                               bm25_weight * scores['bm25_score'])
            
            # Apply exact match bonus (significant boost for exact matches)
            exact_match_bonus = scores['exact_match_bonus']
            final_score = base_hybrid_score + exact_match_bonus
            
            # Remove duplicate matched terms
            unique_matched_terms = list(set(scores['matched_terms']))
            
            search_result = SearchResult(
                dish=scores['dish'],
                relevance_score=final_score,
                search_method="hybrid",
                matched_terms=unique_matched_terms
            )
            hybrid_results.append(search_result)
        
        # Sort by enhanced hybrid score (descending)
        hybrid_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return hybrid_results[:n_results]
    
    def _optimize_search_weights(self, query: str, fuzzy_matcher) -> Tuple[float, float]:
        """
        Auto-optimize search weights based on query characteristics.
        
        Args:
            query: Search query
            fuzzy_matcher: Vietnamese fuzzy matcher instance
            
        Returns:
            Tuple of (vector_weight, bm25_weight)
        """
        query_lower = query.lower().strip()
        query_tokens = fuzzy_matcher.tokenize_vietnamese_food_query(query_lower)
        
        # Default weights favor BM25 for exact matching
        vector_weight = 0.4  # Reduced from 0.7
        bm25_weight = 0.6    # Increased from 0.3
        
        # Adjust weights based on query characteristics
        
        # Single word queries: favor BM25 for exact matching
        if len(query_tokens) == 1:
            vector_weight = 0.3
            bm25_weight = 0.7
        
        # Multi-word queries: balance both methods
        elif len(query_tokens) >= 3:
            vector_weight = 0.5
            bm25_weight = 0.5
        
        # Queries with Vietnamese food terms: favor BM25
        vietnamese_food_terms = ['cháo', 'phở', 'bún', 'cơm', 'bánh', 'mì', 'miến', 'xôi']
        if any(term in query_lower for term in vietnamese_food_terms):
            vector_weight = 0.35
            bm25_weight = 0.65
        
        # Queries with missing tone marks: favor BM25 with phonetic matching
        normalized_query = fuzzy_matcher.normalize_vietnamese_text(query_lower)
        if normalized_query != query_lower:  # Has tone marks
            vector_weight = 0.45
            bm25_weight = 0.55
        else:  # Missing tone marks, likely needs phonetic matching
            vector_weight = 0.3
            bm25_weight = 0.7
        
        return vector_weight, bm25_weight
    
    def _apply_food_domain_boosting(self, results: List[SearchResult], query: str, 
                                  fuzzy_matcher, search_method: str) -> List[SearchResult]:
        """
        Apply Vietnamese food-domain specific boosting to search results.
        
        Args:
            results: Search results to boost
            query: Original query
            fuzzy_matcher: Vietnamese fuzzy matcher
            search_method: "vector" or "bm25"
            
        Returns:
            Boosted search results
        """
        query_lower = query.lower().strip()
        query_tokens = set(fuzzy_matcher.tokenize_vietnamese_food_query(query_lower))
        
        for result in results:
            dish_name_vi = result.dish.search_content.name_vi.lower()
            dish_name_en = (result.dish.search_content.name_en or "").lower()
            
            boost_factor = 1.0
            
            # Use learned food term patterns instead of hardcoded list
            # Count food term matches using learned vocabulary
            food_term_matches = 0
            query_words = query_lower.split()
            dish_words = dish_name_vi.split()
            
            for q_word in query_words:
                if q_word in fuzzy_matcher.common_words:
                    for d_word in dish_words:
                        if d_word in fuzzy_matcher.common_words:
                            similarity = fuzzy_matcher.phonetic_similarity(q_word, d_word)
                            if similarity >= 0.8:
                                food_term_matches += 1
            
            if food_term_matches > 0:
                boost_factor = 1.0 + (food_term_matches * 0.4)  # Progressive boost
            
            # Exact match detection (highest priority)
            if (query_lower == dish_name_vi or 
                fuzzy_matcher.normalize_vietnamese_text(query_lower) == 
                fuzzy_matcher.normalize_vietnamese_text(dish_name_vi)):
                boost_factor = max(boost_factor, 2.5)
            
            # Substring match boosting
            elif query_lower in dish_name_vi or dish_name_vi.startswith(query_lower):
                boost_factor = max(boost_factor, 2.0)
            
            # Phonetic similarity boosting
            elif fuzzy_matcher.phonetic_similarity(query_lower, dish_name_vi) >= 0.85:
                boost_factor = max(boost_factor, 1.8)
            
            # Apply method-specific adjustments
            if search_method == "bm25":
                # BM25 gets extra boost for exact matches (better for precise queries)
                if boost_factor >= 2.0:
                    boost_factor *= 1.2
            elif search_method == "vector":
                # Vector search gets boost for semantic similarity
                if boost_factor < 1.5:  # No exact matches found
                    boost_factor *= 1.1  # Slight boost for semantic understanding
            
            result.relevance_score *= boost_factor
        
        return results
    
    def _calculate_exact_match_bonus(self, query: str, dish, fuzzy_matcher) -> float:
        """
        Calculate exact match bonus for prioritizing exact matches over semantic similarity.
        
        Args:
            query: Search query
            dish: Dish object
            fuzzy_matcher: Vietnamese fuzzy matcher
            
        Returns:
            Exact match bonus score
        """
        query_lower = query.lower().strip()
        dish_name_vi = dish.search_content.name_vi.lower()
        dish_name_en = (dish.search_content.name_en or "").lower()
        
        # Perfect exact match
        if query_lower == dish_name_vi or query_lower == dish_name_en:
            return 1.0  # Significant bonus
        
        # Normalized exact match (ignoring tone marks)
        query_norm = fuzzy_matcher.normalize_vietnamese_text(query_lower)
        name_vi_norm = fuzzy_matcher.normalize_vietnamese_text(dish_name_vi)
        name_en_norm = fuzzy_matcher.normalize_vietnamese_text(dish_name_en)
        
        if query_norm == name_vi_norm or query_norm == name_en_norm:
            return 0.8  # High bonus for tone mark differences only
        
        # Phonetic exact match
        if fuzzy_matcher.phonetic_similarity(query_lower, dish_name_vi) >= 0.95:
            return 0.6  # Good bonus for phonetic matches
        
        # Substring exact match
        if query_lower in dish_name_vi or query_lower in dish_name_en:
            return 0.4  # Medium bonus for substring matches
        
        # Prefix match
        if dish_name_vi.startswith(query_lower) or dish_name_en.startswith(query_lower):
            return 0.3  # Small bonus for prefix matches
        
        return 0.0  # No exact match bonus
    
    def _create_dish_from_metadata(self, dish_id: str, metadata: Dict, document: str) -> Dish:
        """Create a simplified Dish object from ChromaDB metadata."""
        # This is a simplified reconstruction - in production, you'd store complete dish data
        from .menu_database import SearchContent, MenuMetadata, AvailabilityStatus
        
        search_content = SearchContent(
            name_vi=metadata.get('name_vi', ''),
            description_vi=document.split('\n')[0] if document else '',  # Simplified
            name_en=metadata.get('name_en')
        )
        
        # Convert string lists back to lists
        allergens = metadata.get('allergens', '')
        allergens = allergens.split(',') if allergens else []
        
        dietary_labels = metadata.get('dietary_labels', '')
        dietary_labels = dietary_labels.split(',') if dietary_labels else []
        
        seasonal_availability = metadata.get('seasonal_availability', '')
        seasonal_availability = seasonal_availability.split(',') if seasonal_availability else None
        
        menu_metadata = MenuMetadata(
            price_vnd=metadata.get('price_vnd'),
            category=metadata.get('category', 'main'),
            subcategory=metadata.get('subcategory'),
            allergens=allergens,
            dietary_labels=dietary_labels,
            requires_advance_order=metadata.get('requires_advance_order', False),
            availability_status=AvailabilityStatus(metadata.get('availability_status', 'available')),
            preparation_time_minutes=metadata.get('preparation_time_minutes'),
            spice_level=metadata.get('spice_level'),
            seasonal_availability=seasonal_availability
        )
        
        return Dish(
            id=dish_id,
            search_content=search_content,
            metadata=menu_metadata,
            ingredients=[]  # Simplified - would need to reconstruct from stored data
        )
    
    def _extract_matched_terms(self, query: str, document: str) -> List[str]:
        """Extract terms from query that appear in the document."""
        query_terms = set(self._tokenize_vietnamese(query))
        doc_terms = set(self._tokenize_vietnamese(document))
        matched = list(query_terms.intersection(doc_terms))
        return matched
    
    def search_with_availability_check(self, query: str, search_method: str = "hybrid", 
                                      n_results: int = 10, similarity_threshold: float = 0.3,
                                      constraints: Optional[List[Dict]] = None, 
                                      preferences: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Enhanced search interface with availability checking and alternative suggestions.
        
        Args:
            query: Search query text
            search_method: "vector", "bm25", or "hybrid"
            n_results: Number of results to return
            similarity_threshold: Minimum similarity threshold for considering items available
            constraints: List of dietary constraints
            preferences: User preferences for scoring
            
        Returns:
            Dictionary with search results and availability status
        """
        from .vietnamese_fuzzy_matching import create_vietnamese_fuzzy_matcher
        
        # Perform regular search
        results = self.search(query, search_method, n_results, constraints, preferences)
        
        if not results:
            return {
                "results": [],
                "status": "no_results",
                "message": "Không tìm thấy món ăn nào phù hợp với yêu cầu của bạn.",
                "suggestions": []
            }
        
        # Check if top results have good similarity with menu data learning
        fuzzy_matcher = create_vietnamese_fuzzy_matcher("data/processed_menu_v2.json")
        top_result = results[0]
        top_similarity = fuzzy_matcher.phonetic_similarity(
            query.lower(), 
            top_result.dish.search_content.name_vi.lower()
        )
        
        if top_similarity < similarity_threshold:
            # Low similarity suggests the requested item is not available
            # But check if the top result actually contains the query term
            query_lower = query.lower()
            dish_name_lower = top_result.dish.search_content.name_vi.lower()
            
            # If the dish name contains the query term, it's actually a good match
            if (query_lower in dish_name_lower or 
                any(fuzzy_matcher.phonetic_similarity(query_lower, word) >= 0.8 
                    for word in dish_name_lower.split())):
                return {
                    "results": results,
                    "status": "found",
                    "message": f"Tìm thấy {len(results)} món phù hợp với '{query}'.",
                    "suggestions": [],
                    "top_similarity": top_similarity
                }
            
            # Get category-based suggestions
            suggestions = self._get_alternative_suggestions(query, fuzzy_matcher)
            
            return {
                "results": results,
                "status": "low_similarity",
                "message": f"Món '{query}' hiện không có trong menu. Bạn có muốn thử những món tương tự không?",
                "suggestions": suggestions,
                "top_similarity": top_similarity
            }
        
        return {
            "results": results,
            "status": "found",
            "message": f"Tìm thấy {len(results)} món phù hợp với '{query}'.",
            "suggestions": [],
            "top_similarity": top_similarity
        }
    
    def _get_alternative_suggestions(self, query: str, fuzzy_matcher) -> List[str]:
        """
        Get alternative suggestions using learned patterns from menu data.
        
        Args:
            query: Original search query
            fuzzy_matcher: Vietnamese fuzzy matcher instance with learned patterns
            
        Returns:
            List of alternative dish suggestions
        """
        # Get all dishes for category-based suggestions
        all_results = self.search("", search_method="vector", n_results=50)
        
        if not all_results:
            return []
        
        suggestions = []
        query_lower = query.lower()
        
        # Use learned food term variations instead of hardcoded mappings
        query_normalized = fuzzy_matcher.normalize_vietnamese_text(query_lower)
        
        # Find dishes with similar learned patterns
        for result in all_results[:20]:  # Check top 20 results
            dish_name = result.dish.search_content.name_vi
            dish_name_lower = dish_name.lower()
            dish_normalized = fuzzy_matcher.normalize_vietnamese_text(dish_name_lower)
            
            # Check if dish contains learned food terms similar to query
            dish_words = dish_normalized.split()
            query_words = query_normalized.split()
            
            similarity_found = False
            for q_word in query_words:
                if q_word in fuzzy_matcher.common_words:
                    for d_word in dish_words:
                        if d_word in fuzzy_matcher.common_words:
                            if fuzzy_matcher.phonetic_similarity(q_word, d_word) >= 0.7:
                                similarity_found = True
                                break
                    if similarity_found:
                        break
            
            if similarity_found and dish_name not in suggestions:
                suggestions.append(dish_name)
                if len(suggestions) >= 3:
                    break
        
        # If no learned pattern matches, suggest popular items
        if not suggestions:
            popular_dishes = []
            # Get most common dishes from learned vocabulary
            for result in all_results[:10]:
                dish_name = result.dish.search_content.name_vi
                dish_words = fuzzy_matcher.normalize_vietnamese_text(dish_name.lower()).split()
                
                # Check if dish contains many common learned words
                common_word_count = sum(1 for word in dish_words if word in fuzzy_matcher.common_words)
                if common_word_count >= 2:  # Dishes with multiple common food terms
                    popular_dishes.append(dish_name)
            
            suggestions = popular_dishes[:3]
        
        return suggestions[:3]

    def search(self, query: str, search_method: str = "hybrid", n_results: int = 10, 
              constraints: Optional[List[Dict]] = None, preferences: Optional[Dict[str, float]] = None) -> List[SearchResult]:
        """
        Main search interface with constraint filtering and preference scoring.
        
        Args:
            query: Search query text
            search_method: "vector", "bm25", or "hybrid"
            n_results: Number of results to return
            constraints: List of dietary constraints
            preferences: User preferences for scoring
            
        Returns:
            List of SearchResult objects
        """
        # Build metadata filter from constraints
        where_filter = self._build_constraint_filter(constraints)
        
        # Perform search based on method
        if search_method == "vector":
            results = self.vector_search(query, n_results * 2, where_filter)  # Get more for filtering
        elif search_method == "bm25":
            results = self.bm25_search(query, n_results * 2, where_filter)
        else:  # hybrid
            results = self.hybrid_search(query, n_results * 2, where_filter)
        
        # Apply name matching boost for better relevance
        results = self._apply_name_matching_boost(results, query)
        
        # Apply constraint filtering (for constraints not handled by metadata filter)
        if constraints:
            results = self._filter_by_constraints(results, constraints)
        
        # Apply preference scoring
        if preferences:
            results = self._apply_preference_scoring(results, preferences)
        
        return results[:n_results]
    
    def _build_constraint_filter(self, constraints: Optional[List[Dict]]) -> Optional[Dict]:
        """Build ChromaDB where filter from constraints."""
        if not constraints:
            return None
        
        # Build filter for hard constraints that can be handled by metadata
        where_conditions = {}
        
        for constraint in constraints:
            constraint_type = constraint.get('type', '').lower()
            constraint_value = constraint.get('value', '').lower()
            severity = constraint.get('severity', 'strict').lower()
            
            if constraint_type == 'dietary' and severity == 'strict':
                # Filter for dietary labels - use $eq instead of $contains for ChromaDB compatibility
                if constraint_value in ['vegetarian', 'vegan', 'halal', 'gluten_free']:
                    # For now, skip metadata filtering and let logic filters handle it
                    # ChromaDB's where clause is more limited than expected
                    pass
        
        # Return None to skip metadata filtering and rely on logic filters
        return None
    
    def _filter_by_constraints(self, results: List[SearchResult], constraints: List[Dict]) -> List[SearchResult]:
        """Filter search results by dietary constraints."""
        filtered_results = []
        
        for result in results:
            if result.dish.matches_constraints(constraints):
                filtered_results.append(result)
        
        return filtered_results
    
    def _apply_name_matching_boost(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        Enhanced Vietnamese name matching boost with stronger food name prioritization.
        
        Requirements: 8.1, 8.5 - Prioritize exact matches and Vietnamese food terms
        """
        from .vietnamese_fuzzy_matching import create_vietnamese_fuzzy_matcher
        
        # Initialize Vietnamese fuzzy matcher for enhanced matching with menu data
        fuzzy_matcher = create_vietnamese_fuzzy_matcher("data/processed_menu_v2.json")
        
        query_lower = query.lower().strip()
        query_normalized = fuzzy_matcher.normalize_vietnamese_text(query_lower)
        query_corrected = fuzzy_matcher.apply_learned_corrections(query_lower)  # Use learned corrections
        query_tokens = set(fuzzy_matcher.tokenize_vietnamese_food_query(query_corrected))
        
        for result in results:
            dish_name_vi = result.dish.search_content.name_vi.lower()
            dish_name_en = (result.dish.search_content.name_en or "").lower()
            
            # Normalize dish names for better comparison
            dish_name_vi_norm = fuzzy_matcher.normalize_vietnamese_text(dish_name_vi)
            dish_name_en_norm = fuzzy_matcher.normalize_vietnamese_text(dish_name_en)
            
            # Token-based matching with Vietnamese-specific tokenization (define early)
            name_tokens_vi = set(fuzzy_matcher.tokenize_vietnamese_food_query(dish_name_vi))
            name_tokens_en = set(fuzzy_matcher.tokenize_vietnamese_food_query(dish_name_en))
            
            boost_factor = 1.0
            
            # TIER 1: Perfect matches (highest priority)
            if (query_lower == dish_name_vi or query_lower == dish_name_en or
                query_normalized == dish_name_vi_norm or query_normalized == dish_name_en_norm):
                boost_factor = 3.0  # Increased from 2.0
                
            # TIER 2: Phonetic exact matches (Vietnamese-specific)
            elif (query_corrected == dish_name_vi or 
                  fuzzy_matcher.phonetic_similarity(query_lower, dish_name_vi) >= 0.95):
                boost_factor = 2.8
                
            # TIER 3: Substring matches in names (strong boost)
            elif (query_lower in dish_name_vi or query_lower in dish_name_en or
                  query_normalized in dish_name_vi_norm or query_normalized in dish_name_en_norm):
                boost_factor = 2.5  # Increased from 1.8
                
            # TIER 4: Name starts with query (strong boost for prefix matching)
            elif (dish_name_vi.startswith(query_lower) or dish_name_en.startswith(query_lower) or
                  dish_name_vi_norm.startswith(query_normalized)):
                boost_factor = 2.2  # Increased from 1.6
                
            # TIER 5: Vietnamese food term matching (data-driven)
            else:
                # Use learned food terms from menu data instead of hardcoded list
                # Calculate exact token overlap
                exact_overlap_vi = len(query_tokens.intersection(name_tokens_vi))
                exact_overlap_en = len(query_tokens.intersection(name_tokens_en))
                max_exact_overlap = max(exact_overlap_vi, exact_overlap_en)
                
                # Calculate fuzzy token overlap using learned patterns
                fuzzy_overlap_score = 0.0
                for q_token in query_tokens:
                    for name_token in name_tokens_vi.union(name_tokens_en):
                        similarity = fuzzy_matcher.phonetic_similarity(q_token, name_token)
                        if similarity >= 0.8:  # High similarity threshold
                            fuzzy_overlap_score += similarity
                
                # Apply token-based boosting
                if max_exact_overlap > 0:
                    exact_overlap_ratio = max_exact_overlap / len(query_tokens)
                    boost_factor = 1.0 + (exact_overlap_ratio * 1.2)  # Increased multiplier
                
                if fuzzy_overlap_score > 0:
                    fuzzy_boost = min(fuzzy_overlap_score / len(query_tokens), 1.0) * 1.0
                    boost_factor = max(boost_factor, 1.0 + fuzzy_boost)
                
                # Enhanced food term matching using learned common words
                food_term_matches = 0
                for q_token in query_tokens:
                    # Check if query token is in learned common words
                    if q_token in fuzzy_matcher.common_words:
                        for name_token in name_tokens_vi:
                            if name_token in fuzzy_matcher.common_words:
                                if fuzzy_matcher.phonetic_similarity(q_token, name_token) >= 0.8:
                                    food_term_matches += 1
                
                # Apply food term boost based on learned patterns
                if food_term_matches > 0:
                    food_term_boost = 1.5 + (food_term_matches * 0.3)  # Progressive boost
                    boost_factor = max(boost_factor, food_term_boost)
                
                # Special boost for single-word food queries using learned vocabulary
                if len(query_tokens) == 1:
                    single_token = list(query_tokens)[0]
                    if single_token in fuzzy_matcher.common_words:
                        if any(fuzzy_matcher.phonetic_similarity(single_token, nt) >= 0.85 
                               for nt in name_tokens_vi if nt in fuzzy_matcher.common_words):
                            boost_factor = max(boost_factor, 2.0)
            
            # Additional boost for category vs name distinction
            # Names should always rank higher than category matches
            dish_category = result.dish.metadata.category.lower()
            dish_subcategory = (result.dish.metadata.subcategory or "").lower()
            
            # If query matches name tokens but not category, give extra boost
            name_match = any(qt in name_tokens_vi.union(name_tokens_en) for qt in query_tokens)
            category_match = any(qt in dish_category or qt in dish_subcategory for qt in query_tokens)
            
            if name_match and not category_match:
                boost_factor *= 1.3  # Extra boost for name-only matches
            elif category_match and not name_match:
                boost_factor *= 0.8  # Slight penalty for category-only matches
            
            # Apply boost with minimum threshold
            result.relevance_score *= max(boost_factor, 1.0)
        
        # Re-sort by boosted scores
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def _apply_preference_scoring(self, results: List[SearchResult], preferences: Dict[str, float]) -> List[SearchResult]:
        """Apply preference scoring to search results."""
        for result in results:
            preference_score = result.dish.calculate_preference_score(preferences)
            # Combine with relevance score (weighted average)
            combined_score = 0.7 * result.relevance_score + 0.3 * preference_score
            result.relevance_score = combined_score
        
        # Re-sort by combined score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        count = self.collection.count()
        
        if count == 0:
            return {"total_documents": 0}
        
        # Get sample of documents for analysis
        sample_results = self.collection.get(limit=min(100, count), include=['metadatas'])
        
        categories = {}
        price_ranges = {}
        dietary_labels = {}
        
        for metadata in sample_results['metadatas']:
            # Count categories
            category = metadata.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            
            # Count price ranges
            price_range = metadata.get('price_range', 'unknown')
            price_ranges[price_range] = price_ranges.get(price_range, 0) + 1
            
            # Count dietary labels
            labels = metadata.get('dietary_labels', [])
            for label in labels:
                dietary_labels[label] = dietary_labels.get(label, 0) + 1
        
        return {
            "total_documents": count,
            "categories": categories,
            "price_ranges": price_ranges,
            "dietary_labels": dietary_labels,
            "bm25_stats": {
                "total_docs": self.total_docs,
                "avg_doc_length": self.avg_doc_length,
                "unique_terms": len(self.doc_frequencies)
            }
        }


def main():
    """Main function for testing RAG engine."""
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Load sample data
    data_file = "data/sample_menu.csv"
    num_loaded = rag.load_menu_data(data_file, force_reload=True)
    print(f"Loaded {num_loaded} dishes into RAG engine")
    
    # Test searches
    test_queries = [
        "phở bò",
        "món chay",
        "cay",
        "noodle soup",
        "dessert"
    ]
    
    print("\n=== Testing Search Methods ===")
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Test hybrid search
        results = rag.search(query, search_method="hybrid", n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.dish.search_content.name_vi} ({result.dish.search_content.name_en})")
            print(f"     Score: {result.relevance_score:.3f} | Method: {result.search_method}")
            print(f"     Category: {result.dish.metadata.category} | Price: {result.dish.metadata.price_vnd} VND")
    
    # Print collection statistics
    print("\n=== Collection Statistics ===")
    stats = rag.get_collection_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()