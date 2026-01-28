"""
Vietnamese Food Assistant - Retrieval Node

This module implements the retrieval node that executes hybrid search
with corrected query text to find relevant menu items.
"""

from typing import List, Dict, Any, Optional
import logging
from ..rag_engine import RAGEngine
from ..state import AgentState
from ..config import get_config


class RetrievalNode:
    """
    Retrieval node that executes hybrid search with corrected query.
    
    This node takes the corrected input text from the ASR correction node
    and performs hybrid search (vector + BM25) to retrieve relevant menu items.
    """
    
    def __init__(self, rag_engine: Optional[RAGEngine] = None):
        """
        Initialize retrieval node.
        
        Args:
            rag_engine: Optional RAG engine instance. If None, creates new one.
        """
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        self.rag_engine = rag_engine or RAGEngine()
        
        # Ensure menu data is loaded
        self._ensure_menu_data_loaded()
    
    def _ensure_menu_data_loaded(self):
        """Ensure menu data is loaded in RAG engine."""
        try:
            # Check if collection has data
            stats = self.rag_engine.get_collection_stats()
            if stats.get("total_documents", 0) == 0:
                # Load menu data
                menu_data_path = self.config.menu_data_path
                if menu_data_path.exists():
                    self.rag_engine.load_menu_data(str(menu_data_path))
                    self.logger.info(f"Loaded menu data from {menu_data_path}")
                else:
                    self.logger.warning(f"Menu data file not found: {menu_data_path}")
        except Exception as e:
            self.logger.error(f"Error ensuring menu data loaded: {e}")
    
    def __call__(self, state: AgentState) -> AgentState:
        """
        Execute retrieval with corrected query.
        
        Args:
            state: Current agent state containing corrected input
            
        Returns:
            Updated agent state with retrieved documents
        """
        try:
            # Get corrected query from state
            query = state.get("corrected_input", "").strip()
            
            if not query:
                self.logger.warning("No corrected input found in state")
                state["retrieved_docs"] = []
                state["warnings"] = state.get("warnings", []) + ["No query text for retrieval"]
                return state
            
            self.logger.info(f"Executing retrieval for query: '{query}'")
            
            # Extract any existing constraints from state for filtering
            constraints = self._extract_constraints_for_search(state)
            
            # Execute hybrid search
            search_results = self.rag_engine.search(
                query=query,
                search_method="hybrid",
                n_results=self.config.max_retrieved_docs,
                constraints=constraints
            )
            
            # Convert search results to retrieved documents format
            retrieved_docs = []
            for result in search_results:
                retrieved_doc = result.to_retrieved_document()
                retrieved_docs.append(retrieved_doc)
            
            # Update state with retrieved documents
            state["retrieved_docs"] = retrieved_docs
            
            self.logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Log top results for debugging
            if retrieved_docs:
                self.logger.debug("Top retrieval results:")
                for i, doc in enumerate(retrieved_docs[:3], 1):
                    self.logger.debug(f"  {i}. {doc['name_vi']} (score: {doc['score']:.3f})")
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error in retrieval node: {e}")
            state["errors"] = state.get("errors", []) + [f"Retrieval error: {str(e)}"]
            state["retrieved_docs"] = []
            return state
    
    def _extract_constraints_for_search(self, state: AgentState) -> Optional[List[Dict[str, Any]]]:
        """
        Extract constraints from state for search filtering.
        
        Args:
            state: Current agent state
            
        Returns:
            List of constraints or None if no constraints
        """
        active_constraints = state.get("active_constraints", [])
        
        if not active_constraints:
            return None
        
        # Convert constraints to format expected by RAG engine
        search_constraints = []
        for constraint in active_constraints:
            search_constraints.append({
                "type": constraint.get("type", ""),
                "value": constraint.get("value", ""),
                "severity": constraint.get("severity", "strict")
            })
        
        return search_constraints if search_constraints else None
    
    def search_with_query(self, query: str, constraints: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """
        Direct search interface for testing and external use.
        
        Args:
            query: Search query text
            constraints: Optional list of constraints
            
        Returns:
            List of retrieved documents
        """
        try:
            search_results = self.rag_engine.search(
                query=query,
                search_method="hybrid",
                n_results=self.config.max_retrieved_docs,
                constraints=constraints
            )
            
            retrieved_docs = []
            for result in search_results:
                retrieved_doc = result.to_retrieved_document()
                retrieved_docs.append(retrieved_doc)
            
            return retrieved_docs
            
        except Exception as e:
            self.logger.error(f"Error in direct search: {e}")
            return []
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        try:
            return self.rag_engine.get_collection_stats()
        except Exception as e:
            self.logger.error(f"Error getting retrieval stats: {e}")
            return {"error": str(e)}


def create_retrieval_node(rag_engine: Optional[RAGEngine] = None) -> RetrievalNode:
    """
    Factory function to create a retrieval node.
    
    Args:
        rag_engine: Optional RAG engine instance
        
    Returns:
        Configured retrieval node
    """
    return RetrievalNode(rag_engine=rag_engine)


# For testing
if __name__ == "__main__":
    # Test the retrieval node
    logging.basicConfig(level=logging.INFO)
    
    # Create retrieval node
    retrieval_node = create_retrieval_node()
    
    # Test queries
    test_queries = [
        "phở bò",
        "món chay",
        "cay",
        "noodle soup",
        "dessert"
    ]
    
    print("=== Testing Retrieval Node ===")
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = retrieval_node.search_with_query(query)
        
        for i, doc in enumerate(results[:3], 1):
            print(f"  {i}. {doc['name_vi']} ({doc.get('name_en', 'N/A')})")
            print(f"     Score: {doc['score']:.3f} | Category: {doc['category']}")
    
    # Print stats
    print("\n=== Retrieval Statistics ===")
    stats = retrieval_node.get_retrieval_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")