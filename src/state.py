"""
Vietnamese Food Assistant - Agent State Definition

This module defines the AgentState structure used throughout the LangGraph workflow.
The state maintains conversation context, constraints, and retrieved documents.
"""

from typing import TypedDict, List, Dict, Optional, Any
from datetime import datetime


class ConversationTurn(TypedDict):
    """Represents a single turn in the conversation history."""
    timestamp: datetime
    user_input: str
    corrected_input: str
    system_response: str
    mentioned_entities: List[str]


class Constraint(TypedDict):
    """Represents a dietary constraint or preference."""
    type: str  # ALLERGY, DIETARY, PREFERENCE, DISLIKE
    value: str
    severity: str  # STRICT, MODERATE, MILD
    source_text: str


class RetrievedDocument(TypedDict):
    """Represents a document retrieved from the menu database."""
    id: str
    name_vi: str
    name_en: Optional[str]
    description: str
    category: str
    ingredients: List[str]
    allergens: List[str]
    requires_advance_order: bool
    availability_status: str
    score: float  # relevance score from retrieval


class AgentState(TypedDict):
    """
    Main state structure for the Vietnamese Food Assistant LangGraph workflow.
    
    This state is passed between nodes and maintains all necessary context
    for processing user requests and generating appropriate responses.
    """
    
    # Input Processing
    raw_input: str  # Original ASR text input
    corrected_input: str  # Text after ASR correction
    processed_input: str  # Text after language processing
    
    # Conversation History
    session_id: str
    conversation_history: List[ConversationTurn]
    mentioned_dishes: List[str]  # Dishes mentioned in current session
    
    # Constraints and Preferences
    active_constraints: List[Constraint]
    session_preferences: Dict[str, float]  # preference scores
    
    # Retrieved Documents
    retrieved_docs: List[RetrievedDocument]
    filtered_docs: List[RetrievedDocument]  # After applying constraints
    reranked_docs: List[RetrievedDocument]  # After reranking
    
    # Processing Metadata
    intent: str  # Extracted user intent
    entities: List[str]  # Extracted entities
    references: List[str]  # Resolved references ("món này", "món đó")
    
    # Response Generation
    response_text: str
    clarification_needed: bool
    follow_up_questions: List[str]
    availability_warnings: List[str]
    
    # Error Handling
    errors: List[str]
    warnings: List[str]