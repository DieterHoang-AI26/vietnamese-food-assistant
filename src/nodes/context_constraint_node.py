"""
Vietnamese Food Assistant - Combined Context & Constraint Node

This module combines context management and constraint processing into a single
LangGraph node for efficient processing in the pipeline.

Requirements: 2.1, 2.4, 3.1, 7.2
"""

from typing import Dict, Any, List
from src.state import AgentState
from src.nodes.context_manager import ContextManager
from src.nodes.constraint_extraction import ConstraintExtractor
from src.nodes.constraint_accumulator import ConstraintAccumulator


class ContextConstraintNode:
    """
    Combined node that handles both context management and constraint processing.
    
    This node orchestrates:
    1. Context management (conversation history)
    2. Constraint extraction from user input
    3. Constraint accumulation and conflict resolution
    """
    
    def __init__(self):
        self.context_manager = ContextManager()
        self.constraint_extractor = ConstraintExtractor()
        self.constraint_accumulator = ConstraintAccumulator()
    
    def process(self, state: AgentState) -> AgentState:
        """
        Process context and constraints in the correct order.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with context and constraints processed
        """
        try:
            # Step 1: Update conversation context
            state = self._update_context(state)
            
            # Step 2: Extract constraints from current input
            state = self._extract_constraints(state)
            
            # Step 3: Accumulate constraints with session history
            state = self._accumulate_constraints(state)
            
            return state
            
        except Exception as e:
            # Handle errors gracefully
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"Context & Constraint processing failed: {str(e)}")
            return state
    
    def _update_context(self, state: AgentState) -> AgentState:
        """Update conversation context."""
        if "raw_input" in state and state["raw_input"]:
            corrected_input = state.get("corrected_input", state["raw_input"])
            entities = state.get("entities", [])
            
            state = self.context_manager.update_context(
                state=state,
                user_input=state["raw_input"],
                corrected_input=corrected_input,
                mentioned_entities=entities
            )
        
        # Check for session expiration
        if self.context_manager.is_session_expired(state):
            if "warnings" not in state:
                state["warnings"] = []
            state["warnings"].append("Session expired - context may be limited")
        
        return state
    
    def _extract_constraints(self, state: AgentState) -> AgentState:
        """Extract constraints from user input."""
        user_input = state.get("corrected_input", state.get("raw_input", ""))
        conversation_history = state.get("conversation_history", [])
        mentioned_dishes = state.get("mentioned_dishes", [])
        
        if user_input:
            # Build context for reference resolution
            context = self._build_extraction_context(conversation_history, mentioned_dishes)
            
            # Extract constraints using LLM
            extracted_constraints = self.constraint_extractor._prompt_constraint_extraction(
                user_input, context
            )
            
            # Store extracted constraints temporarily for accumulation step
            state["extracted_constraints"] = extracted_constraints
            
            # Update entities and references
            if "entities" not in state:
                state["entities"] = []
            if "references" not in state:
                state["references"] = []
            
            # Add extracted constraint entities
            for constraint in extracted_constraints:
                if constraint.get("entity"):
                    state["entities"].append(constraint["entity"])
            
            # Add resolved references
            references = self._extract_references(extracted_constraints)
            state["references"].extend(references)
        
        return state
    
    def _accumulate_constraints(self, state: AgentState) -> AgentState:
        """Accumulate constraints with session history."""
        new_constraints = state.get("extracted_constraints", [])
        
        if new_constraints:
            state = self.constraint_accumulator.accumulate_constraints(state, new_constraints)
            
            # Clean up temporary field
            if "extracted_constraints" in state:
                del state["extracted_constraints"]
        
        return state
    
    def _build_extraction_context(self, conversation_history: List[Dict], mentioned_dishes: List[str]) -> str:
        """Build context string for constraint extraction."""
        context_parts = []
        
        # Add mentioned dishes
        if mentioned_dishes:
            context_parts.append(f"Các món đã được đề cập: {', '.join(mentioned_dishes)}")
        
        # Add recent conversation (last 3 turns)
        if conversation_history:
            recent_turns = conversation_history[-3:]
            context_parts.append("Cuộc hội thoại gần đây:")
            for i, turn in enumerate(recent_turns, 1):
                context_parts.append(f"{i}. Khách: {turn.get('corrected_input', turn.get('user_input', ''))}")
                if turn.get('system_response'):
                    context_parts.append(f"   Hệ thống: {turn['system_response']}")
        
        return "\n".join(context_parts)
    
    def _extract_references(self, constraints: List[Dict[str, Any]]) -> List[str]:
        """Extract reference strings from constraints."""
        references = []
        for constraint in constraints:
            if constraint.get("type") == "REFERENCE":
                if constraint.get("resolved_entity"):
                    references.append(constraint["resolved_entity"])
                else:
                    references.append(constraint["value"])
        
        return references


def create_context_constraint_node():
    """
    Create a LangGraph node for combined context and constraint processing.
    
    Returns:
        Function that can be used as a LangGraph node
    """
    processor = ContextConstraintNode()
    
    def context_constraint_node(state: AgentState) -> AgentState:
        """
        LangGraph node function for context and constraint processing.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with context and constraints processed
        """
        return processor.process(state)
    
    return context_constraint_node