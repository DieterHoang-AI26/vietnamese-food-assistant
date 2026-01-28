"""
Vietnamese Food Assistant - Context Manager Node

This module implements the Context Manager component responsible for managing
conversation history and context throughout the session. It handles appending
and retrieving chat history from the AgentState, with support for session
recovery and isolation.

Requirements: 2.4 - Context Management and Reference Resolution
Requirements: 7.5 - Session Management and Data Persistence
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import threading
from src.state import AgentState, ConversationTurn
from src.config import get_config


class ContextManager:
    """
    Manages conversation context and history for the Vietnamese Food Assistant.
    
    Responsibilities:
    - Append new conversation turns to history
    - Retrieve conversation history from state
    - Maintain session context within configured limits
    - Clean up expired or excessive history entries
    - Handle session recovery and isolation
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Session isolation tracking
        self.active_sessions = set()
        self.session_locks = {}
    
    def update_context(self, state: AgentState, user_input: str, corrected_input: str, 
                      system_response: str = "", mentioned_entities: List[str] = None) -> AgentState:
        """
        Update the conversation context with a new turn.
        
        Args:
            state: Current agent state
            user_input: Original user input
            corrected_input: Corrected version of user input
            system_response: System's response (if available)
            mentioned_entities: List of entities mentioned in this turn
            
        Returns:
            Updated agent state with new conversation turn
        """
        if mentioned_entities is None:
            mentioned_entities = []
        
        # Create new conversation turn
        new_turn = ConversationTurn(
            timestamp=datetime.now(),
            user_input=user_input,
            corrected_input=corrected_input,
            system_response=system_response,
            mentioned_entities=mentioned_entities
        )
        
        # Initialize conversation history if not exists
        if "conversation_history" not in state or state["conversation_history"] is None:
            state["conversation_history"] = []
        
        # Add new turn to history
        state["conversation_history"].append(new_turn)
        
        # Maintain history size limit
        max_history = self.config.max_conversation_history
        if len(state["conversation_history"]) > max_history:
            state["conversation_history"] = state["conversation_history"][-max_history:]
        
        # Update mentioned dishes from entities
        self._update_mentioned_dishes(state, mentioned_entities)
        
        return state
    
    def get_conversation_history(self, state: AgentState, limit: Optional[int] = None) -> List[ConversationTurn]:
        """
        Retrieve conversation history from state.
        
        Args:
            state: Current agent state
            limit: Maximum number of turns to return (most recent first)
            
        Returns:
            List of conversation turns, most recent first
        """
        if "conversation_history" not in state or state["conversation_history"] is None:
            return []
        
        history = state["conversation_history"]
        
        if limit is not None and limit > 0:
            history = history[-limit:]
        
        # Return in reverse order (most recent first)
        return list(reversed(history))
    
    def get_recent_context(self, state: AgentState, minutes: int = 10) -> List[ConversationTurn]:
        """
        Get conversation turns from the last N minutes.
        
        Args:
            state: Current agent state
            minutes: Number of minutes to look back
            
        Returns:
            List of recent conversation turns
        """
        if "conversation_history" not in state or state["conversation_history"] is None:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_turns = []
        
        for turn in reversed(state["conversation_history"]):
            if turn["timestamp"] >= cutoff_time:
                recent_turns.append(turn)
            else:
                break
        
        return recent_turns
    
    def get_mentioned_dishes(self, state: AgentState) -> List[str]:
        """
        Get list of dishes mentioned in the current session.
        
        Args:
            state: Current agent state
            
        Returns:
            List of mentioned dish names
        """
        if "mentioned_dishes" not in state or state["mentioned_dishes"] is None:
            return []
        
        return state["mentioned_dishes"]
    
    def add_mentioned_dish(self, state: AgentState, dish_name: str) -> AgentState:
        """
        Add a dish to the mentioned dishes list.
        
        Args:
            state: Current agent state
            dish_name: Name of the dish to add
            
        Returns:
            Updated agent state
        """
        if "mentioned_dishes" not in state or state["mentioned_dishes"] is None:
            state["mentioned_dishes"] = []
        
        # Avoid duplicates
        if dish_name not in state["mentioned_dishes"]:
            state["mentioned_dishes"].append(dish_name)
        
        return state
    
    def initialize_session(self, session_id: str, recovered_state: Optional[AgentState] = None) -> AgentState:
        """
        Initialize a new session or recover an existing one.
        
        Args:
            session_id: Unique session identifier
            recovered_state: Previously saved session state (if any)
            
        Returns:
            Initialized session state
        """
        with self._get_session_lock(session_id):
            self.logger.info(f"Initializing session: {session_id}")
            
            if recovered_state:
                self.logger.info(f"Recovering session {session_id} from saved state")
                state = self._validate_recovered_state(recovered_state)
            else:
                self.logger.info(f"Creating new session {session_id}")
                state = self._create_new_session_state(session_id)
            
            # Mark session as active
            self.active_sessions.add(session_id)
            
            return state
    
    def finalize_session(self, session_id: str, state: AgentState) -> AgentState:
        """
        Finalize session and prepare for cleanup.
        
        Args:
            session_id: Session identifier
            state: Current session state
            
        Returns:
            Finalized state
        """
        with self._get_session_lock(session_id):
            self.logger.info(f"Finalizing session: {session_id}")
            
            # Mark session as inactive
            self.active_sessions.discard(session_id)
            
            # Add finalization timestamp
            state["session_finalized_at"] = datetime.now()
            
            return state
    
    def recover_session_gracefully(self, session_id: str, partial_state: AgentState) -> AgentState:
        """
        Gracefully recover a session after system restart or failure.
        
        Implements Requirement 7.5: Handle session recovery gracefully without
        data corruption or cross-session contamination.
        
        Args:
            session_id: Session to recover
            partial_state: Partially recovered state
            
        Returns:
            Fully recovered and validated state
        """
        with self._get_session_lock(session_id):
            self.logger.info(f"Gracefully recovering session: {session_id}")
            
            try:
                # Validate session isolation
                if not self._validate_session_isolation(session_id, partial_state):
                    self.logger.warning(f"Session isolation validation failed for {session_id}")
                    return self._create_new_session_state(session_id)
                
                # Validate conversation history integrity
                if not self._validate_conversation_integrity(partial_state):
                    self.logger.warning(f"Conversation integrity validation failed for {session_id}")
                    partial_state = self._repair_conversation_history(partial_state)
                
                # Validate constraints consistency
                if not self._validate_constraints_consistency(partial_state):
                    self.logger.warning(f"Constraints consistency validation failed for {session_id}")
                    partial_state = self._repair_constraints(partial_state)
                
                # Ensure session metadata is correct
                partial_state["session_id"] = session_id
                partial_state["session_recovered_at"] = datetime.now()
                partial_state["recovery_validation_passed"] = True
                
                # Mark as active
                self.active_sessions.add(session_id)
                
                self.logger.info(f"Session {session_id} recovered successfully")
                return partial_state
                
            except Exception as e:
                self.logger.error(f"Failed to recover session {session_id}: {e}")
                # Fall back to new session
                return self._create_new_session_state(session_id)
    def clear_session_context(self, state: AgentState) -> AgentState:
        """
        Clear session-specific context while preserving session ID.
        
        Args:
            state: Current agent state
            
        Returns:
            State with cleared context
        """
        session_id = state.get("session_id", "unknown")
        
        with self._get_session_lock(session_id):
            state["conversation_history"] = []
            state["mentioned_dishes"] = []
            state["active_constraints"] = []
            state["session_preferences"] = {}
            state["context_cleared_at"] = datetime.now()
            
            return state
    
    def is_session_expired(self, state: AgentState) -> bool:
        """
        Check if the current session has expired based on last activity.
        
        Args:
            state: Current agent state
            
        Returns:
            True if session has expired, False otherwise
        """
        if "conversation_history" not in state or not state["conversation_history"]:
            return False
        
        last_turn = state["conversation_history"][-1]
        session_timeout = timedelta(minutes=self.config.session_timeout_minutes)
        
        return datetime.now() - last_turn["timestamp"] > session_timeout
    
    def _update_mentioned_dishes(self, state: AgentState, mentioned_entities: List[str]) -> None:
        """
        Update mentioned dishes list from entities in the current turn.
        
        Args:
            state: Current agent state
            mentioned_entities: List of entities mentioned in current turn
        """
        if "mentioned_dishes" not in state or state["mentioned_dishes"] is None:
            state["mentioned_dishes"] = []
        
        # Filter entities that are likely dish names
        # This is a simple heuristic - in practice, this would use NER
        for entity in mentioned_entities:
            if self._is_likely_dish_name(entity):
                if entity not in state["mentioned_dishes"]:
                    state["mentioned_dishes"].append(entity)
    
    def _is_likely_dish_name(self, entity: str) -> bool:
        """
        Simple heuristic to determine if an entity is likely a dish name.
        
        Args:
            entity: Entity string to check
            
        Returns:
            True if entity is likely a dish name
        """
        # Simple heuristics for Vietnamese dish names
        dish_indicators = [
            "phở", "bún", "bánh", "cơm", "chả", "nem", "gỏi", "canh",
            "soup", "rice", "noodle", "spring roll"
        ]
        
        entity_lower = entity.lower()
        return any(indicator in entity_lower for indicator in dish_indicators)
    
    def _get_session_lock(self, session_id: str) -> threading.Lock:
        """
        Get or create a lock for session isolation.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Thread lock for the session
        """
        if session_id not in self.session_locks:
            self.session_locks[session_id] = threading.Lock()
        return self.session_locks[session_id]
    
    def _create_new_session_state(self, session_id: str) -> AgentState:
        """
        Create a new clean session state.
        
        Args:
            session_id: Session identifier
            
        Returns:
            New session state
        """
        return {
            "session_id": session_id,
            "conversation_history": [],
            "mentioned_dishes": [],
            "active_constraints": [],
            "session_preferences": {},
            "session_created_at": datetime.now(),
            "raw_input": "",
            "corrected_input": "",
            "processed_input": "",
            "retrieved_docs": [],
            "filtered_docs": [],
            "reranked_docs": [],
            "intent": "",
            "entities": [],
            "references": [],
            "response_text": "",
            "clarification_needed": False,
            "follow_up_questions": [],
            "availability_warnings": [],
            "errors": [],
            "warnings": []
        }
    
    def _validate_recovered_state(self, state: AgentState) -> AgentState:
        """
        Validate and clean up recovered session state.
        
        Args:
            state: Recovered state
            
        Returns:
            Validated and cleaned state
        """
        # Ensure required fields exist
        required_fields = [
            "session_id", "conversation_history", "mentioned_dishes",
            "active_constraints", "session_preferences"
        ]
        
        for field in required_fields:
            if field not in state or state[field] is None:
                if field in ["conversation_history", "mentioned_dishes", "active_constraints"]:
                    state[field] = []
                elif field == "session_preferences":
                    state[field] = {}
                else:
                    state[field] = ""
        
        # Validate conversation history format
        if isinstance(state["conversation_history"], list):
            validated_history = []
            for turn in state["conversation_history"]:
                if isinstance(turn, dict) and "timestamp" in turn:
                    validated_history.append(turn)
            state["conversation_history"] = validated_history
        
        return state
    
    def _validate_session_isolation(self, session_id: str, state: AgentState) -> bool:
        """
        Validate that session state belongs to the correct session.
        
        Args:
            session_id: Expected session ID
            state: State to validate
            
        Returns:
            True if session isolation is maintained
        """
        state_session_id = state.get("session_id")
        
        if state_session_id != session_id:
            self.logger.warning(f"Session ID mismatch: expected {session_id}, got {state_session_id}")
            return False
        
        return True
    
    def _validate_conversation_integrity(self, state: AgentState) -> bool:
        """
        Validate conversation history integrity.
        
        Args:
            state: State to validate
            
        Returns:
            True if conversation history is valid
        """
        history = state.get("conversation_history", [])
        
        if not isinstance(history, list):
            return False
        
        # Check each turn has required fields
        for turn in history:
            if not isinstance(turn, dict):
                return False
            
            required_turn_fields = ["timestamp", "user_input", "corrected_input"]
            if not all(field in turn for field in required_turn_fields):
                return False
            
            # Validate timestamp
            timestamp = turn.get("timestamp")
            if isinstance(timestamp, str):
                try:
                    datetime.fromisoformat(timestamp)
                except ValueError:
                    return False
            elif not isinstance(timestamp, datetime):
                return False
        
        return True
    
    def _validate_constraints_consistency(self, state: AgentState) -> bool:
        """
        Validate constraints consistency.
        
        Args:
            state: State to validate
            
        Returns:
            True if constraints are consistent
        """
        constraints = state.get("active_constraints", [])
        
        if not isinstance(constraints, list):
            return False
        
        # Check each constraint has required structure
        for constraint in constraints:
            if not isinstance(constraint, dict):
                return False
            
            required_fields = ["type", "value"]
            if not all(field in constraint for field in required_fields):
                return False
        
        return True
    
    def _repair_conversation_history(self, state: AgentState) -> AgentState:
        """
        Repair corrupted conversation history.
        
        Args:
            state: State with potentially corrupted history
            
        Returns:
            State with repaired history
        """
        self.logger.info("Repairing conversation history")
        
        history = state.get("conversation_history", [])
        repaired_history = []
        
        for turn in history:
            if isinstance(turn, dict):
                # Repair missing fields
                repaired_turn = {
                    "timestamp": turn.get("timestamp", datetime.now()),
                    "user_input": turn.get("user_input", ""),
                    "corrected_input": turn.get("corrected_input", turn.get("user_input", "")),
                    "system_response": turn.get("system_response", ""),
                    "mentioned_entities": turn.get("mentioned_entities", [])
                }
                
                # Fix timestamp format
                if isinstance(repaired_turn["timestamp"], str):
                    try:
                        repaired_turn["timestamp"] = datetime.fromisoformat(repaired_turn["timestamp"])
                    except ValueError:
                        repaired_turn["timestamp"] = datetime.now()
                
                repaired_history.append(repaired_turn)
        
        state["conversation_history"] = repaired_history
        state["history_repaired"] = True
        
        return state
    
    def _repair_constraints(self, state: AgentState) -> AgentState:
        """
        Repair corrupted constraints.
        
        Args:
            state: State with potentially corrupted constraints
            
        Returns:
            State with repaired constraints
        """
        self.logger.info("Repairing constraints")
        
        constraints = state.get("active_constraints", [])
        repaired_constraints = []
        
        for constraint in constraints:
            if isinstance(constraint, dict) and "type" in constraint and "value" in constraint:
                repaired_constraint = {
                    "type": constraint["type"],
                    "value": constraint["value"],
                    "severity": constraint.get("severity", "MODERATE"),
                    "source_text": constraint.get("source_text", "")
                }
                repaired_constraints.append(repaired_constraint)
        
        state["active_constraints"] = repaired_constraints
        state["constraints_repaired"] = True
        
        return state


def create_context_node():
    """
    Create a LangGraph node for context management.
    
    Returns:
        Function that can be used as a LangGraph node
    """
    context_manager = ContextManager()
    
    def context_node(state: AgentState) -> AgentState:
        """
        LangGraph node function for context management.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with context management applied
        """
        session_id = state.get("session_id", "default")
        
        # Handle session recovery if needed
        if state.get("session_recovered_at") or state.get("recovery_validation_passed"):
            # Session was already recovered, just validate
            state = context_manager._validate_recovered_state(state)
        elif not state.get("session_created_at") and not state.get("conversation_history"):
            # Initialize new session
            state = context_manager.initialize_session(session_id)
        
        # Update context with current input if available
        if "raw_input" in state and state["raw_input"]:
            corrected_input = state.get("corrected_input", state["raw_input"])
            entities = state.get("entities", [])
            
            state = context_manager.update_context(
                state=state,
                user_input=state["raw_input"],
                corrected_input=corrected_input,
                mentioned_entities=entities
            )
        
        # Check for session expiration
        if context_manager.is_session_expired(state):
            state["warnings"] = state.get("warnings", [])
            state["warnings"].append("Session expired - context may be limited")
        
        return state
    
    return context_node