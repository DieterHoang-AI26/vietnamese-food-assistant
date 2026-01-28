"""
Vietnamese Food Assistant - Constraint Accumulation Logic

This module implements Python logic for merging new extracted constraints
with session history, handling constraint conflicts and priority resolution.

Requirements: 7.2 - Session Management and Data Persistence
"""

from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from src.state import AgentState, Constraint
from src.config import get_config


class ConstraintAccumulator:
    """
    Manages constraint accumulation and merging logic for session persistence.
    
    Responsibilities:
    - Merge new extracted constraints with session history
    - Handle constraint conflicts and priority resolution
    - Maintain constraint consistency across session
    - Apply constraint decay and temporal relevance
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Constraint priority mapping (higher = more important)
        self.constraint_priorities = {
            "ALLERGY": 100,      # Highest priority - safety critical
            "DIETARY": 80,       # High priority - religious/ethical
            "DISLIKE": 60,       # Medium priority - strong preference
            "PREFERENCE": 40,    # Lower priority - mild preference
            "ACTION": 20,        # Lowest priority - temporary action
            "REFERENCE": 10      # Informational only
        }
        
        # Severity multipliers
        self.severity_multipliers = {
            "STRICT": 1.0,
            "MODERATE": 0.8,
            "MILD": 0.6
        }
    
    def accumulate_constraints(self, state: AgentState, new_constraints: List[Dict[str, Any]]) -> AgentState:
        """
        Merge new constraints with existing session constraints.
        
        Args:
            state: Current agent state
            new_constraints: Newly extracted constraints from current input
            
        Returns:
            Updated state with accumulated constraints
        """
        # Initialize active constraints if not exists
        if "active_constraints" not in state or state["active_constraints"] is None:
            state["active_constraints"] = []
        
        # Convert new constraints to Constraint objects
        new_constraint_objects = [self._dict_to_constraint(c) for c in new_constraints]
        
        # Filter out non-persistent constraint types
        persistent_constraints = [c for c in new_constraint_objects 
                                if c["type"] in ["ALLERGY", "DIETARY", "DISLIKE", "PREFERENCE"]]
        
        # Merge with existing constraints
        merged_constraints = self._merge_constraints(
            existing=state["active_constraints"],
            new=persistent_constraints
        )
        
        # Apply constraint decay (reduce relevance of old constraints)
        decayed_constraints = self._apply_constraint_decay(merged_constraints)
        
        # Resolve conflicts
        resolved_constraints = self._resolve_constraint_conflicts(decayed_constraints)
        
        # Update state
        state["active_constraints"] = resolved_constraints
        
        # Update session preferences based on constraints
        state = self._update_session_preferences(state, resolved_constraints)
        
        return state
    
    def _dict_to_constraint(self, constraint_dict: Dict[str, Any]) -> Constraint:
        """
        Convert constraint dictionary to Constraint TypedDict.
        
        Args:
            constraint_dict: Dictionary representation of constraint
            
        Returns:
            Constraint object
        """
        return Constraint(
            type=constraint_dict.get("type", "PREFERENCE"),
            value=constraint_dict.get("value", ""),
            severity=constraint_dict.get("severity", "MILD"),
            source_text=constraint_dict.get("source_text", "")
        )
    
    def _merge_constraints(self, existing: List[Constraint], new: List[Constraint]) -> List[Constraint]:
        """
        Merge new constraints with existing ones, handling duplicates and updates.
        
        Args:
            existing: List of existing constraints
            new: List of new constraints to merge
            
        Returns:
            Merged list of constraints
        """
        # Create a mapping of existing constraints by (type, value)
        existing_map = {}
        for constraint in existing:
            key = (constraint["type"], constraint["value"].lower())
            existing_map[key] = constraint
        
        # Process new constraints
        for new_constraint in new:
            key = (new_constraint["type"], new_constraint["value"].lower())
            
            if key in existing_map:
                # Update existing constraint if new one has higher severity
                existing_constraint = existing_map[key]
                if self._get_constraint_priority(new_constraint) > self._get_constraint_priority(existing_constraint):
                    existing_map[key] = new_constraint
            else:
                # Add new constraint
                existing_map[key] = new_constraint
        
        return list(existing_map.values())
    
    def _apply_constraint_decay(self, constraints: List[Constraint]) -> List[Constraint]:
        """
        Apply temporal decay to constraints based on when they were mentioned.
        
        Args:
            constraints: List of constraints to apply decay to
            
        Returns:
            List of constraints with decay applied
        """
        # For now, we don't have timestamp info in constraints
        # In a full implementation, we would reduce the effective priority
        # of constraints that haven't been mentioned recently
        
        # Simple implementation: keep all constraints as-is
        # Future enhancement: add timestamp tracking and decay logic
        return constraints
    
    def _resolve_constraint_conflicts(self, constraints: List[Constraint]) -> List[Constraint]:
        """
        Resolve conflicts between constraints using priority rules.
        
        Args:
            constraints: List of potentially conflicting constraints
            
        Returns:
            List of constraints with conflicts resolved
        """
        # Group constraints by value to detect conflicts
        value_groups = {}
        for constraint in constraints:
            value_key = constraint["value"].lower()
            if value_key not in value_groups:
                value_groups[value_key] = []
            value_groups[value_key].append(constraint)
        
        resolved_constraints = []
        
        for value_key, group in value_groups.items():
            if len(group) == 1:
                # No conflict
                resolved_constraints.append(group[0])
            else:
                # Resolve conflict by priority
                resolved_constraint = self._resolve_constraint_group_conflict(group)
                if resolved_constraint:
                    resolved_constraints.append(resolved_constraint)
        
        return resolved_constraints
    
    def _resolve_constraint_group_conflict(self, constraint_group: List[Constraint]) -> Optional[Constraint]:
        """
        Resolve conflicts within a group of constraints for the same value.
        
        Args:
            constraint_group: List of constraints with the same value
            
        Returns:
            Winning constraint or None if all should be discarded
        """
        # Check for direct conflicts (e.g., PREFERENCE vs DISLIKE for same item)
        types_in_group = set(c["type"] for c in constraint_group)
        
        # Handle PREFERENCE vs DISLIKE conflict
        if "PREFERENCE" in types_in_group and "DISLIKE" in types_in_group:
            # DISLIKE wins over PREFERENCE
            dislike_constraints = [c for c in constraint_group if c["type"] == "DISLIKE"]
            return max(dislike_constraints, key=self._get_constraint_priority)
        
        # For other conflicts, highest priority wins
        return max(constraint_group, key=self._get_constraint_priority)
    
    def _get_constraint_priority(self, constraint: Constraint) -> float:
        """
        Calculate the effective priority of a constraint.
        
        Args:
            constraint: Constraint to calculate priority for
            
        Returns:
            Numerical priority value (higher = more important)
        """
        base_priority = self.constraint_priorities.get(constraint["type"], 0)
        severity_multiplier = self.severity_multipliers.get(constraint["severity"], 0.5)
        
        return base_priority * severity_multiplier
    
    def _update_session_preferences(self, state: AgentState, constraints: List[Constraint]) -> AgentState:
        """
        Update session preferences based on accumulated constraints.
        
        Args:
            state: Current agent state
            constraints: List of accumulated constraints
            
        Returns:
            Updated state with preference scores
        """
        if "session_preferences" not in state or state["session_preferences"] is None:
            state["session_preferences"] = {}
        
        # Calculate preference scores from constraints
        for constraint in constraints:
            value = constraint["value"].lower()
            constraint_type = constraint["type"]
            severity = constraint["severity"]
            
            # Calculate preference score
            if constraint_type == "ALLERGY":
                # Allergies get maximum negative score
                state["session_preferences"][value] = -1.0
            elif constraint_type == "DIETARY":
                # Dietary restrictions get high negative score
                state["session_preferences"][value] = -0.8
            elif constraint_type == "DISLIKE":
                # Dislikes get moderate negative score
                severity_factor = {"STRICT": -0.8, "MODERATE": -0.6, "MILD": -0.4}
                state["session_preferences"][value] = severity_factor.get(severity, -0.5)
            elif constraint_type == "PREFERENCE":
                # Preferences get positive score
                severity_factor = {"STRICT": 0.8, "MODERATE": 0.6, "MILD": 0.4}
                state["session_preferences"][value] = severity_factor.get(severity, 0.5)
        
        return state
    
    def get_active_constraints_by_type(self, state: AgentState, constraint_type: str) -> List[Constraint]:
        """
        Get active constraints of a specific type.
        
        Args:
            state: Current agent state
            constraint_type: Type of constraints to retrieve
            
        Returns:
            List of constraints of the specified type
        """
        if "active_constraints" not in state or state["active_constraints"] is None:
            return []
        
        return [c for c in state["active_constraints"] if c["type"] == constraint_type]
    
    def get_strict_constraints(self, state: AgentState) -> List[Constraint]:
        """
        Get all strict constraints (allergies and strict dietary restrictions).
        
        Args:
            state: Current agent state
            
        Returns:
            List of strict constraints that must be enforced
        """
        if "active_constraints" not in state or state["active_constraints"] is None:
            return []
        
        strict_types = ["ALLERGY"]
        strict_severity_constraints = [
            c for c in state["active_constraints"] 
            if c["severity"] == "STRICT" and c["type"] in ["DIETARY", "DISLIKE"]
        ]
        
        allergy_constraints = [
            c for c in state["active_constraints"] 
            if c["type"] in strict_types
        ]
        
        return allergy_constraints + strict_severity_constraints
    
    def check_constraint_violation(self, state: AgentState, item_ingredients: List[str], 
                                 item_name: str = "") -> List[str]:
        """
        Check if an item violates any active constraints.
        
        Args:
            state: Current agent state
            item_ingredients: List of ingredients in the item
            item_name: Name of the item (optional)
            
        Returns:
            List of violation messages
        """
        violations = []
        strict_constraints = self.get_strict_constraints(state)
        
        # Check ingredient-based violations
        for constraint in strict_constraints:
            constraint_value = constraint["value"].lower()
            
            # Check against ingredients
            for ingredient in item_ingredients:
                if constraint_value in ingredient.lower():
                    violations.append(
                        f"Món này chứa {ingredient} - vi phạm ràng buộc {constraint['type'].lower()}: {constraint['value']}"
                    )
            
            # Check against item name
            if item_name and constraint_value in item_name.lower():
                violations.append(
                    f"Món {item_name} - vi phạm ràng buộc {constraint['type'].lower()}: {constraint['value']}"
                )
        
        return violations


def create_constraint_accumulation_node():
    """
    Create a LangGraph node for constraint accumulation.
    
    Returns:
        Function that can be used as a LangGraph node
    """
    accumulator = ConstraintAccumulator()
    
    def constraint_accumulation_node(state: AgentState) -> AgentState:
        """
        LangGraph node function for constraint accumulation.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with accumulated constraints
        """
        try:
            # Get newly extracted constraints from previous node
            # This would typically come from the constraint extraction node
            # For now, we'll look for them in a temporary field
            new_constraints = state.get("extracted_constraints", [])
            
            if new_constraints:
                state = accumulator.accumulate_constraints(state, new_constraints)
                
                # Clean up temporary field
                if "extracted_constraints" in state:
                    del state["extracted_constraints"]
            
        except Exception as e:
            # Handle errors gracefully
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"Constraint accumulation failed: {str(e)}")
        
        return state
    
    return constraint_accumulation_node