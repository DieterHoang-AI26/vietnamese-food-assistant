"""
Vietnamese Food Assistant - Constraint Extraction Node

This module implements constraint extraction using LLM prompts to identify
dietary constraints, preferences, and handle reference resolution from user input.

Requirements: 2.1, 3.1 - Reference Resolution and Constraint Management
"""

import json
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from src.state import AgentState, Constraint
from src.config import get_config


class ConstraintExtractor:
    """
    Extracts dietary constraints and preferences from user input using LLM prompts.
    
    Responsibilities:
    - Extract JSON constraints from current input
    - Handle reference resolution ("món đó", "món này")
    - Identify dietary restrictions, allergies, and preferences
    - Resolve references to previously mentioned dishes
    """
    
    def __init__(self):
        self.config = get_config()
        model_config = self.config.get_current_model_config()
        
        self.llm = ChatOllama(
            base_url=model_config.base_url,
            model=model_config.model_id,
            temperature=model_config.temperature,
            num_predict=model_config.max_tokens
        )
    
    def extract_constraints(self, state: AgentState) -> AgentState:
        """
        Extract constraints from the current user input.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with extracted constraints
        """
        user_input = state.get("corrected_input", state.get("raw_input", ""))
        conversation_history = state.get("conversation_history", [])
        mentioned_dishes = state.get("mentioned_dishes", [])
        
        # Create context for reference resolution
        context = self._build_context(conversation_history, mentioned_dishes)
        
        # Extract constraints using LLM
        extracted_constraints = self._prompt_constraint_extraction(user_input, context)
        
        # Update state with extracted constraints
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
        
        # Store extracted constraints for the accumulator node
        state["extracted_constraints"] = extracted_constraints
        
        return state
    
    def _build_context(self, conversation_history: List[Dict], mentioned_dishes: List[str]) -> str:
        """
        Build context string for reference resolution.
        
        Args:
            conversation_history: Recent conversation turns
            mentioned_dishes: List of previously mentioned dishes
            
        Returns:
            Context string for the LLM prompt
        """
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
    
    def _prompt_constraint_extraction(self, user_input: str, context: str) -> List[Dict[str, Any]]:
        """
        Use LLM to extract constraints from user input with context.
        
        Args:
            user_input: Current user input
            context: Conversation context for reference resolution
            
        Returns:
            List of extracted constraints as dictionaries
        """
        prompt = self._create_constraint_extraction_prompt(user_input, context)
        
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_constraint_response(response.content)
            return result
        except Exception as e:
            # Fallback to empty list if LLM fails
            return []
    
    def _create_constraint_extraction_prompt(self, user_input: str, context: str) -> str:
        """
        Create the constraint extraction prompt for the LLM.
        
        Args:
            user_input: Current user input
            context: Conversation context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Bạn là một trợ lý thông minh chuyên phân tích yêu cầu về món ăn. Nhiệm vụ của bạn là trích xuất các ràng buộc và sở thích từ câu nói của khách hàng.

NGỮ CẢNH CUỘC HỘI THOẠI:
{context if context else "Chưa có ngữ cảnh trước đó"}

CÂU NÓI CỦA KHÁCH HÀNG:
"{user_input}"

HƯỚNG DẪN PHÂN TÍCH:

1. RÀNG BUỘC DINH DƯỠNG (DIETARY CONSTRAINTS):
   - Dị ứng: "tôi dị ứng với...", "không được ăn...", "bị dị ứng..."
   - Kiêng: "tôi kiêng...", "không ăn được...", "không thích..."
   - Chế độ ăn: "tôi ăn chay", "tôi ăn kiêng", "không ăn thịt"

2. SỞ THÍCH (PREFERENCES):
   - Thích: "tôi thích...", "tôi thèm...", "muốn ăn..."
   - Không thích: "tôi ngán...", "không muốn...", "chán..."
   - Mức độ cay: "cay", "không cay", "ít cay", "cay nhiều"

3. GIẢI QUYẾT THAM CHIẾU (REFERENCE RESOLUTION):
   - "món này", "món đó" → tham chiếu đến món gần nhất được đề cập
   - "món thứ 1", "món đầu" → tham chiếu vị trí trong danh sách
   - "món vừa nói", "món kia" → tham chiếu ngữ cảnh

4. HÀNH ĐỘNG (ACTIONS):
   - "lấy luôn", "đặt món này", "chọn món đó"

ĐỊNH DẠNG ĐẦU RA (JSON):
Trả về một mảng JSON với các đối tượng constraint, mỗi đối tượng có:
- "type": "ALLERGY" | "DIETARY" | "PREFERENCE" | "DISLIKE" | "ACTION" | "REFERENCE"
- "value": giá trị cụ thể (tên món, thành phần, hành động)
- "severity": "STRICT" | "MODERATE" | "MILD"
- "source_text": đoạn text gốc chứa constraint này
- "entity": tên thực thể được đề cập (nếu có)
- "reference_type": loại tham chiếu (nếu là REFERENCE)
- "resolved_entity": thực thể được giải quyết (nếu có thể xác định từ ngữ cảnh)

VÍ DỤ:
Input: "Tôi dị ứng tôm, không thích món cay. Lấy luôn món phở đó."
Output:
[
  {{
    "type": "ALLERGY",
    "value": "tôm",
    "severity": "STRICT",
    "source_text": "dị ứng tôm",
    "entity": "tôm"
  }},
  {{
    "type": "DISLIKE",
    "value": "cay",
    "severity": "MODERATE", 
    "source_text": "không thích món cay",
    "entity": "cay"
  }},
  {{
    "type": "REFERENCE",
    "value": "món phở đó",
    "severity": "MILD",
    "source_text": "món phở đó",
    "reference_type": "demonstrative",
    "resolved_entity": "phở"
  }},
  {{
    "type": "ACTION",
    "value": "lấy luôn",
    "severity": "MILD",
    "source_text": "Lấy luôn món phở đó",
    "entity": "phở"
  }}
]

CHỈ TRẢ VỀ JSON, KHÔNG CÓ TEXT GIẢI THÍCH THÊM:"""

        return prompt
    
    def _parse_constraint_response(self, response_content: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response to extract constraint information.
        
        Args:
            response_content: Raw response from LLM
            
        Returns:
            List of parsed constraints
        """
        try:
            # Try to extract JSON from response
            response_content = response_content.strip()
            
            # Handle cases where LLM adds extra text
            if response_content.startswith('['):
                json_end = response_content.rfind(']') + 1
                json_str = response_content[:json_end]
            else:
                # Look for JSON array in the response
                start_idx = response_content.find('[')
                end_idx = response_content.rfind(']') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_content[start_idx:end_idx]
                else:
                    return []
            
            constraints = json.loads(json_str)
            
            # Validate and clean up constraints
            validated_constraints = []
            for constraint in constraints:
                if self._validate_constraint(constraint):
                    validated_constraints.append(constraint)
            
            return validated_constraints
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Return empty list if parsing fails
            return []
    
    def _validate_constraint(self, constraint: Dict[str, Any]) -> bool:
        """
        Validate a constraint object has required fields.
        
        Args:
            constraint: Constraint dictionary to validate
            
        Returns:
            True if constraint is valid
        """
        required_fields = ["type", "value", "severity", "source_text"]
        valid_types = ["ALLERGY", "DIETARY", "PREFERENCE", "DISLIKE", "ACTION", "REFERENCE"]
        valid_severities = ["STRICT", "MODERATE", "MILD"]
        
        # Check required fields
        for field in required_fields:
            if field not in constraint:
                return False
        
        # Check valid values
        if constraint["type"] not in valid_types:
            return False
        
        if constraint["severity"] not in valid_severities:
            return False
        
        return True
    
    def _extract_references(self, constraints: List[Dict[str, Any]]) -> List[str]:
        """
        Extract reference strings from constraints.
        
        Args:
            constraints: List of constraint dictionaries
            
        Returns:
            List of reference strings
        """
        references = []
        for constraint in constraints:
            if constraint.get("type") == "REFERENCE":
                if constraint.get("resolved_entity"):
                    references.append(constraint["resolved_entity"])
                else:
                    references.append(constraint["value"])
        
        return references


def create_constraint_extraction_node():
    """
    Create a LangGraph node for constraint extraction.
    
    Returns:
        Function that can be used as a LangGraph node
    """
    extractor = ConstraintExtractor()
    
    def constraint_extraction_node(state: AgentState) -> AgentState:
        """
        LangGraph node function for constraint extraction.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with extracted constraints
        """
        try:
            state = extractor.extract_constraints(state)
        except Exception as e:
            # Handle errors gracefully
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"Constraint extraction failed: {str(e)}")
        
        return state
    
    return constraint_extraction_node