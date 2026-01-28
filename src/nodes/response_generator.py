"""
Vietnamese Food Assistant - Response Generator Node

This module implements the response generator node that creates friendly
responses based ONLY on provided context from retrieved and filtered documents.
"""

from typing import Dict, Any, List, Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser

from ..config import get_config
from ..state import AgentState


class ResponseOutputParser(BaseOutputParser[str]):
    """Parser for response generation output that extracts clean response text."""
    
    def parse(self, text: str) -> str:
        """Parse the LLM output to extract clean response text."""
        # Remove any metadata or formatting, keep only the response
        lines = text.strip().split('\n')
        
        # Look for the actual response (skip any system messages or metadata)
        response_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith('Response:'):
                response_lines.append(line)
        
        # Join lines and clean up
        response = '\n'.join(response_lines).strip()
        
        # Remove quotes if the entire response is wrapped in quotes
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        elif response.startswith("'") and response.endswith("'"):
            response = response[1:-1]
        
        return response


class ResponseGeneratorPrompts:
    """
    Collection of system prompts for response generation.
    
    These prompts are designed to generate friendly, helpful responses
    based ONLY on the provided context from retrieved menu items.
    """
    
    SYSTEM_PROMPT = """Bạn là trợ lý thông minh tư vấn món ăn Việt Nam thân thiện và chuyên nghiệp.

NHIỆM VỤ:
- Tạo câu trả lời thân thiện, hữu ích dựa TUYỆT ĐỐI CHỈ trên thông tin được cung cấp
- TUYỆT ĐỐI KHÔNG tự sáng tạo hoặc thêm thông tin không có trong context
- TUYỆT ĐỐI KHÔNG đề xuất món ăn không có trong danh sách được cung cấp
- Trả lời bằng tiếng Việt tự nhiên, thân thiện

QUY TẮC QUAN TRỌNG:
1. CHỈ sử dụng thông tin từ các món ăn được cung cấp trong context
2. KHÔNG tự sáng tạo tên món, giá cả, hoặc thông tin khác
3. KHÔNG đề xuất món ăn không có trong danh sách
4. Nếu không có món phù hợp, hãy thông báo lịch sự
5. Luôn đề cập tình trạng có sẵn (ngay lập tức / cần đặt trước)
6. Sử dụng giọng điệu thân thiện, tự nhiên như nhân viên phục vụ

ĐỊNH DẠNG PHẢN HỒI:
- Chào hỏi ngắn gọn (nếu phù hợp)
- Giới thiệu món ăn phù hợp từ danh sách
- Thông tin về tình trạng có sẵn
- Câu hỏi tiếp theo (nếu cần)

VÍ DỤ PHONG CÁCH:
- "Dạ, tôi có thể gợi ý cho bạn..."
- "Món này hiện có sẵn ngay..."
- "Món này cần đặt trước khoảng..."
- "Bạn có muốn biết thêm về..."
"""

    RECOMMENDATION_TEMPLATE = """Yêu cầu của khách: "{user_input}"

Các món ăn phù hợp từ menu:
{menu_context}

Ràng buộc và sở thích:
{constraints_context}

Tình trạng có sẵn:
{availability_context}

Hãy tạo câu trả lời thân thiện và hữu ích:"""

    NO_RESULTS_TEMPLATE = """Yêu cầu của khách: "{user_input}"

Ràng buộc và sở thích:
{constraints_context}

Không tìm thấy món ăn phù hợp trong menu hiện tại.

Hãy tạo câu trả lời lịch sự và gợi ý cách khác:"""

    CLARIFICATION_TEMPLATE = """Yêu cầu của khách: "{user_input}"

Các món ăn có thể phù hợp:
{menu_context}

Thông tin chưa rõ ràng hoặc cần làm rõ:
{clarification_needed}

Hãy tạo câu hỏi làm rõ thân thiện:"""

    @classmethod
    def get_recommendation_prompt(cls) -> PromptTemplate:
        """Get the main recommendation prompt template."""
        return PromptTemplate(
            input_variables=["user_input", "menu_context", "constraints_context", "availability_context"],
            template=f"{cls.SYSTEM_PROMPT}\n\n{cls.RECOMMENDATION_TEMPLATE}"
        )

    @classmethod
    def get_no_results_prompt(cls) -> PromptTemplate:
        """Get prompt for when no suitable dishes are found."""
        return PromptTemplate(
            input_variables=["user_input", "constraints_context"],
            template=f"{cls.SYSTEM_PROMPT}\n\n{cls.NO_RESULTS_TEMPLATE}"
        )

    @classmethod
    def get_clarification_prompt(cls) -> PromptTemplate:
        """Get prompt for requesting clarification."""
        return PromptTemplate(
            input_variables=["user_input", "menu_context", "clarification_needed"],
            template=f"{cls.SYSTEM_PROMPT}\n\n{cls.CLARIFICATION_TEMPLATE}"
        )

    @classmethod
    def get_availability_focused_prompt(cls) -> PromptTemplate:
        """Get prompt that emphasizes availability information."""
        availability_system = f"""{cls.SYSTEM_PROMPT}

ĐẶC BIỆT CHÚ Ý VỀ TÌNH TRẠNG CÓ SẴN:
- Ưu tiên đề cập món có sẵn ngay lập tức
- Thông báo rõ ràng món nào cần đặt trước
- Gợi ý thay thế nếu món yêu cầu không có sẵn ngay
- Hỏi khách về thời gian nếu cần thiết

VÍ DỤ:
- "Món này có sẵn ngay, bạn có muốn gọi không?"
- "Món này cần đặt trước 30 phút, hoặc bạn có thể thử..."
- "Hiện tại món này hết, nhưng tôi có thể gợi ý..."
"""

        return PromptTemplate(
            input_variables=["user_input", "menu_context", "constraints_context", "availability_context"],
            template=f"{availability_system}\n\n{cls.RECOMMENDATION_TEMPLATE}"
        )


def create_response_generator_node():
    """
    Create the response generator node function for LangGraph.
    
    This function generates friendly responses based ONLY on provided
    context from retrieved and filtered menu items.
    
    Returns:
        Callable that takes AgentState and returns updated AgentState
    """
    
    def response_generator_node(state: AgentState) -> Dict[str, Any]:
        """
        Response Generator Node - Creates friendly responses based on context.
        
        Args:
            state: Current agent state containing reranked documents
            
        Returns:
            Updated state with response_text and related fields
        """
        config = get_config()
        model_config = config.get_current_model_config()
        
        # Initialize LLM
        llm = OllamaLLM(
            model=model_config.model_id,
            base_url=model_config.base_url,
            temperature=model_config.temperature,
            num_predict=model_config.max_tokens,
            timeout=model_config.timeout
        )
        
        # Get input and context from state
        user_input = state.get("corrected_input", state.get("raw_input", ""))
        reranked_docs = state.get("reranked_docs", [])
        active_constraints = state.get("active_constraints", [])
        
        if not user_input or not user_input.strip():
            return {
                "response_text": "Xin lỗi, tôi không nhận được yêu cầu của bạn. Bạn có thể nói lại được không?",
                "clarification_needed": True,
                "follow_up_questions": ["Bạn muốn tìm món gì?"],
                "availability_warnings": []
            }
        
        try:
            # Determine response type based on available data
            if not reranked_docs:
                # No results found
                response_data = _generate_no_results_response(
                    llm, user_input, active_constraints
                )
            elif _needs_clarification(state):
                # Need clarification
                response_data = _generate_clarification_response(
                    llm, user_input, reranked_docs, state
                )
            else:
                # Generate recommendation
                response_data = _generate_recommendation_response(
                    llm, user_input, reranked_docs, active_constraints
                )
            
            # Update state with response data
            return {
                "response_text": response_data["response_text"],
                "clarification_needed": response_data.get("clarification_needed", False),
                "follow_up_questions": response_data.get("follow_up_questions", []),
                "availability_warnings": response_data.get("availability_warnings", [])
            }
                
        except Exception as e:
            # Fallback response on error
            return {
                "response_text": "Xin lỗi, tôi gặp sự cố khi xử lý yêu cầu của bạn. Bạn có thể thử lại được không?",
                "errors": state.get("errors", []) + [f"Response generation failed: {str(e)}"],
                "clarification_needed": True,
                "follow_up_questions": ["Bạn có thể nói lại yêu cầu được không?"],
                "availability_warnings": []
            }
    
    return response_generator_node


def _generate_recommendation_response(llm: OllamaLLM, user_input: str, 
                                    reranked_docs: List[Dict[str, Any]], 
                                    active_constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate recommendation response based on reranked documents."""
    
    # Build context strings
    menu_context = _build_menu_context(reranked_docs)
    constraints_context = _build_constraints_context(active_constraints)
    availability_context = _build_availability_context(reranked_docs)
    
    # Choose appropriate prompt based on availability concerns
    has_availability_issues = any(
        doc.get("requires_advance_order", False) or 
        doc.get("availability_status") != "available"
        for doc in reranked_docs
    )
    
    if has_availability_issues:
        prompt = ResponseGeneratorPrompts.get_availability_focused_prompt()
    else:
        prompt = ResponseGeneratorPrompts.get_recommendation_prompt()
    
    # Create chain with output parser
    chain = prompt | llm | ResponseOutputParser()
    
    # Generate response
    response_text = chain.invoke({
        "user_input": user_input,
        "menu_context": menu_context,
        "constraints_context": constraints_context,
        "availability_context": availability_context
    })
    
    # Extract availability warnings
    availability_warnings = _extract_availability_warnings(reranked_docs)
    
    # Generate follow-up questions
    follow_up_questions = _generate_follow_up_questions(reranked_docs, active_constraints)
    
    return {
        "response_text": response_text,
        "clarification_needed": False,
        "follow_up_questions": follow_up_questions,
        "availability_warnings": availability_warnings
    }


def _generate_no_results_response(llm: OllamaLLM, user_input: str, 
                                active_constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate response when no suitable dishes are found."""
    
    constraints_context = _build_constraints_context(active_constraints)
    
    prompt = ResponseGeneratorPrompts.get_no_results_prompt()
    chain = prompt | llm | ResponseOutputParser()
    
    response_text = chain.invoke({
        "user_input": user_input,
        "constraints_context": constraints_context
    })
    
    # Suggest ways to broaden search
    follow_up_questions = [
        "Bạn có muốn thử tìm món khác không?",
        "Bạn có thể bỏ bớt một số yêu cầu để có nhiều lựa chọn hơn không?"
    ]
    
    return {
        "response_text": response_text,
        "clarification_needed": True,
        "follow_up_questions": follow_up_questions,
        "availability_warnings": []
    }


def _generate_clarification_response(llm: OllamaLLM, user_input: str, 
                                   reranked_docs: List[Dict[str, Any]], 
                                   state: AgentState) -> Dict[str, Any]:
    """Generate response requesting clarification."""
    
    menu_context = _build_menu_context(reranked_docs)
    clarification_needed = _identify_clarification_needs(state)
    
    prompt = ResponseGeneratorPrompts.get_clarification_prompt()
    chain = prompt | llm | ResponseOutputParser()
    
    response_text = chain.invoke({
        "user_input": user_input,
        "menu_context": menu_context,
        "clarification_needed": clarification_needed
    })
    
    # Generate specific clarification questions
    follow_up_questions = _generate_clarification_questions(state)
    
    return {
        "response_text": response_text,
        "clarification_needed": True,
        "follow_up_questions": follow_up_questions,
        "availability_warnings": []
    }


def _build_menu_context(reranked_docs: List[Dict[str, Any]]) -> str:
    """Build formatted menu context from reranked documents."""
    if not reranked_docs:
        return "Không có món ăn phù hợp."
    
    context_parts = []
    for i, doc in enumerate(reranked_docs, 1):
        name_vi = doc.get("name_vi", "")
        name_en = doc.get("name_en", "")
        description = doc.get("description", "")
        category = doc.get("category", "")
        ingredients = doc.get("ingredients", [])
        
        # Format dish information
        dish_info = f"{i}. {name_vi}"
        if name_en:
            dish_info += f" ({name_en})"
        
        if description:
            dish_info += f"\n   Mô tả: {description}"
        
        if category:
            dish_info += f"\n   Loại: {category}"
        
        if ingredients:
            ingredients_str = ", ".join(ingredients[:5])  # Limit to first 5 ingredients
            dish_info += f"\n   Nguyên liệu: {ingredients_str}"
        
        context_parts.append(dish_info)
    
    return "\n\n".join(context_parts)


def _build_constraints_context(active_constraints: List[Dict[str, Any]]) -> str:
    """Build formatted constraints context."""
    if not active_constraints:
        return "Không có ràng buộc đặc biệt."
    
    constraint_parts = []
    for constraint in active_constraints:
        constraint_type = constraint.get("type", "")
        value = constraint.get("value", "")
        severity = constraint.get("severity", "")
        
        if constraint_type == "ALLERGY":
            constraint_parts.append(f"Dị ứng: {value} ({severity})")
        elif constraint_type == "DIETARY":
            constraint_parts.append(f"Chế độ ăn: {value}")
        elif constraint_type == "PREFERENCE":
            constraint_parts.append(f"Sở thích: {value}")
        elif constraint_type == "DISLIKE":
            constraint_parts.append(f"Không thích: {value}")
    
    return "; ".join(constraint_parts) if constraint_parts else "Không có ràng buộc đặc biệt."


def _build_availability_context(reranked_docs: List[Dict[str, Any]]) -> str:
    """Build formatted availability context."""
    if not reranked_docs:
        return "Không có thông tin về tình trạng có sẵn."
    
    availability_parts = []
    for doc in reranked_docs:
        name_vi = doc.get("name_vi", "")
        availability_status = doc.get("availability_status", "available")
        requires_advance_order = doc.get("requires_advance_order", False)
        
        if availability_status == "available" and not requires_advance_order:
            availability_parts.append(f"{name_vi}: Có sẵn ngay")
        elif availability_status == "available" and requires_advance_order:
            availability_parts.append(f"{name_vi}: Có sẵn nhưng cần đặt trước")
        elif availability_status == "limited":
            availability_parts.append(f"{name_vi}: Số lượng có hạn")
        elif availability_status == "advance_order_only":
            availability_parts.append(f"{name_vi}: Chỉ nhận đặt trước")
        elif availability_status == "unavailable":
            availability_parts.append(f"{name_vi}: Hiện không có sẵn")
    
    return "\n".join(availability_parts)


def _needs_clarification(state: AgentState) -> bool:
    """Determine if clarification is needed based on state."""
    # Check for ambiguous references
    references = state.get("references", [])
    if any("ambiguous" in ref.lower() for ref in references):
        return True
    
    # Check for conflicting constraints
    active_constraints = state.get("active_constraints", [])
    if _has_conflicting_constraints(active_constraints):
        return True
    
    # Check for very low confidence results
    reranked_docs = state.get("reranked_docs", [])
    if reranked_docs and all(doc.get("score", 0) < 0.3 for doc in reranked_docs):
        return True
    
    return False


def _has_conflicting_constraints(constraints: List[Dict[str, Any]]) -> bool:
    """Check if there are conflicting constraints."""
    # Simple conflict detection - can be enhanced
    constraint_values = [c.get("value", "").lower() for c in constraints]
    
    # Check for obvious conflicts
    conflicts = [
        ("chay", "thịt"),
        ("cay", "không cay"),
        ("nóng", "lạnh")
    ]
    
    for conflict_pair in conflicts:
        if all(term in " ".join(constraint_values) for term in conflict_pair):
            return True
    
    return False


def _identify_clarification_needs(state: AgentState) -> str:
    """Identify what needs clarification."""
    needs = []
    
    # Check references
    references = state.get("references", [])
    if references:
        needs.append("Cần làm rõ tham chiếu: " + ", ".join(references))
    
    # Check constraints
    active_constraints = state.get("active_constraints", [])
    if _has_conflicting_constraints(active_constraints):
        needs.append("Có ràng buộc mâu thuẫn cần làm rõ")
    
    # Check low confidence
    reranked_docs = state.get("reranked_docs", [])
    if reranked_docs and all(doc.get("score", 0) < 0.3 for doc in reranked_docs):
        needs.append("Yêu cầu không rõ ràng, cần thông tin thêm")
    
    return "; ".join(needs) if needs else "Cần thông tin bổ sung"


def _extract_availability_warnings(reranked_docs: List[Dict[str, Any]]) -> List[str]:
    """Extract availability warnings from documents."""
    warnings = []
    
    for doc in reranked_docs:
        name_vi = doc.get("name_vi", "")
        availability_status = doc.get("availability_status", "available")
        requires_advance_order = doc.get("requires_advance_order", False)
        
        if availability_status == "unavailable":
            warnings.append(f"{name_vi} hiện không có sẵn")
        elif availability_status == "limited":
            warnings.append(f"{name_vi} chỉ còn số lượng có hạn")
        elif requires_advance_order:
            warnings.append(f"{name_vi} cần đặt trước")
    
    return warnings


def _generate_follow_up_questions(reranked_docs: List[Dict[str, Any]], 
                                active_constraints: List[Dict[str, Any]]) -> List[str]:
    """Generate relevant follow-up questions."""
    questions = []
    
    # Questions based on availability
    if any(doc.get("requires_advance_order", False) for doc in reranked_docs):
        questions.append("Bạn có muốn đặt trước không?")
    
    # Questions based on multiple options
    if len(reranked_docs) > 1:
        questions.append("Bạn muốn biết thêm chi tiết về món nào?")
    
    # Questions based on constraints
    if not active_constraints:
        questions.append("Bạn có ràng buộc gì về ăn uống không?")
    
    # Default questions
    if not questions:
        questions.extend([
            "Bạn có muốn gọi món nào không?",
            "Bạn cần thêm thông tin gì khác?"
        ])
    
    return questions[:2]  # Limit to 2 questions


def _generate_clarification_questions(state: AgentState) -> List[str]:
    """Generate specific clarification questions based on state."""
    questions = []
    
    # Questions about references
    references = state.get("references", [])
    if references:
        questions.append("Bạn có thể nói rõ hơn về món nào bạn đang nhắc đến?")
    
    # Questions about constraints
    active_constraints = state.get("active_constraints", [])
    if _has_conflicting_constraints(active_constraints):
        questions.append("Có vẻ như có một số yêu cầu mâu thuẫn, bạn có thể làm rõ không?")
    
    # Default clarification questions
    if not questions:
        questions.extend([
            "Bạn có thể nói rõ hơn về món ăn bạn muốn tìm?",
            "Bạn có sở thích hay ràng buộc gì đặc biệt không?"
        ])
    
    return questions[:2]  # Limit to 2 questions


# Additional utility functions for testing and direct usage

class ResponseGenerator:
    """
    Response Generator class for direct usage and testing.
    
    This class provides a convenient interface for generating responses
    outside of the LangGraph workflow.
    """
    
    def __init__(self):
        """Initialize response generator."""
        self.config = get_config()
        model_config = self.config.get_current_model_config()
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model=model_config.model_id,
            base_url=model_config.base_url,
            temperature=model_config.temperature,
            num_predict=model_config.max_tokens,
            timeout=model_config.timeout
        )
    
    def generate_response(self, user_input: str, 
                         reranked_docs: List[Dict[str, Any]], 
                         active_constraints: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Generate response directly without using AgentState.
        
        Args:
            user_input: User's input text
            reranked_docs: List of reranked documents
            active_constraints: Optional list of active constraints
            
        Returns:
            Dictionary containing response data
        """
        if active_constraints is None:
            active_constraints = []
        
        try:
            if not reranked_docs:
                return _generate_no_results_response(self.llm, user_input, active_constraints)
            else:
                return _generate_recommendation_response(self.llm, user_input, reranked_docs, active_constraints)
        except Exception as e:
            return {
                "response_text": "Xin lỗi, tôi gặp sự cố khi xử lý yêu cầu của bạn. Bạn có thể thử lại được không?",
                "errors": [f"Response generation failed: {str(e)}"],
                "clarification_needed": True,
                "follow_up_questions": ["Bạn có thể nói lại yêu cầu được không?"],
                "availability_warnings": []
            }
    
    def test_prompts(self, test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Test response generation with multiple test cases.
        
        Args:
            test_cases: List of test case dictionaries
            
        Returns:
            List of response results
        """
        results = []
        
        for i, test_case in enumerate(test_cases):
            try:
                user_input = test_case.get("user_input", "")
                reranked_docs = test_case.get("reranked_docs", [])
                active_constraints = test_case.get("active_constraints", [])
                
                response = self.generate_response(user_input, reranked_docs, active_constraints)
                
                results.append({
                    "test_case": i + 1,
                    "input": user_input,
                    "success": True,
                    "response": response
                })
                
            except Exception as e:
                results.append({
                    "test_case": i + 1,
                    "input": test_case.get("user_input", ""),
                    "success": False,
                    "error": str(e)
                })
        
        return results


# Factory function for creating the LangGraph node
def create_response_generator_langgraph_node():
    """
    Create a LangGraph-compatible response generator node.
    
    This is the main factory function that should be used when
    integrating the response generator into a LangGraph workflow.
    
    Returns:
        Function that can be used as a LangGraph node
    """
    node_function = create_response_generator_node()
    
    def response_generator_langgraph_node(state: AgentState) -> AgentState:
        """
        LangGraph node wrapper for response generation.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with response fields
        """
        # Call the core node function
        response_data = node_function(state)
        
        # Merge response data into state
        updated_state = state.copy()
        updated_state.update(response_data)
        
        return updated_state
    
    return response_generator_langgraph_node


# For testing and development
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the response generator
    generator = ResponseGenerator()
    
    # Test case 1: Successful recommendation
    test_docs = [
        {
            "id": "pho_bo",
            "name_vi": "Phở Bò",
            "name_en": "Beef Pho",
            "description": "Traditional Vietnamese beef noodle soup with herbs",
            "category": "main",
            "ingredients": ["beef", "rice noodles", "herbs", "broth"],
            "availability_status": "available",
            "requires_advance_order": False,
            "score": 0.9
        },
        {
            "id": "bun_bo_hue",
            "name_vi": "Bún Bò Huế",
            "name_en": "Hue Beef Noodle Soup",
            "description": "Spicy beef noodle soup from Hue",
            "category": "main",
            "ingredients": ["beef", "rice vermicelli", "lemongrass", "chili"],
            "availability_status": "available",
            "requires_advance_order": True,
            "score": 0.8
        }
    ]
    
    test_constraints = [
        {
            "type": "PREFERENCE",
            "value": "beef",
            "severity": "MODERATE",
            "source_text": "tôi thích thịt bò"
        }
    ]
    
    print("=== Testing Response Generator ===")
    
    # Test successful recommendation
    print("\n1. Testing successful recommendation:")
    result = generator.generate_response(
        user_input="tôi muốn ăn phở bò",
        reranked_docs=test_docs,
        active_constraints=test_constraints
    )
    print(f"Response: {result['response_text']}")
    print(f"Follow-up questions: {result['follow_up_questions']}")
    print(f"Availability warnings: {result['availability_warnings']}")
    
    # Test no results
    print("\n2. Testing no results:")
    result = generator.generate_response(
        user_input="tôi muốn ăn pizza",
        reranked_docs=[],
        active_constraints=[]
    )
    print(f"Response: {result['response_text']}")
    print(f"Clarification needed: {result['clarification_needed']}")
    
    # Test LangGraph node
    print("\n3. Testing LangGraph node:")
    node = create_response_generator_langgraph_node()
    
    test_state = {
        "raw_input": "tôi muốn ăn phở",
        "corrected_input": "tôi muốn ăn phở",
        "reranked_docs": test_docs[:1],  # Just one doc
        "active_constraints": [],
        "session_id": "test_session"
    }
    
    updated_state = node(test_state)
    print(f"Generated response: {updated_state.get('response_text', 'No response')}")
    print(f"State keys: {list(updated_state.keys())}")
    
    print("\n=== Response Generator Testing Complete ===")