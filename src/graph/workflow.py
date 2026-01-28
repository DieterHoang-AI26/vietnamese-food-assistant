"""
Vietnamese Food Assistant - LangGraph Workflow

This module implements the LangGraph orchestration for the Vietnamese Food Assistant.
It defines the graph topology with nodes for ASR correction, context management,
retrieval, logic filtering, and response generation with comprehensive error handling.

Requirements: 4.1 - Menu Integration and Availability Management
Requirements: 7.5 - Session Management and Data Persistence
"""

from typing import Dict, Any, List, Optional
import logging
from langgraph.graph import StateGraph, END

from ..state import AgentState
from ..nodes.asr_correction import create_asr_correction_node
from ..nodes.context_manager import create_context_node
from ..nodes.constraint_extraction import create_constraint_extraction_node
from ..nodes.constraint_accumulator import create_constraint_accumulation_node
from ..nodes.retrieval_node import create_retrieval_node
from ..nodes.logic_filters import create_logic_filters
from ..nodes.reranking_node import create_reranking_node
from ..nodes.response_generator import create_response_generator_node
from ..config import get_config
from ..error_handling import ErrorHandler, ErrorCategory, handle_error


class VietnamFoodAssistantWorkflow:
    """
    Main workflow class for the Vietnamese Food Assistant using LangGraph.
    
    This class orchestrates the entire pipeline from ASR input to final response,
    managing the flow between different processing nodes.
    """
    
    def __init__(self):
        """Initialize the workflow with all necessary components."""
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        self.error_handler = ErrorHandler()
        
        # Initialize all nodes
        self.asr_node = create_asr_correction_node()
        self.context_node = create_context_node()
        self.constraint_extraction_node = create_constraint_extraction_node()
        self.constraint_accumulator_node = create_constraint_accumulation_node()
        self.retrieval_node = create_retrieval_node()
        self.logic_filters_node = create_logic_filters()
        self.reranking_node = create_reranking_node()
        self.response_generator_node = create_response_generator_node()
        
        # Build the graph
        self.graph = self._build_graph()
        
        self.logger.info("Vietnamese Food Assistant workflow initialized")
    
    def _build_graph(self):
        """
        Build the LangGraph workflow with all nodes and edges.
        
        Returns:
            Compiled LangGraph workflow
        """
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add all nodes to the graph
        workflow.add_node("asr_correction", self._wrap_node(self.asr_node, "ASR Correction"))
        workflow.add_node("context_management", self._wrap_node(self.context_node, "Context Management"))
        workflow.add_node("constraint_extraction", self._wrap_node(self.constraint_extraction_node, "Constraint Extraction"))
        workflow.add_node("constraint_accumulation", self._wrap_node(self.constraint_accumulator_node, "Constraint Accumulation"))
        workflow.add_node("retrieval", self._wrap_node(self.retrieval_node, "Retrieval"))
        workflow.add_node("logic_filters", self._wrap_node(self.logic_filters_node, "Logic Filters"))
        workflow.add_node("reranking", self._wrap_node(self.reranking_node, "Reranking"))
        workflow.add_node("response_generation", self._wrap_node(self.response_generator_node, "Response Generation"))
        
        # Define the flow with early conditional routing
        workflow.set_entry_point("asr_correction")
        
        # Early conditional routing after ASR correction for special intents
        workflow.add_conditional_edges(
            "asr_correction",
            self._should_continue_pipeline,
            {
                "continue": "context_management",
                "end_early": END
            }
        )
        
        # Linear flow: Context -> Constraint Extraction -> Constraint Accumulation
        workflow.add_edge("context_management", "constraint_extraction")
        workflow.add_edge("constraint_extraction", "constraint_accumulation")
        
        # From constraint accumulation to retrieval
        workflow.add_edge("constraint_accumulation", "retrieval")
        
        # Conditional routing from retrieval to determine filtering path
        workflow.add_conditional_edges(
            "retrieval",
            self._should_apply_filters,
            {
                "apply_filters": "logic_filters",
                "skip_to_reranking": "reranking"
            }
        )
        
        # From logic filters to reranking (when filters are applied)
        workflow.add_edge("logic_filters", "reranking")
        
        # Conditional routing from reranking
        workflow.add_conditional_edges(
            "reranking",
            self._should_generate_response,
            {
                "generate_response": "response_generation",
                "end": END
            }
        )
        
        # From response generation to end
        workflow.add_edge("response_generation", END)
        
        # Compile the graph
        return workflow.compile()
    
    def _wrap_node(self, node_func, node_name: str):
        """
        Wrap a node function with logging and comprehensive error handling.
        
        Args:
            node_func: The node function to wrap
            node_name: Name of the node for logging
            
        Returns:
            Wrapped node function with error handling
        """
        def wrapped_node(state: AgentState) -> AgentState:
            """Wrapped node with logging and comprehensive error handling."""
            session_id = state.get("session_id", "unknown")
            
            try:
                self.logger.debug(f"Executing {node_name} node for session {session_id}")
                
                # Execute the node
                result = node_func(state)
                
                # Ensure result is a dictionary (AgentState)
                if not isinstance(result, dict):
                    self.logger.error(f"{node_name} node returned non-dict result: {type(result)}")
                    return state
                
                # Merge result into state
                updated_state = state.copy()
                updated_state.update(result)
                
                self.logger.debug(f"Completed {node_name} node for session {session_id}")
                return updated_state
                
            except Exception as e:
                self.logger.error(f"Error in {node_name} node for session {session_id}: {e}")
                
                # Determine error category based on node name
                error_category = self._get_error_category_for_node(node_name)
                
                # Handle error with comprehensive error handling
                error_response = self.error_handler.handle_error(
                    exception=e,
                    category=error_category,
                    component=node_name,
                    session_id=session_id,
                    context={"state": state, "node_name": node_name}
                )
                
                # Merge error response into state
                error_state = state.copy()
                
                # Add error information
                if "errors" not in error_state:
                    error_state["errors"] = []
                error_state["errors"].append(f"{node_name} failed: {str(e)}")
                
                # Add recovery information if available
                if error_response.get("error_handled"):
                    error_state["error_recovery"] = error_response
                    
                    # If response text is provided, use it
                    if error_response.get("response_text"):
                        error_state["response_text"] = error_response["response_text"]
                        error_state["clarification_needed"] = error_response.get("clarification_needed", True)
                        error_state["follow_up_questions"] = error_response.get("follow_up_questions", [])
                
                # Mark node as failed but allow workflow to continue
                error_state[f"{node_name}_failed"] = True
                error_state[f"{node_name}_error"] = str(e)
                
                return error_state
        
        return wrapped_node
    
    def _get_error_category_for_node(self, node_name: str) -> ErrorCategory:
        """
        Get appropriate error category for a node.
        
        Args:
            node_name: Name of the node
            
        Returns:
            Appropriate error category
        """
        node_category_mapping = {
            "asr_correction": ErrorCategory.ASR_CORRECTION,
            "context_management": ErrorCategory.CONTEXT_MANAGEMENT,
            "constraint_extraction": ErrorCategory.CONSTRAINT_PROCESSING,
            "constraint_accumulation": ErrorCategory.CONSTRAINT_PROCESSING,
            "retrieval": ErrorCategory.RETRIEVAL,
            "logic_filters": ErrorCategory.FILTERING,
            "reranking": ErrorCategory.FILTERING,
            "response_generation": ErrorCategory.RESPONSE_GENERATION
        }
        
        return node_category_mapping.get(node_name, ErrorCategory.UNKNOWN)
    
    def _should_generate_response(self, state: AgentState) -> str:
        """
        Conditional routing function to determine if response should be generated.
        
        This implements the conditional logic:
        - If len(docs) == 0 -> Skip Generator
        - Intent-based routing using Python string matching/classification
        
        Args:
            state: Current agent state
            
        Returns:
            "generate_response" or "end"
        """
        reranked_docs = state.get("reranked_docs", [])
        corrected_input = state.get("corrected_input", "").lower()
        
        # Check if we have documents
        has_documents = reranked_docs and len(reranked_docs) > 0
        
        # Intent-based routing using Python string matching
        intent = self._classify_intent(corrected_input)
        
        self.logger.debug(f"Intent classification: {intent}, Documents: {len(reranked_docs) if reranked_docs else 0}")
        
        # Handle different intents
        if intent == "greeting":
            # For greetings, generate response even without documents
            state["response_text"] = "Xin chào! Tôi là trợ lý tư vấn món ăn Việt Nam. Bạn muốn tìm món gì hôm nay?"
            state["clarification_needed"] = False
            state["follow_up_questions"] = [
                "Bạn có muốn xem menu không?",
                "Bạn thích món gì?"
            ]
            return "end"
        
        elif intent == "menu_inquiry":
            # For menu inquiries, try to generate response even with few documents
            if has_documents:
                return "generate_response"
            else:
                state["response_text"] = "Menu của chúng tôi có nhiều món Việt Nam truyền thống. Bạn có muốn tìm loại món nào cụ thể không?"
                state["clarification_needed"] = True
                state["follow_up_questions"] = [
                    "Bạn thích món chính hay món phụ?",
                    "Bạn có sở thích gì đặc biệt không?"
                ]
                return "end"
        
        elif intent == "specific_dish":
            # For specific dish requests, need documents to generate meaningful response
            if has_documents:
                return "generate_response"
            else:
                # Try to provide helpful response even without exact matches
                state["response_text"] = "Xin lỗi, tôi không tìm thấy món ăn chính xác mà bạn yêu cầu. Bạn có thể mô tả rõ hơn hoặc thử tên món khác không?"
                state["clarification_needed"] = True
                state["follow_up_questions"] = [
                    "Bạn có thể mô tả món ăn đó như thế nào?",
                    "Bạn có nhớ tên tiếng Anh của món đó không?"
                ]
                return "end"
        
        elif intent == "dietary_constraint":
            # For dietary constraints, provide guidance even without specific dishes
            if has_documents:
                return "generate_response"
            else:
                constraints = state.get("active_constraints", [])
                constraint_text = self._format_constraints_for_response(constraints)
                state["response_text"] = f"Tôi đã ghi nhận các yêu cầu của bạn: {constraint_text}. Để tìm món phù hợp, bạn có thể cho tôi biết thêm về loại món bạn muốn ăn không?"
                state["clarification_needed"] = True
                state["follow_up_questions"] = [
                    "Bạn muốn ăn món chính hay món nhẹ?",
                    "Bạn thích món nóng hay món lạnh?"
                ]
                return "end"
        
        elif intent == "availability_inquiry":
            # For availability questions, need documents to provide accurate info
            if has_documents:
                return "generate_response"
            else:
                state["response_text"] = "Để kiểm tra tình trạng có sẵn, bạn có thể cho tôi biết tên món cụ thể không?"
                state["clarification_needed"] = True
                state["follow_up_questions"] = [
                    "Bạn muốn kiểm tra món nào?",
                    "Bạn cần món có sẵn ngay hay có thể đặt trước?"
                ]
                return "end"
        
        else:
            # Default behavior: generate response if we have documents
            if has_documents:
                self.logger.debug(f"Found {len(reranked_docs)} documents, generating response")
                return "generate_response"
            
            # If no documents, provide default no-results response
            self.logger.debug("No documents found, skipping response generation")
            state["response_text"] = "Xin lỗi, tôi không tìm thấy món ăn phù hợp với yêu cầu của bạn. Bạn có thể thử tìm kiếm với từ khóa khác không?"
            state["clarification_needed"] = True
            state["follow_up_questions"] = [
                "Bạn có thể mô tả rõ hơn món ăn bạn muốn tìm?",
                "Bạn có muốn xem menu tổng quát không?"
            ]
            return "end"
    
    def _classify_intent(self, input_text: str) -> str:
        """
        Classify user intent using Python string matching and pattern recognition.
        
        Args:
            input_text: Processed user input (lowercase)
            
        Returns:
            Classified intent string
        """
        # Greeting patterns
        greeting_patterns = [
            "xin chào", "chào", "hello", "hi", "good morning", "good afternoon", 
            "good evening", "chào bạn", "xin chào bạn"
        ]
        
        # Menu inquiry patterns
        menu_patterns = [
            "menu", "thực đơn", "có món gì", "món nào", "danh sách món", 
            "xem menu", "menu có gì", "có những món gì"
        ]
        
        # Specific dish patterns
        specific_dish_patterns = [
            "phở", "bún", "bánh", "cơm", "chả", "nem", "gỏi", "canh",
            "tôi muốn", "cho tôi", "lấy", "đặt", "gọi món"
        ]
        
        # Dietary constraint patterns
        dietary_patterns = [
            "dị ứng", "kiêng", "không ăn", "chay", "không thích", "ngán",
            "allergy", "vegetarian", "vegan", "halal", "không được ăn"
        ]
        
        # Availability inquiry patterns
        availability_patterns = [
            "có sẵn", "còn không", "hết chưa", "available", "có không",
            "đặt trước", "bao lâu", "mất bao lâu", "nhanh không"
        ]
        
        # Check patterns in order of specificity
        if any(pattern in input_text for pattern in greeting_patterns):
            return "greeting"
        
        if any(pattern in input_text for pattern in availability_patterns):
            return "availability_inquiry"
        
        if any(pattern in input_text for pattern in dietary_patterns):
            return "dietary_constraint"
        
        if any(pattern in input_text for pattern in specific_dish_patterns):
            return "specific_dish"
        
        if any(pattern in input_text for pattern in menu_patterns):
            return "menu_inquiry"
        
        # Default intent
        return "general_inquiry"
    
    def _format_constraints_for_response(self, constraints: List[Dict[str, Any]]) -> str:
        """
        Format constraints for inclusion in response text.
        
        Args:
            constraints: List of active constraints
            
        Returns:
            Formatted constraint string
        """
        if not constraints:
            return "không có yêu cầu đặc biệt"
        
        constraint_texts = []
        for constraint in constraints:
            constraint_type = constraint.get("type", "")
            value = constraint.get("value", "")
            
            if constraint_type == "ALLERGY":
                constraint_texts.append(f"dị ứng {value}")
            elif constraint_type == "DIETARY":
                constraint_texts.append(f"chế độ ăn {value}")
            elif constraint_type == "DISLIKE":
                constraint_texts.append(f"không thích {value}")
            elif constraint_type == "PREFERENCE":
                constraint_texts.append(f"thích {value}")
        
        return ", ".join(constraint_texts)
    
    def _should_apply_filters(self, state: AgentState) -> str:
        """
        Conditional routing function to determine if logic filters should be applied.
        
        This implements conditional logic based on:
        - Number of retrieved documents
        - Presence of strict constraints
        - Intent classification
        
        Args:
            state: Current agent state
            
        Returns:
            "apply_filters" or "skip_to_reranking"
        """
        retrieved_docs = state.get("retrieved_docs", [])
        active_constraints = state.get("active_constraints", [])
        corrected_input = state.get("corrected_input", "").lower()
        
        # Always apply filters if we have strict constraints (safety critical)
        strict_constraints = [
            c for c in active_constraints 
            if c.get("type") in ["ALLERGY"] or 
            (c.get("type") in ["DIETARY", "DISLIKE"] and c.get("severity") == "STRICT")
        ]
        
        if strict_constraints:
            self.logger.debug("Applying filters due to strict constraints")
            return "apply_filters"
        
        # If we have very few documents, skip filtering to avoid empty results
        if len(retrieved_docs) <= 2:
            self.logger.debug("Skipping filters due to few retrieved documents")
            return "skip_to_reranking"
        
        # For greeting or general inquiries, skip filtering
        intent = self._classify_intent(corrected_input)
        if intent in ["greeting", "menu_inquiry"]:
            self.logger.debug(f"Skipping filters for intent: {intent}")
            return "skip_to_reranking"
        
        # Default: apply filters
        self.logger.debug("Applying filters (default behavior)")
        return "apply_filters"
    
    def _should_continue_pipeline(self, state: AgentState) -> str:
        """
        Early conditional routing to determine if we should continue the full pipeline.
        
        This handles special cases like greetings that can be answered immediately
        without going through retrieval and filtering.
        
        Args:
            state: Current agent state after ASR correction
            
        Returns:
            "continue" or "end_early"
        """
        corrected_input = state.get("corrected_input", "").lower()
        intent = self._classify_intent(corrected_input)
        
        # Handle greetings immediately
        if intent == "greeting":
            self.logger.debug("Ending early for greeting intent")
            state["response_text"] = "Xin chào! Tôi là trợ lý tư vấn món ăn Việt Nam. Bạn muốn tìm món gì hôm nay?"
            state["clarification_needed"] = False
            state["follow_up_questions"] = [
                "Bạn có muốn xem menu không?",
                "Bạn thích món gì?"
            ]
            return "end_early"
        
        # For all other intents, continue with the full pipeline
        self.logger.debug(f"Continuing pipeline for intent: {intent}")
        return "continue"
    
    def process_request(self, raw_input: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Process a user request through the complete workflow with comprehensive error handling.
        
        Args:
            raw_input: Raw user input (from ASR or text)
            session_id: Session identifier for context management
            
        Returns:
            Final response with all processing results and error handling
        """
        try:
            self.logger.info(f"Processing request: '{raw_input}' for session: {session_id}")
            
            # Initialize state
            initial_state = self._create_initial_state(raw_input, session_id)
            
            # Execute the workflow with error handling
            final_state = self._execute_workflow_with_error_handling(initial_state)
            
            # Extract response data
            response = self._extract_response(final_state)
            
            # Check if workflow completed successfully
            if final_state.get("errors"):
                self.logger.warning(f"Workflow completed with errors for session {session_id}: {final_state['errors']}")
                response["success"] = False
                response["errors"] = final_state["errors"]
            else:
                self.logger.info(f"Request processed successfully for session: {session_id}")
                response["success"] = True
            
            return response
            
        except Exception as e:
            self.logger.error(f"Critical error processing request for session {session_id}: {e}")
            
            # Handle critical workflow errors
            error_response = self.error_handler.handle_error(
                exception=e,
                category=ErrorCategory.UNKNOWN,
                component="workflow",
                session_id=session_id,
                context={"raw_input": raw_input}
            )
            
            return {
                "response_text": error_response.get("response_text", "Xin lỗi, tôi gặp sự cố khi xử lý yêu cầu của bạn. Bạn có thể thử lại được không?"),
                "error": str(e),
                "success": False,
                "critical_error": True,
                "clarification_needed": error_response.get("clarification_needed", True),
                "follow_up_questions": error_response.get("follow_up_questions", [])
            }
    
    def _execute_workflow_with_error_handling(self, initial_state: AgentState) -> AgentState:
        """
        Execute workflow with comprehensive error handling and recovery.
        
        Args:
            initial_state: Initial workflow state
            
        Returns:
            Final state after workflow execution
        """
        try:
            # Execute the main workflow
            final_state = self.graph.invoke(initial_state)
            
            # Check for node failures and attempt recovery
            if self._has_critical_node_failures(final_state):
                self.logger.warning("Critical node failures detected, attempting recovery")
                final_state = self._attempt_workflow_recovery(final_state)
            
            return final_state
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            
            # Create error state
            error_state = initial_state.copy()
            error_state["errors"] = [f"Workflow execution failed: {str(e)}"]
            error_state["workflow_failed"] = True
            
            return error_state
    
    def _has_critical_node_failures(self, state: AgentState) -> bool:
        """
        Check if there are critical node failures that require recovery.
        
        Args:
            state: Current workflow state
            
        Returns:
            True if critical failures detected
        """
        critical_nodes = ["asr_correction", "context_management"]
        
        for node in critical_nodes:
            if state.get(f"{node}_failed"):
                return True
        
        return False
    
    def _attempt_workflow_recovery(self, state: AgentState) -> AgentState:
        """
        Attempt to recover from workflow failures.
        
        Args:
            state: State with failures
            
        Returns:
            Recovered state
        """
        session_id = state.get("session_id", "unknown")
        self.logger.info(f"Attempting workflow recovery for session {session_id}")
        
        # If ASR correction failed, use original input
        if state.get("asr_correction_failed"):
            state["corrected_input"] = state.get("raw_input", "")
            self.logger.info("Recovered from ASR correction failure using original input")
        
        # If context management failed, create minimal context
        if state.get("context_management_failed"):
            state["conversation_history"] = []
            state["mentioned_dishes"] = []
            state["active_constraints"] = []
            self.logger.info("Recovered from context management failure with minimal context")
        
        # If retrieval failed, provide fallback response
        if state.get("retrieval_failed") or not state.get("retrieved_docs"):
            state["response_text"] = "Xin lỗi, tôi không tìm thấy món ăn phù hợp. Bạn có thể thử với từ khóa khác không?"
            state["clarification_needed"] = True
            state["follow_up_questions"] = [
                "Bạn có thể mô tả rõ hơn món ăn bạn muốn tìm?",
                "Bạn có muốn xem menu tổng quát không?"
            ]
            self.logger.info("Recovered from retrieval failure with fallback response")
        
        state["workflow_recovered"] = True
        return state
    
    def _create_initial_state(self, raw_input: str, session_id: str) -> AgentState:
        """
        Create initial state for the workflow.
        
        Args:
            raw_input: Raw user input
            session_id: Session identifier
            
        Returns:
            Initial agent state
        """
        return {
            # Input Processing
            "raw_input": raw_input,
            "corrected_input": "",
            "processed_input": "",
            
            # Session Management
            "session_id": session_id,
            "conversation_history": [],
            "mentioned_dishes": [],
            
            # Constraints and Preferences
            "active_constraints": [],
            "session_preferences": {},
            
            # Retrieved Documents
            "retrieved_docs": [],
            "filtered_docs": [],
            "reranked_docs": [],
            
            # Processing Metadata
            "intent": "",
            "entities": [],
            "references": [],
            
            # Response Generation
            "response_text": "",
            "clarification_needed": False,
            "follow_up_questions": [],
            "availability_warnings": [],
            
            # Error Handling
            "errors": [],
            "warnings": []
        }
    
    def _extract_response(self, final_state: AgentState) -> Dict[str, Any]:
        """
        Extract response data from final state.
        
        Args:
            final_state: Final state after workflow execution
            
        Returns:
            Response dictionary
        """
        return {
            "response_text": final_state.get("response_text", ""),
            "clarification_needed": final_state.get("clarification_needed", False),
            "follow_up_questions": final_state.get("follow_up_questions", []),
            "availability_warnings": final_state.get("availability_warnings", []),
            "mentioned_dishes": final_state.get("mentioned_dishes", []),
            "active_constraints": final_state.get("active_constraints", []),
            "reranked_docs": final_state.get("reranked_docs", []),
            "errors": final_state.get("errors", []),
            "warnings": final_state.get("warnings", []),
            "success": len(final_state.get("errors", [])) == 0
        }
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """
        Get information about the workflow structure.
        
        Returns:
            Dictionary with workflow information
        """
        return {
            "nodes": [
                "asr_correction",
                "context_management", 
                "constraint_extraction",
                "constraint_accumulation",
                "retrieval",
                "logic_filters",
                "reranking",
                "response_generation"
            ],
            "entry_point": "asr_correction",
            "conditional_routing": {
                "retrieval": "Determines whether to apply logic filters based on constraints and document count",
                "reranking": "Checks if documents exist and classifies intent to determine response generation"
            },
            "intent_classification": [
                "greeting",
                "menu_inquiry", 
                "specific_dish",
                "dietary_constraint",
                "availability_inquiry",
                "general_inquiry"
            ],
            "description": "Vietnamese Food Assistant workflow with ASR correction, context management, retrieval, and intelligent conditional routing"
        }


def create_workflow() -> VietnamFoodAssistantWorkflow:
    """
    Factory function to create the Vietnamese Food Assistant workflow.
    
    Returns:
        Configured workflow instance
    """
    return VietnamFoodAssistantWorkflow()


# For testing and development
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create workflow
    workflow = create_workflow()
    
    # Test requests
    test_requests = [
        "tôi muốn ăn phở bò",
        "có món chay nào không",
        "tôi dị ứng tôm, có món gì phù hợp",
        "món cay",
        "bánh mì"
    ]
    
    print("=== Testing Vietnamese Food Assistant Workflow ===")
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n{i}. Testing request: '{request}'")
        
        try:
            response = workflow.process_request(request, f"test_session_{i}")
            
            print(f"   Success: {response['success']}")
            print(f"   Response: {response['response_text'][:100]}...")
            
            if response.get("reranked_docs"):
                print(f"   Found {len(response['reranked_docs'])} dishes")
            
            if response.get("errors"):
                print(f"   Errors: {response['errors']}")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    # Print workflow info
    print("\n=== Workflow Information ===")
    info = workflow.get_workflow_info()
    print(f"Nodes: {', '.join(info['nodes'])}")
    print(f"Entry point: {info['entry_point']}")
    print(f"Description: {info['description']}")