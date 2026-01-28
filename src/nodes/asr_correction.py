"""
Vietnamese Food Assistant - ASR Correction Node

This module implements the ASR correction node that uses a conservative approach
to correct spelling errors and normalize text from ASR input while strictly
preserving semantic meaning and intent.
"""

from typing import Dict, Any, Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
import sys
from pathlib import Path

# Add root directory to path for conservative correction import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import conservative ASR correction if available, otherwise use fallback
try:
    from conservative_asr_correction import ConservativeASRCorrection
except ImportError:
    # Fallback implementation for demo purposes
    class ConservativeASRCorrection:
        def correct_asr_text(self, text: str) -> str:
            """Simple fallback ASR correction."""
            # Basic Vietnamese text normalization
            corrections = {
                'pho': 'phở',
                'bun': 'bún', 
                'com': 'cơm',
                'banh mi': 'bánh mì',
                'banh': 'bánh',
                'toi': 'tôi',
                'muon': 'muốn',
                'an': 'ăn',
                'co': 'có',
                'gi': 'gì',
                'khong': 'không'
            }
            
            corrected = text.lower()
            for wrong, correct in corrections.items():
                corrected = corrected.replace(wrong, correct)
            
            return corrected
        
        def get_correction_confidence(self, original: str, corrected: str) -> float:
            """Return confidence score for correction."""
            if original == corrected:
                return 1.0
            return 0.8  # Default confidence for corrections
from ..config import get_config
from ..state import AgentState


class TextCorrectionOutputParser(BaseOutputParser[str]):
    """Parser for text correction output that extracts only the corrected text."""
    
    def parse(self, text: str) -> str:
        """Parse the LLM output to extract corrected text."""
        # Remove any explanations or metadata, keep only the corrected text
        lines = text.strip().split('\n')
        
        # Look for the corrected text (usually the first substantial line)
        for line in lines:
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith('Corrected:'):
                return line
        
        # Fallback to the first non-empty line
        return lines[0].strip() if lines else text.strip()


class ASRCorrectionPrompts:
    """
    Collection of system prompts for ASR text correction.
    
    These prompts are designed to correct Vietnamese spelling errors
    from ASR systems while preserving the original intent and meaning.
    """
    
    SYSTEM_PROMPT = """Bạn là một chuyên gia sửa lỗi chính tả tiếng Việt từ hệ thống nhận dạng giọng nói (ASR).

NHIỆM VỤ:
- CHỈ sửa lỗi chính tả và ngữ pháp rõ ràng
- TUYỆT ĐỐI KHÔNG thay đổi ý nghĩa hoặc nội dung
- TUYỆT ĐỐI KHÔNG thay thế tên món ăn bằng món ăn khác
- Giữ nguyên 100% ý định và ngữ cảnh gốc

QUY TẮC QUAN TRỌNG:
1. CHỈ sửa lỗi chính tả rõ ràng, KHÔNG đoán ý
2. KHÔNG thay đổi tên món ăn hoặc nguyên liệu
3. KHÔNG thêm hoặc bớt thông tin
4. Nếu không chắc chắn, GIỮ NGUYÊN văn bản gốc
5. Trả về CHỈ văn bản đã sửa, không giải thích

CÁC LỖI ASR CÓ THỂ SỬA:
- Lỗi dấu thanh rõ ràng: "pho" → "phở"
- Lỗi chính tả đơn giản: "banh mi" → "bánh mì"
- Lỗi viết hoa/thường: "PHO" → "phở"
- Lỗi khoảng trắng: "banhmi" → "bánh mì"

KHÔNG SỬA:
- Tên món ăn đã đúng
- Từ có thể có nhiều cách hiểu
- Nội dung không rõ ràng là lỗi

ĐỊNH DẠNG ĐẦU RA:
Chỉ trả về văn bản đã được sửa lỗi, không có thêm gì khác."""

    CORRECTION_TEMPLATE = """Văn bản cần sửa: "{input_text}"

Văn bản đã sửa:"""

    @classmethod
    def get_correction_prompt(cls) -> PromptTemplate:
        """Get the main correction prompt template."""
        return PromptTemplate(
            input_variables=["input_text"],
            template=f"{cls.SYSTEM_PROMPT}\n\n{cls.CORRECTION_TEMPLATE}"
        )

    @classmethod
    def get_food_specific_prompt(cls) -> PromptTemplate:
        """Get food-specific correction prompt with enhanced food terminology handling."""
        food_specific_system = f"""{cls.SYSTEM_PROMPT}

THUẬT NGỮ ẨM THỰC QUAN TRỌNG - TUYỆT ĐỐI KHÔNG THAY ĐỔI:
- Món chính: phở, bún bò, cơm tấm, bánh mì, bánh xèo, chả cá, thịt bò, thịt heo
- Nguyên liệu: tôm, cua, rau muống, giá đỗ, thịt bò, thịt heo, gà
- Gia vị: nước mắm, tương ớt, chanh, ngò, hành
- Đặc điểm: cay, ngọt, chua, mặn, béo, thanh đạm

CHÚ Ý ĐẶC BIỆT:
- TUYỆT ĐỐI KHÔNG thay "thịt bò" thành "phở"
- TUYỆT ĐỐI KHÔNG thay món ăn này thành món ăn khác
- CHỈ sửa lỗi chính tả rõ ràng, KHÔNG đoán ý
- Nếu không chắc chắn, GIỮ NGUYÊN"""

        return PromptTemplate(
            input_variables=["input_text"],
            template=f"{food_specific_system}\n\n{cls.CORRECTION_TEMPLATE}"
        )

    @classmethod
    def get_mixed_language_prompt(cls) -> PromptTemplate:
        """Get prompt for handling mixed Vietnamese-English text."""
        mixed_language_system = f"""{cls.SYSTEM_PROMPT}

XỬ LÝ NGÔN NGỮ PHA TRỘN:
- Giữ nguyên từ tiếng Anh khi phù hợp: "order", "menu", "spicy"
- Sửa lỗi tiếng Việt: "tôi muốn order món phở"
- Không ép buộc dịch sang một ngôn ngữ duy nhất
- Tôn trọng cách nói tự nhiên của người Việt

VÍ DỤ:
- "tôi muốn oder món fở" → "tôi muốn order món phở"
- "cho tôi một bánh mì spicy" → "cho tôi một bánh mì spicy"
- "menu có gì ngon" → "menu có gì ngon" """

        return PromptTemplate(
            input_variables=["input_text"],
            template=f"{mixed_language_system}\n\n{cls.CORRECTION_TEMPLATE}"
        )


def create_asr_correction_node():
    """
    Create the ASR correction node function for LangGraph.
    
    This function corrects ASR text input using a conservative approach that
    prioritizes semantic meaning preservation over aggressive correction.
    
    Returns:
        Callable that takes AgentState and returns updated AgentState
    """
    
    # Initialize conservative ASR corrector
    conservative_corrector = ConservativeASRCorrection()
    
    def asr_correction_node(state: AgentState) -> Dict[str, Any]:
        """
        ASR Correction Node - Corrects spelling errors in ASR text input using conservative approach.
        
        Args:
            state: Current agent state containing raw_input
            
        Returns:
            Updated state with corrected_input field
        """
        # Get input text
        raw_input = state.get("raw_input", "")
        
        if not raw_input or not raw_input.strip():
            return {
                "corrected_input": raw_input,
                "warnings": state.get("warnings", []) + ["Empty input received"]
            }
        
        try:
            # Apply conservative ASR correction with semantic validation
            corrected_text = conservative_corrector.correct_asr_text(raw_input)
            
            # Get confidence score for the correction
            confidence = conservative_corrector.get_correction_confidence(raw_input, corrected_text)
            
            # Log correction details
            if corrected_text != raw_input:
                print(f"ASR Correction applied: '{raw_input}' -> '{corrected_text}' (confidence: {confidence:.3f})")
            
            return {
                "corrected_input": corrected_text,
                "warnings": state.get("warnings", [])
            }
                
        except Exception as e:
            # Fallback to original text on error (conservative approach)
            return {
                "corrected_input": raw_input,
                "errors": state.get("errors", []) + [f"ASR correction failed: {str(e)}"]
            }
    
    return asr_correction_node


# Legacy functions kept for backward compatibility but not used in conservative approach

def _contains_mixed_language(text: str) -> bool:
    """Check if text contains mixed Vietnamese-English content."""
    english_words = ["order", "menu", "spicy", "mild", "hot", "cold", "drink", "food"]
    vietnamese_words = ["món", "tôi", "muốn", "cho", "với", "và", "có", "gì", "ngon"]
    
    text_lower = text.lower()
    has_english = any(word in text_lower for word in english_words)
    has_vietnamese = any(word in text_lower for word in vietnamese_words)
    
    return has_english and has_vietnamese


def _contains_food_terms(text: str) -> bool:
    """Check if text contains Vietnamese food terminology."""
    food_terms = [
        "phở", "bún", "cơm", "bánh", "chả", "nem", "gỏi", "canh", "soup",
        "thịt", "tôm", "cua", "cá", "gà", "heo", "bò",
        "cay", "ngọt", "chua", "mặn", "béo", "thanh đạm",
        "nước", "trà", "cà phê", "bia", "rượu"
    ]
    
    text_lower = text.lower()
    return any(term in text_lower for term in food_terms)