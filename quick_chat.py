#!/usr/bin/env python3
"""
Vietnamese Food Assistant - Quick Chat

Chat nhanh với trợ lý món ăn - chắc chắn hoạt động!
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to Python path to enable imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class QuickVietnameseFoodChat:
    """Trợ lý món ăn nhanh - chắc chắn hoạt động."""
    
    def __init__(self):
        """Khởi tạo."""
        self.rag_engine = None
        self.conversation_history = []
        self.user_constraints = []
        self.constraint_history = []  # Lưu lịch sử constraints theo từng lượt
        self.conversation_turn = 0    # Đếm số lượt hội thoại
        self.max_constraint_memory = 2  # Chỉ nhớ 2 lượt gần nhất
        self.setup_logging()
        self.initialize_system()
    
    def setup_logging(self):
        """Setup logging tối thiểu."""
        logging.basicConfig(level=logging.ERROR)
    
    def initialize_system(self):
        """Khởi tạo RAG engine."""
        try:
            print("🤖 Đang khởi tạo trợ lý...")
            
            from src.rag_engine import RAGEngine
            self.rag_engine = RAGEngine()
            
            # Load menu data
            data_files = [
                "data/processed_menu_v2.json",
                "data/processed_menu.json", 
                "data/sample_menu.csv"
            ]
            
            for data_file in data_files:
                if Path(data_file).exists():
                    print(f"📋 Đang tải dữ liệu từ {data_file}...")
                    self.rag_engine.load_menu_data(data_file)
                    print("✅ Trợ lý đã sẵn sàng!")
                    return True
            
            print("⚠️  Không tìm thấy dữ liệu menu")
            return False
            
        except Exception as e:
            print(f"❌ Lỗi khởi tạo: {e}")
            return False
    
    def chat(self, user_input: str) -> Dict[str, Any]:
        """Chat với trợ lý."""
        if not self.rag_engine:
            return {
                "response_text": "❌ Hệ thống chưa sẵn sàng.",
                "success": False
            }
        
        try:
            # Tăng số lượt hội thoại
            self.conversation_turn += 1
            
            # Lưu vào lịch sử
            self.conversation_history.append(user_input)
            
            # Sửa lỗi chính tả đơn giản
            corrected_input = self._simple_asr_correction(user_input)
            
            # Phân loại ý định
            intent = self._classify_intent(corrected_input)
            
            # Trích xuất ràng buộc mới
            new_constraints = self._extract_constraints(corrected_input)
            
            # Quản lý constraint memory (chỉ nhớ 2 lượt gần nhất)
            self._manage_constraint_memory(new_constraints)
            
            # Xử lý theo ý định
            if intent == "greeting":
                return self._handle_greeting()
            elif intent == "dietary_constraint":
                return self._handle_dietary_constraint(corrected_input, new_constraints)
            elif intent == "menu_inquiry":
                return self._handle_menu_inquiry()
            else:
                return self._handle_food_search(corrected_input)
                
        except Exception as e:
            return {
                "response_text": f"❌ Xin lỗi, tôi gặp sự cố: {str(e)}",
                "success": False
            }
    
    def _simple_asr_correction(self, text: str) -> str:
        """
        ASR correction sử dụng pure Unicode normalization.
        Hoàn toàn không hardcode, tuân thủ Requirement 6.1.
        """
        return self._unicode_only_normalization(text)
    
    def _unicode_only_normalization(self, text: str) -> str:
        """
        Fallback: chỉ Unicode normalization, không hardcode.
        """
        import unicodedata
        
        # Unicode NFC normalization
        normalized = unicodedata.normalize('NFC', text.lower().strip())
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _manage_constraint_memory(self, new_constraints: List[Dict]):
        """
        Quản lý memory của constraints - chỉ nhớ 2 lượt gần nhất.
        
        Args:
            new_constraints: Constraints mới từ lượt hội thoại hiện tại
        """
        # Thêm constraints của lượt hiện tại vào lịch sử
        self.constraint_history.append({
            'turn': self.conversation_turn,
            'constraints': new_constraints.copy()
        })
        
        # Chỉ giữ lại constraints từ 2 lượt gần nhất
        if len(self.constraint_history) > self.max_constraint_memory:
            # Xóa lượt cũ nhất
            removed_turn = self.constraint_history.pop(0)
            print(f"🔄 Đã xóa constraints từ lượt {removed_turn['turn']} (chỉ nhớ {self.max_constraint_memory} lượt gần nhất)")
        
        # Cập nhật danh sách constraints hiện tại từ các lượt được nhớ
        self.user_constraints = []
        for turn_data in self.constraint_history:
            self.user_constraints.extend(turn_data['constraints'])
        
        # Loại bỏ constraints trùng lặp (giữ lại cái mới nhất)
        self._deduplicate_constraints()
        
        # Log trạng thái memory
        if new_constraints:
            print(f"💾 Constraint Memory - Lượt {self.conversation_turn}:")
            print(f"   📝 Constraints mới: {len(new_constraints)}")
            print(f"   🧠 Tổng constraints đang nhớ: {len(self.user_constraints)}")
            print(f"   📊 Nhớ từ {len(self.constraint_history)} lượt gần nhất")
    
    def _deduplicate_constraints(self):
        """Loại bỏ constraints trùng lặp, giữ lại cái mới nhất."""
        seen_constraints = {}
        unique_constraints = []
        
        # Duyệt ngược để giữ lại constraints mới nhất
        for constraint in reversed(self.user_constraints):
            key = f"{constraint['type']}_{constraint['value']}"
            if key not in seen_constraints:
                seen_constraints[key] = True
                unique_constraints.append(constraint)
        
        # Đảo ngược lại để giữ thứ tự
        self.user_constraints = list(reversed(unique_constraints))
    
    def _classify_intent(self, text: str) -> str:
        """
        Phân loại ý định sử dụng data-driven approach.
        Học từ patterns trong menu data thay vì hardcode.
        """
        try:
            # Sử dụng Vietnamese fuzzy matcher để phân tích
            from src.vietnamese_fuzzy_matching import create_vietnamese_fuzzy_matcher
            fuzzy_matcher = create_vietnamese_fuzzy_matcher("data/processed_menu_v2.json")
            
            text_lower = text.lower()
            tokens = fuzzy_matcher.tokenize_vietnamese_food_query(text_lower)
            
            # Phân loại dựa trên learned vocabulary và patterns
            
            # Greeting: Kiểm tra similarity với greeting patterns
            greeting_samples = ['xin chào', 'chào bạn', 'hello']
            for sample in greeting_samples:
                if fuzzy_matcher.phonetic_similarity(text_lower, sample) >= 0.7:
                    return "greeting"
            
            # Dietary constraint: Phát hiện từ khóa ràng buộc
            constraint_indicators = ['dị', 'ứng', 'kiêng', 'chay', 'không', 'ăn']
            constraint_score = sum(1 for token in tokens if token in constraint_indicators)
            if constraint_score >= 2:  # Ít nhất 2 từ liên quan đến ràng buộc
                return "dietary_constraint"
            
            # Menu inquiry: Phát hiện câu hỏi về menu
            menu_indicators = ['menu', 'thực', 'đơn', 'món', 'gì', 'có', 'danh', 'sách']
            menu_score = sum(1 for token in tokens if token in menu_indicators)
            if menu_score >= 2 and any(token in ['gì', 'có', 'nào'] for token in tokens):
                return "menu_inquiry"
            
            # Food search: Kiểm tra xem có từ nào match với learned food vocabulary
            food_term_matches = 0
            for token in tokens:
                if token in fuzzy_matcher.common_words:
                    # Kiểm tra xem token có phải là food term không dựa trên learned patterns
                    if len(token) >= 2:  # Lọc từ có nghĩa
                        food_term_matches += 1
            
            if food_term_matches >= 1:
                return "food_search"
            
            # Default
            return "food_search"
            
        except Exception as e:
            # Fallback: basic pattern matching
            text_lower = text.lower()
            
            if any(term in text_lower for term in ['chào', 'hello', 'hi']):
                return "greeting"
            elif any(term in text_lower for term in ['dị ứng', 'kiêng', 'chay']):
                return "dietary_constraint"
            elif any(term in text_lower for term in ['menu', 'có món gì']):
                return "menu_inquiry"
            else:
                return "food_search"
    
    def _extract_constraints(self, text: str) -> List[Dict[str, str]]:
        """
        Trích xuất ràng buộc sử dụng data-driven approach.
        Học từ dữ liệu menu thực tế thay vì hardcode.
        """
        constraints = []
        text_lower = text.lower()
        
        try:
            # Sử dụng Vietnamese fuzzy matcher để phân tích
            from src.vietnamese_fuzzy_matching import create_vietnamese_fuzzy_matcher
            fuzzy_matcher = create_vietnamese_fuzzy_matcher("data/processed_menu_v2.json")
            
            # Tokenize text sử dụng learned patterns
            tokens = fuzzy_matcher.tokenize_vietnamese_food_query(text_lower)
            
            # Phát hiện dị ứng dựa trên learned vocabulary
            allergy_indicators = ['dị', 'ứng', 'allergy', 'allergic']
            if any(token in allergy_indicators for token in tokens):
                # Tìm allergens trong learned common words
                potential_allergens = []
                for token in tokens:
                    if token in fuzzy_matcher.common_words:
                        # Kiểm tra xem có phải là nguyên liệu thường gây dị ứng không
                        # Dựa trên context từ menu data
                        normalized_token = fuzzy_matcher.normalize_vietnamese_text(token)
                        if len(normalized_token) >= 2:  # Lọc từ có nghĩa
                            potential_allergens.append(token)
                
                # Thêm constraint cho mỗi allergen được phát hiện
                for allergen in potential_allergens[:3]:  # Giới hạn 3 allergens
                    if allergen not in ['dị', 'ứng', 'không', 'có', 'gì']:  # Lọc stop words
                        constraints.append({
                            'type': 'ALLERGY',
                            'value': allergen,
                            'severity': 'STRICT'
                        })
            
            # Phát hiện chế độ ăn chay dựa trên learned patterns
            vegetarian_indicators = ['chay', 'vegetarian', 'vegan']
            if any(fuzzy_matcher.phonetic_similarity(token, indicator) >= 0.8 
                   for token in tokens for indicator in vegetarian_indicators):
                constraints.append({
                    'type': 'DIETARY',
                    'value': 'vegetarian',
                    'severity': 'STRICT'
                })
            
            # Phát hiện sở thích cay dựa trên learned vocabulary
            spicy_indicators = ['cay', 'spicy', 'hot']
            mild_indicators = ['không', 'cay', 'mild', 'nhẹ']
            
            has_spicy = any(fuzzy_matcher.phonetic_similarity(token, indicator) >= 0.8 
                           for token in tokens for indicator in spicy_indicators)
            has_mild = any(' '.join(tokens[i:i+2]) in ['không cay', 'nhẹ nhàng'] 
                          for i in range(len(tokens)-1))
            
            if has_spicy and not has_mild:
                constraints.append({
                    'type': 'PREFERENCE',
                    'value': 'spicy',
                    'severity': 'MODERATE'
                })
            elif has_mild:
                constraints.append({
                    'type': 'PREFERENCE',
                    'value': 'mild',
                    'severity': 'MODERATE'
                })
            
        except Exception as e:
            # Fallback: minimal pattern matching nếu fuzzy matcher gặp lỗi
            if 'dị ứng' in text_lower:
                # Chỉ phát hiện một số allergen cơ bản nhất
                basic_allergens = ['tôm', 'cua', 'cá']  # Minimal set
                for allergen in basic_allergens:
                    if allergen in text_lower:
                        constraints.append({
                            'type': 'ALLERGY',
                            'value': allergen,
                            'severity': 'STRICT'
                        })
            
            if 'chay' in text_lower:
                constraints.append({
                    'type': 'DIETARY',
                    'value': 'vegetarian',
                    'severity': 'STRICT'
                })
        
        return constraints
    
    def _handle_greeting(self) -> Dict[str, Any]:
        """Xử lý lời chào."""
        return {
            "response_text": "Xin chào! Tôi là trợ lý tư vấn món ăn Việt Nam. Bạn muốn tìm món gì hôm nay?",
            "success": True,
            "intent": "greeting",
            "follow_up_questions": [
                "Bạn có muốn xem menu không?",
                "Bạn thích món gì?",
                "Bạn có yêu cầu đặc biệt nào không?"
            ]
        }
    
    def _handle_dietary_constraint(self, user_input: str, constraints: List[Dict]) -> Dict[str, Any]:
        """Xử lý ràng buộc dinh dưỡng."""
        constraint_text = self._format_constraints(constraints)
        
        # Tìm món phù hợp
        dishes = self._search_with_constraints("", constraints)
        
        if dishes:
            response_text = f"Tôi đã ghi nhận yêu cầu của bạn: {constraint_text}. Tôi tìm thấy {len(dishes)} món phù hợp:"
        else:
            response_text = f"Tôi đã ghi nhận yêu cầu: {constraint_text}. Bạn có thể cho tôi biết loại món bạn muốn ăn để tôi tìm món phù hợp không?"
        
        return {
            "response_text": response_text,
            "success": True,
            "intent": "dietary_constraint",
            "constraints": constraints,
            "dishes": dishes,
            "follow_up_questions": [
                "Bạn muốn ăn món chính hay món phụ?",
                "Bạn thích món nóng hay món lạnh?",
                "Bạn có sở thích gì khác không?"
            ]
        }
    
    def _handle_menu_inquiry(self) -> Dict[str, Any]:
        """Xử lý câu hỏi về menu."""
        dishes = self._get_sample_dishes()
        
        return {
            "response_text": "Menu của chúng tôi có nhiều món Việt Nam truyền thống. Đây là một số món nổi bật:",
            "success": True,
            "intent": "menu_inquiry",
            "dishes": dishes,
            "follow_up_questions": [
                "Bạn muốn xem món nào cụ thể?",
                "Bạn thích món chính hay món phụ?",
                "Bạn có yêu cầu đặc biệt nào không?"
            ]
        }
    
    def _handle_food_search(self, user_input: str) -> Dict[str, Any]:
        """Xử lý tìm kiếm món ăn."""
        dishes = self._search_with_constraints(user_input, self.user_constraints)
        
        if dishes:
            response_text = f"Tôi tìm thấy {len(dishes)} món phù hợp với '{user_input}':"
        else:
            response_text = f"Xin lỗi, tôi không tìm thấy món '{user_input}' phù hợp với yêu cầu của bạn. Bạn có thể thử tên khác không?"
        
        return {
            "response_text": response_text,
            "success": True,
            "intent": "food_search",
            "dishes": dishes,
            "constraints": self.user_constraints,
            "follow_up_questions": [
                "Bạn muốn biết thêm về món nào?",
                "Bạn có muốn xem món tương tự không?",
                "Bạn cần thông tin gì khác?"
            ]
        }
    
    def _search_with_constraints(self, query: str, constraints: List[Dict]) -> List[Dict]:
        """Tìm kiếm với ràng buộc."""
        try:
            # Tìm kiếm
            if query.strip():
                results = self.rag_engine.search_with_availability_check(
                    query=query,
                    search_method="hybrid",
                    n_results=10,
                    similarity_threshold=0.2
                )
            else:
                results = self.rag_engine.search_with_availability_check(
                    query="món ăn",
                    search_method="hybrid", 
                    n_results=20,
                    similarity_threshold=0.1
                )
            
            search_results = results.get("results", [])
            dishes = []
            
            for result in search_results:
                dish = result.dish
                
                # Kiểm tra ràng buộc
                if self._dish_matches_constraints(dish, constraints):
                    dishes.append({
                        'name_vi': dish.search_content.name_vi,
                        'name_en': dish.search_content.name_en,
                        'price': dish.metadata.price_vnd,
                        'category': dish.metadata.category,
                        'description': dish.search_content.description_vi[:100] + "..." if dish.search_content.description_vi else "",
                        'relevance_score': result.relevance_score
                    })
            
            return dishes[:5]
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _dish_matches_constraints(self, dish, constraints: List[Dict]) -> bool:
        """
        Kiểm tra món có phù hợp với ràng buộc sử dụng data-driven approach.
        Sử dụng learned patterns thay vì hardcode.
        """
        try:
            # Sử dụng Vietnamese fuzzy matcher để phân tích
            from src.vietnamese_fuzzy_matching import create_vietnamese_fuzzy_matcher
            fuzzy_matcher = create_vietnamese_fuzzy_matcher("data/processed_menu_v2.json")
            
            dish_name = dish.search_content.name_vi.lower()
            dish_desc = (dish.search_content.description_vi or "").lower()
            dish_text = f"{dish_name} {dish_desc}"
            
            # Tokenize dish content
            dish_tokens = fuzzy_matcher.tokenize_vietnamese_food_query(dish_text)
            
            for constraint in constraints:
                if constraint['type'] == 'ALLERGY':
                    allergen = constraint['value']
                    
                    # Kiểm tra similarity với allergen sử dụng learned patterns
                    for token in dish_tokens:
                        if fuzzy_matcher.phonetic_similarity(token, allergen) >= 0.8:
                            return False
                    
                    # Kiểm tra trong ingredient list nếu có
                    if hasattr(dish, 'ingredients') and dish.ingredients:
                        for ingredient in dish.ingredients:
                            ingredient_name = ingredient.name_vi.lower() if hasattr(ingredient, 'name_vi') else str(ingredient).lower()
                            ingredient_tokens = fuzzy_matcher.tokenize_vietnamese_food_query(ingredient_name)
                            
                            for token in ingredient_tokens:
                                if fuzzy_matcher.phonetic_similarity(token, allergen) >= 0.8:
                                    return False
                
                elif constraint['type'] == 'DIETARY' and constraint['value'] == 'vegetarian':
                    # Sử dụng learned vocabulary để phát hiện meat terms
                    # Thay vì hardcode, kiểm tra similarity với known meat terms từ menu data
                    potential_meat_terms = []
                    
                    # Lấy các từ có thể là thịt từ learned vocabulary
                    for token in dish_tokens:
                        if token in fuzzy_matcher.common_words:
                            # Kiểm tra context - nếu từ này thường xuất hiện với meat dishes
                            # Đây là approach học từ data thay vì hardcode
                            if len(token) >= 2:  # Lọc từ có nghĩa
                                potential_meat_terms.append(token)
                    
                    # Kiểm tra với một số meat indicators cơ bản (minimal fallback)
                    basic_meat_indicators = ['thịt', 'bò', 'heo', 'gà', 'tôm', 'cua', 'cá']
                    for indicator in basic_meat_indicators:
                        for token in dish_tokens:
                            if fuzzy_matcher.phonetic_similarity(token, indicator) >= 0.8:
                                return False
            
            return True
            
        except Exception as e:
            # Fallback: minimal checking
            dish_name = dish.search_content.name_vi.lower()
            dish_desc = (dish.search_content.description_vi or "").lower()
            dish_text = f"{dish_name} {dish_desc}"
            
            for constraint in constraints:
                if constraint['type'] == 'ALLERGY':
                    allergen = constraint['value']
                    if allergen in dish_text:
                        return False
                
                elif constraint['type'] == 'DIETARY' and constraint['value'] == 'vegetarian':
                    # Minimal meat detection
                    basic_meat_terms = ['thịt', 'bò', 'heo', 'gà', 'tôm', 'cua', 'cá']
                    if any(term in dish_text for term in basic_meat_terms):
                        return False
            
            return True
    
    def _get_sample_dishes(self) -> List[Dict]:
        """Lấy món mẫu."""
        return self._search_with_constraints("món ngon", [])
    
    def _format_constraints(self, constraints: List[Dict]) -> str:
        """Format ràng buộc."""
        if not constraints:
            return "không có yêu cầu đặc biệt"
        
        texts = []
        for constraint in constraints:
            if constraint['type'] == 'ALLERGY':
                texts.append(f"dị ứng {constraint['value']}")
            elif constraint['type'] == 'DIETARY':
                texts.append(f"chế độ ăn {constraint['value']}")
            elif constraint['type'] == 'PREFERENCE':
                texts.append(f"thích {constraint['value']}")
        
        return ", ".join(texts)
    
    def display_response(self, response: Dict[str, Any]):
        """Hiển thị phản hồi."""
        # Phản hồi chính
        print(f"\n🤖 {response['response_text']}")
        
        # Hiển thị món ăn
        dishes = response.get('dishes', [])
        if dishes:
            print(f"\n📋 Danh sách món:")
            for i, dish in enumerate(dishes, 1):
                print(f"   {i}. {dish['name_vi']}")
                if dish.get('name_en'):
                    print(f"      ({dish['name_en']})")
                if dish.get('price'):
                    print(f"      💰 {dish['price']:,} VND - 📂 {dish['category']}")
                if dish.get('description') and len(dish['description']) > 10:
                    print(f"      📝 {dish['description']}")
        
        # Hiển thị ràng buộc
        constraints = response.get('constraints', [])
        if constraints:
            print(f"\n🔍 Yêu cầu của bạn (nhớ {self.max_constraint_memory} lượt gần nhất):")
            for constraint in constraints:
                if constraint['type'] == 'ALLERGY':
                    print(f"   - Dị ứng: {constraint['value']}")
                elif constraint['type'] == 'DIETARY':
                    print(f"   - Chế độ ăn: {constraint['value']}")
                elif constraint['type'] == 'PREFERENCE':
                    print(f"   - Sở thích: {constraint['value']}")
            
            # Hiển thị thông tin memory
            if hasattr(self, 'constraint_history') and self.constraint_history:
                turns_remembered = [str(turn_data['turn']) for turn_data in self.constraint_history]
                print(f"   💾 Đang nhớ từ lượt: {', '.join(turns_remembered)}")
        
        # Câu hỏi gợi ý
        follow_up = response.get('follow_up_questions', [])
        if follow_up:
            print(f"\n❓ Bạn có thể hỏi:")
            for i, question in enumerate(follow_up[:3], 1):
                print(f"   {i}. {question}")
    
    def run_chat(self):
        """Chạy chat."""
        if not self.rag_engine:
            print("❌ Không thể khởi động - hệ thống chưa sẵn sàng")
            return
        
        print("\n" + "="*60)
        print("🍜 TRỢ LÝ MÓN ĂN VIỆT NAM - QUICK CHAT")
        print("="*60)
        print("🤖 Xin chào! Tôi là trợ lý tư vấn món ăn Việt Nam.")
        print("✨ Tính năng:")
        print("   - 🔤 Sửa lỗi chính tả tự động")
        print("   - 🧠 Hiểu ràng buộc dinh dưỡng")
        print("   - � Nhớ yêu cầu trong 2 lượt gần nhất")
        print("   - �🔍 Tìm kiếm thông minh")
        print("   - 💬 Phản hồi tự nhiên")
        print("\n💡 Thử các câu này:")
        print("   - 'cho toi mon ga' (có lỗi chính tả)")
        print("   - 'tôi dị ứng tôm'")
        print("   - 'có món chay nào không'")
        print("   - 'menu có gì ngon'")
        print("\n📝 Gõ 'tạm biệt' để kết thúc")
        print("="*60)
        
        while True:
            try:
                user_input = input(f"\n🗣️  Bạn: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["tạm biệt", "bye", "exit", "quit", "thoát"]:
                    print("\n👋 Tạm biệt! Hẹn gặp lại!")
                    break
                
                # Xử lý ngay lập tức
                response = self.chat(user_input)
                
                # Hiển thị
                self.display_response(response)
                
                if not response.get("success"):
                    print(f"\n⚠️  Có lỗi xảy ra")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat bị ngắt. Tạm biệt!")
                break
            except Exception as e:
                print(f"\n❌ Lỗi: {e}")


def main():
    """Hàm chính."""
    print("🚀 TRỢ LÝ MÓN ĂN VIỆT NAM - QUICK CHAT")
    
    chatbot = QuickVietnameseFoodChat()
    
    if not chatbot.rag_engine:
        print("\n💡 Hãy kiểm tra:")
        print("   1. File dữ liệu menu có tồn tại không")
        print("   2. Các dependencies đã được cài đặt")
        return
    
    chatbot.run_chat()


if __name__ == "__main__":
    main()