#!/usr/bin/env python3
"""
Test Script for Phase 1 Enhanced Japanese Quiz Solver Integration

Tests all Phase 1 components:
1. OCR preprocessing and multi-PSM optimization
2. Date/reading rule engine functionality  
3. Katakana fuzzy matching accuracy
4. Unified rule engine coordination
5. Structured LLM JSON response validation
6. End-to-end integration testing

This script validates the accuracy improvements and performance gains
from the Phase 1 enhancements.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import logging
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import test modules
try:
    from main_enhanced import EnhancedQuizSolver
    from ocr.ocr_preprocess import preprocess_image_for_ocr
    from ocr.ocr_multi_psm import best_ocr_result
    from rules.rules_engine import UnifiedRuleEngine
    from rules.rules_date import DateReadingRuleEngine
    from rules.fuzzy_kata import KatakanaFuzzyMatcher
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Ensure all Phase 1 modules are properly installed.")
    sys.exit(1)

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase1IntegrationTest:
    """Comprehensive test suite for Phase 1 enhancements"""
    
    def __init__(self):
        self.solver = EnhancedQuizSolver()
        self.rule_engine = UnifiedRuleEngine()
        self.date_engine = DateReadingRuleEngine()
        self.fuzzy_matcher = KatakanaFuzzyMatcher()
        
        # Test results storage
        self.test_results = {
            "ocr_tests": [],
            "rule_engine_tests": [],
            "integration_tests": [],
            "performance_tests": []
        }
    
    def create_test_image(self, text: str, size: tuple = (400, 200)) -> Image.Image:
        """Create a test image with Japanese text"""
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a Japanese font, fallback to default
        try:
            # Common Japanese font paths
            font_paths = [
                "C:/Windows/Fonts/msgothic.ttc",
                "C:/Windows/Fonts/msmincho.ttc", 
                "/System/Library/Fonts/Hiragino Sans GB.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ]
            
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, 24)
                        break
                    except:
                        continue
            
            if font is None:
                font = ImageFont.load_default()
                
        except Exception:
            font = ImageFont.load_default()
        
        # Draw text in center
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2
        
        draw.text((x, y), text, fill='black', font=font)
        return img
    
    def test_ocr_preprocessing(self) -> Dict[str, Any]:
        """Test OCR preprocessing improvements"""
        print("🔍 Testing OCR preprocessing...")
        
        test_cases = [
            "今日は何曜日ですか？ 1.月曜日 2.火曜日 3.水曜日 4.木曜日",
            "カタカナのテスト: ア.アメリカ イ.イギリス ウ.フランス エ.ドイツ", 
            "漢字の読み方: A.あした B.きのう C.きょう D.あさって",
            "昭和55年12月25日",
            "平成元年4月1日"
        ]
        
        results = []
        
        for i, text in enumerate(test_cases):
            print(f"  Testing case {i+1}: {text[:30]}...")
            
            # Create test image
            test_img = self.create_test_image(text)
            
            # Test without preprocessing
            start_time = time.time()
            basic_text = ""
            try:
                import pytesseract
                basic_text = pytesseract.image_to_string(test_img, lang='jpn').strip()
            except Exception as e:
                basic_text = f"Error: {e}"
            basic_time = time.time() - start_time
            
            # Test with preprocessing
            start_time = time.time()
            preprocessed_img = preprocess_image_for_ocr(test_img)
            enhanced_result = best_ocr_result(preprocessed_img)
            enhanced_time = time.time() - start_time
            
            enhanced_text = enhanced_result.get('text', '') if enhanced_result else ''
            
            # Calculate accuracy metrics
            original_chars = set(text)
            basic_chars = set(basic_text)
            enhanced_chars = set(enhanced_text)
            
            basic_accuracy = len(original_chars & basic_chars) / len(original_chars) if original_chars else 0
            enhanced_accuracy = len(original_chars & enhanced_chars) / len(original_chars) if original_chars else 0
            
            result = {
                "test_case": i + 1,
                "original_text": text,
                "basic_ocr": basic_text,
                "enhanced_ocr": enhanced_text,
                "basic_accuracy": basic_accuracy,
                "enhanced_accuracy": enhanced_accuracy,
                "basic_time": basic_time,
                "enhanced_time": enhanced_time,
                "improvement": enhanced_accuracy - basic_accuracy,
                "psm_used": enhanced_result.get('psm_used', 'N/A') if enhanced_result else 'N/A'
            }
            
            results.append(result)
            
            print(f"    Basic: {basic_accuracy:.2f} accuracy, Enhanced: {enhanced_accuracy:.2f} accuracy")
            print(f"    Improvement: {result['improvement']:.2f}")
        
        self.test_results["ocr_tests"] = results
        return {"ocr_results": results}
    
    def test_rule_engines(self) -> Dict[str, Any]:
        """Test rule engines functionality"""
        print("🔧 Testing rule engines...")
        
        # Date/reading rule tests
        date_tests = [
            ("昭和55年12月25日", "1980年12月25日"),
            ("平成元年4月1日", "1989年4月1日"),
            ("令和3年10月15日", "2021年10月15日"),
            ("おとうさん", "父"),
            ("おかあさん", "母"),
            ("きのう", "昨日"),
            ("あした", "明日")
        ]
        
        date_results = []
        for input_text, expected in date_tests:
            result = self.date_engine.find_matches(input_text)
            matched = len(result) > 0
            correct = any(match['target_text'] == expected for match in result) if matched else False
            
            date_results.append({
                "input": input_text,
                "expected": expected,
                "matched": matched,
                "correct": correct,
                "matches": result
            })
            
            print(f"  Date test '{input_text}': {'✅' if correct else '❌'}")
        
        # Katakana fuzzy matching tests
        fuzzy_tests = [
            ("アメリヵ", "アメリカ"),  # OCR error: small ka instead of large ka
            ("コンビユータ", "コンピュータ"),  # OCR error: yu instead of pu 
            ("テレビジヨン", "テレビジョン"),  # OCR error: yo instead of jo
            ("スマートフオン", "スマートフォン"),  # OCR error: o instead of fo
            ("インタ一ネツト", "インターネット")  # OCR errors: long vowel mark, tsu
        ]
        
        fuzzy_results = []
        for input_text, expected in fuzzy_tests:
            result = self.fuzzy_matcher.find_fuzzy_matches(input_text)
            matched = len(result) > 0
            correct = any(abs(match['similarity'] - 1.0) < 0.1 and expected in match['candidate'] 
                         for match in result) if matched else False
            
            fuzzy_results.append({
                "input": input_text,
                "expected": expected,
                "matched": matched,
                "correct": correct,
                "matches": result
            })
            
            print(f"  Fuzzy test '{input_text}': {'✅' if correct else '❌'}")
        
        # Unified rule engine tests
        unified_tests = [
            "昭和55年12月25日はいつですか？ A.1979年 B.1980年 C.1981年 D.1982年",
            "アメリヵはどこですか？ 1.ヨーロッパ 2.アジア 3.アメリカ 4.アフリカ",
            "おとうさんの漢字は？ ア.父 イ.母 ウ.兄 エ.姉"
        ]
        
        unified_results = []
        for test_text in unified_tests:
            result = self.rule_engine.process_text(test_text)
            
            unified_results.append({
                "input": test_text,
                "matches_found": result.get('matches_found', False),
                "best_match": result.get('best_match'),
                "confidence": result.get('confidence', 0.0),
                "should_override": result.get('should_override_llm', False),
                "details": result
            })
            
            status = "✅" if result.get('matches_found', False) else "❌"
            print(f"  Unified test: {status} (conf: {result.get('confidence', 0.0):.2f})")
        
        rule_results = {
            "date_results": date_results,
            "fuzzy_results": fuzzy_results, 
            "unified_results": unified_results
        }
        
        self.test_results["rule_engine_tests"] = rule_results
        return rule_results
    
    def test_llm_structured_responses(self) -> Dict[str, Any]:
        """Test structured LLM JSON responses"""
        print("🤖 Testing LLM structured responses...")
        
        test_cases = [
            "今日は何曜日ですか？ 1.月曜日 2.火曜日 3.水曜日 4.木曜日",
            "次の単語の意味は？ 友達 A.family B.friend C.teacher D.student",
            "正しい読み方を選んでください。 昨日 ア.きのう イ.あした ウ.きょう エ.おととい"
        ]
        
        llm_results = []
        
        for i, test_text in enumerate(test_cases):
            print(f"  Testing LLM case {i+1}...")
            
            try:
                # Test Gemini structured response
                start_time = time.time()
                gemini_result = self.solver.get_answer_from_gemini(test_text)
                gemini_time = time.time() - start_time
                
                # Validate JSON structure
                required_fields = [
                    'question_detected', 'question_type', 'japanese_text',
                    'correct_answer', 'confidence', 'translation', 
                    'explanation', 'difficulty_level', 'reasoning_steps'
                ]
                
                gemini_valid = all(field in gemini_result for field in required_fields)
                gemini_json_valid = isinstance(gemini_result, dict)
                
                result = {
                    "test_case": i + 1,
                    "input": test_text,
                    "gemini_response": gemini_result,
                    "gemini_time": gemini_time,
                    "gemini_valid_structure": gemini_valid,
                    "gemini_valid_json": gemini_json_valid,
                    "question_detected": gemini_result.get('question_detected', False),
                    "confidence": gemini_result.get('confidence', 0.0),
                    "answer_provided": bool(gemini_result.get('correct_answer'))
                }
                
                status = "✅" if gemini_valid and gemini_json_valid else "❌"
                print(f"    Gemini: {status} (conf: {result['confidence']:.2f})")
                
            except Exception as e:
                result = {
                    "test_case": i + 1,
                    "input": test_text,
                    "error": str(e),
                    "gemini_valid_structure": False,
                    "gemini_valid_json": False
                }
                print(f"    Gemini: ❌ Error: {e}")
            
            llm_results.append(result)
        
        self.test_results["llm_tests"] = llm_results
        return {"llm_results": llm_results}
    
    def test_end_to_end_integration(self) -> Dict[str, Any]:
        """Test complete end-to-end integration"""
        print("🔄 Testing end-to-end integration...")
        
        # Create comprehensive test cases
        integration_tests = [
            {
                "text": "昭和55年12月25日は何年ですか？ A.1979年 B.1980年 C.1981年 D.1982年",
                "expected_answer": "B",
                "expected_source": "rule_engine",
                "test_type": "date_rule"
            },
            {
                "text": "アメリヵの首都は？ 1.ニューヨーク 2.ワシントン 3.ロサンゼルス 4.シカゴ",
                "expected_answer": "2",
                "expected_source": "llm_primary",  # Fuzzy match + LLM
                "test_type": "fuzzy_katakana"
            },
            {
                "text": "次の文を読んでください。私は学生です。A.がくせい B.せんせい C.かいしゃいん D.いしゃ",
                "expected_answer": "A", 
                "expected_source": "llm_primary",
                "test_type": "standard_quiz"
            }
        ]
        
        integration_results = []
        
        for i, test in enumerate(integration_tests):
            print(f"  Integration test {i+1} ({test['test_type']})...")
            
            # Create test image
            test_img = self.create_test_image(test["text"])
            
            # Run full pipeline
            start_time = time.time()
            result = self.solver.process_image(test_img)
            processing_time = time.time() - start_time
            
            # Evaluate results
            final_answer = result.get('final_answer')
            confidence = result.get('confidence', 0.0)
            source = result.get('source', 'unknown')
            
            correct_answer = final_answer == test['expected_answer']
            correct_source = source == test['expected_source']
            
            test_result = {
                "test_case": i + 1,
                "test_type": test['test_type'],
                "input_text": test['text'],
                "expected_answer": test['expected_answer'],
                "expected_source": test['expected_source'],
                "actual_answer": final_answer,
                "actual_source": source,
                "confidence": confidence,
                "processing_time": processing_time,
                "correct_answer": correct_answer,
                "correct_source": correct_source,
                "overall_correct": correct_answer and confidence > 0.5,
                "full_result": result
            }
            
            integration_results.append(test_result)
            
            status = "✅" if test_result['overall_correct'] else "❌"
            print(f"    Result: {status} Answer: {final_answer}, Source: {source}")
            print(f"    Time: {processing_time:.2f}s, Confidence: {confidence:.2f}")
        
        self.test_results["integration_tests"] = integration_results
        return {"integration_results": integration_results}
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance improvements"""
        print("⚡ Testing performance benchmarks...")
        
        # Test processing speed across different complexity levels
        performance_tests = [
            ("Simple", "今日は月曜日です。"),
            ("Medium", "昭和55年12月25日は何年ですか？ A.1979年 B.1980年 C.1981年 D.1982年"),
            ("Complex", "次の文章を読んで、正しい答えを選んでください。私は毎日学校に行きます。友達と一緒に勉強します。A.学校 B.仕事 C.家 D.公園"),
            ("Mixed Script", "私のfavorite foodは寿司です。1.すし 2.ラーメン 3.カレー 4.ピザ")
        ]
        
        performance_results = []
        
        for complexity, text in performance_tests:
            print(f"  Testing {complexity} complexity...")
            
            # Run multiple iterations for average timing
            times = []
            results = []
            
            for iteration in range(3):  # 3 iterations for averaging
                test_img = self.create_test_image(text)
                
                start_time = time.time()
                result = self.solver.process_image(test_img)
                end_time = time.time()
                
                processing_time = end_time - start_time
                times.append(processing_time)
                results.append(result)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            # Check consistency of results
            final_answers = [r.get('final_answer') for r in results]
            consistent = len(set(final_answers)) <= 1  # All same or mostly same
            
            perf_result = {
                "complexity": complexity,
                "text": text,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "consistent_results": consistent,
                "sample_result": results[0] if results else None
            }
            
            performance_results.append(perf_result)
            
            print(f"    Avg time: {avg_time:.2f}s, Consistent: {'✅' if consistent else '❌'}")
        
        self.test_results["performance_tests"] = performance_results
        return {"performance_results": performance_results}
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        report_lines = [
            "=" * 80,
            "🏮 ULTIMATE JAPANESE QUIZ SOLVER - PHASE 1 TEST REPORT",
            "=" * 80,
            "",
            f"📅 Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"🔧 Test Components: OCR, Rule Engines, LLM Integration, Performance",
            "",
            "📊 SUMMARY STATISTICS:",
            "=" * 40
        ]
        
        # OCR Test Summary
        ocr_results = self.test_results.get("ocr_tests", [])
        if ocr_results:
            improvements = [r["improvement"] for r in ocr_results]
            avg_improvement = sum(improvements) / len(improvements)
            positive_improvements = sum(1 for i in improvements if i > 0)
            
            report_lines.extend([
                "",
                f"🔍 OCR PREPROCESSING TESTS: {len(ocr_results)} cases",
                f"   • Average accuracy improvement: {avg_improvement:.3f}",
                f"   • Cases with improvement: {positive_improvements}/{len(ocr_results)}",
                f"   • Success rate: {positive_improvements/len(ocr_results)*100:.1f}%"
            ])
        
        # Rule Engine Test Summary
        rule_tests = self.test_results.get("rule_engine_tests", {})
        date_results = rule_tests.get("date_results", [])
        fuzzy_results = rule_tests.get("fuzzy_results", [])
        unified_results = rule_tests.get("unified_results", [])
        
        if date_results or fuzzy_results:
            date_correct = sum(1 for r in date_results if r["correct"])
            fuzzy_correct = sum(1 for r in fuzzy_results if r["correct"])
            unified_matches = sum(1 for r in unified_results if r["matches_found"])
            
            report_lines.extend([
                "",
                f"🔧 RULE ENGINE TESTS:",
                f"   • Date/Reading Rules: {date_correct}/{len(date_results)} correct ({date_correct/len(date_results)*100:.1f}%)" if date_results else "",
                f"   • Fuzzy Katakana: {fuzzy_correct}/{len(fuzzy_results)} correct ({fuzzy_correct/len(fuzzy_results)*100:.1f}%)" if fuzzy_results else "",
                f"   • Unified Engine: {unified_matches}/{len(unified_results)} matches found" if unified_results else ""
            ])
        
        # Integration Test Summary
        integration_results = self.test_results.get("integration_tests", [])
        if integration_results:
            correct_answers = sum(1 for r in integration_results if r["correct_answer"])
            high_confidence = sum(1 for r in integration_results if r["confidence"] > 0.7)
            
            report_lines.extend([
                "",
                f"🔄 INTEGRATION TESTS: {len(integration_results)} cases",
                f"   • Correct answers: {correct_answers}/{len(integration_results)} ({correct_answers/len(integration_results)*100:.1f}%)",
                f"   • High confidence (>0.7): {high_confidence}/{len(integration_results)} ({high_confidence/len(integration_results)*100:.1f}%)"
            ])
        
        # Performance Summary
        performance_results = self.test_results.get("performance_tests", [])
        if performance_results:
            avg_times = [r["avg_time"] for r in performance_results]
            overall_avg_time = sum(avg_times) / len(avg_times)
            consistent_results = sum(1 for r in performance_results if r["consistent_results"])
            
            report_lines.extend([
                "",
                f"⚡ PERFORMANCE TESTS: {len(performance_results)} complexity levels",
                f"   • Average processing time: {overall_avg_time:.2f}s",
                f"   • Consistent results: {consistent_results}/{len(performance_results)} ({consistent_results/len(performance_results)*100:.1f}%)"
            ])
        
        report_lines.extend([
            "",
            "=" * 80,
            "✅ PHASE 1 INTEGRATION TEST COMPLETED",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def save_detailed_results(self, filename: str = "phase1_test_results.json"):
        """Save detailed test results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"💾 Detailed results saved to: {filename}")
    
    def run_all_tests(self):
        """Run all test suites"""
        print("🧪 Starting Phase 1 Integration Tests...")
        print("=" * 60)
        
        # Run all test suites
        self.test_ocr_preprocessing()
        print()
        self.test_rule_engines() 
        print()
        self.test_llm_structured_responses()
        print()
        self.test_end_to_end_integration()
        print()
        self.test_performance_benchmarks()
        print()
        
        # Generate and display report
        report = self.generate_test_report()
        print(report)
        
        # Save detailed results
        self.save_detailed_results()
        
        return self.test_results

def main():
    """Main test runner"""
    print("🏮 Ultimate Japanese Quiz Solver - Phase 1 Integration Tests")
    print("Testing enhanced OCR, rule engines, and structured LLM responses")
    print()
    
    # Initialize and run tests
    tester = Phase1IntegrationTest()
    
    try:
        results = tester.run_all_tests()
        print("\n🎉 All tests completed successfully!")
        return results
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
