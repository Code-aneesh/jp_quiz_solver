#!/usr/bin/env python3
"""
🧪 JAPANESE QUIZ SOLVER ACCURACY TEST 🧪
Comprehensive testing script to validate answer accuracy and performance.
"""

import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import our solver
from ultimate_main import UltimateQuizSolver

class AccuracyTester:
    """Test suite for Japanese quiz solver accuracy"""
    
    def __init__(self):
        self.solver = UltimateQuizSolver()
        self.test_results = []
        
    def run_test_case(self, question_text: str, expected_answer: str, test_name: str) -> Dict:
        """Run a single test case and return results"""
        print(f"\n🧪 Testing: {test_name}")
        print(f"📝 Question: {question_text}")
        print(f"✅ Expected: {expected_answer}")
        
        start_time = time.time()
        
        # Get AI answer
        ai_response, provider, processing_time = self.solver.get_ai_answer(question_text)
        
        # Extract actual answer from AI response
        actual_answer = self.extract_answer_from_response(ai_response)
        
        # Check if correct
        is_correct = self.is_answer_correct(actual_answer, expected_answer)
        
        total_time = time.time() - start_time
        
        result = {
            "test_name": test_name,
            "question": question_text,
            "expected": expected_answer,
            "actual": actual_answer,
            "ai_response": ai_response,
            "is_correct": is_correct,
            "processing_time": processing_time,
            "total_time": total_time,
            "provider": provider
        }
        
        self.test_results.append(result)
        
        status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        print(f"🎯 Actual: {actual_answer}")
        print(f"⚡ Result: {status} ({processing_time:.2f}s)")
        
        return result
    
    def extract_answer_from_response(self, response: str) -> str:
        """Extract the answer from AI response"""
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('🎯 ANSWER:'):
                answer = line.replace('🎯 ANSWER:', '').strip()
                # Clean up answer format
                answer = answer.split(',')[0].strip()  # Take first answer if multiple
                answer = answer.split(' ')[0].strip()  # Take first part
                return answer
        return "NO_ANSWER_FOUND"
    
    def is_answer_correct(self, actual: str, expected: str) -> bool:
        """Check if the actual answer matches expected"""
        actual_clean = actual.strip().lower()
        expected_clean = expected.strip().lower()
        
        # Direct match
        if actual_clean == expected_clean:
            return True
        
        # Check if actual contains expected (for multiple choice symbols)
        if expected_clean in actual_clean or actual_clean in expected_clean:
            return True
        
        # Check for common variations
        variations = {
            '①': ['1', 'ichi', 'first'],
            '②': ['2', 'ni', 'second'],
            '③': ['3', 'san', 'third'],
            '④': ['4', 'yon', 'fourth'],
            '⑤': ['5', 'go', 'fifth']
        }
        
        for symbol, alts in variations.items():
            if (expected_clean == symbol and actual_clean in alts) or \
               (actual_clean == symbol and expected_clean in alts):
                return True
        
        return False
    
    def run_comprehensive_tests(self):
        """Run comprehensive test suite"""
        print("\n🎯 ULTIMATE JAPANESE QUIZ SOLVER - ACCURACY TEST SUITE 🎯")
        print("=" * 70)
        
        # Test cases based on common JLPT patterns
        test_cases = [
            # Basic vocabulary tests
            {
                "question": "ははと　やまに　のぼりました。(16). はは　①.娠　②.卵　③.奴　④.母",
                "expected": "④",
                "test_name": "Basic Vocabulary - Mother (母)"
            },
            {
                "question": "ははと　やまに　のぼりました。(17). やま　①.上　②.山　③.止　④.凸",
                "expected": "②",
                "test_name": "Basic Vocabulary - Mountain (山)"
            },
            {
                "question": "こんしゅうは　てんきが　よかった。(18). こんしゅう　①.今週　②.今過　③.令週　④.今習",
                "expected": "①",
                "test_name": "Time Expression - This Week (今週)"
            },
            {
                "question": "こんしゅうは　てんきが　よかった。(19). てんき　①.天気　②.天汽　③.矢気　④.転気",
                "expected": "①",
                "test_name": "Weather - Weather (天気)"
            },
            {
                "question": "その　ちいさい　かれんだーを　ください。(20). ちいさい　①.小い　②.小さい　③.少い　④.少さい",
                "expected": "②",
                "test_name": "Adjective - Small (小さい)"
            },
            {
                "question": "その　ちいさい　かれんだーを　ください。(21). かれんだー　①.カトングー　②.カトンダー　③.カレングー　④.カレンダー",
                "expected": "④",
                "test_name": "Katakana Loanword - Calendar (カレンダー)"
            },
            {
                "question": "ひがしの　そらが　きれいです。(22). ひがし　①.西　②.東　③.南　④.北",
                "expected": "②",
                "test_name": "Direction - East (東)"
            },
            {
                "question": "ひがしの　そらが　きれいです。(23). そら　①.川　②.池　③.空　④.風",
                "expected": "③",
                "test_name": "Nature - Sky (空)"
            },
            {
                "question": "むいかの　ごごに　あいましょう。(24). むいか　①.九日　②.三日　③.六日　④.五日",
                "expected": "③",
                "test_name": "Date - Sixth Day (六日)"
            }
        ]
        
        # Run all tests
        for test_case in test_cases:
            self.run_test_case(
                test_case["question"],
                test_case["expected"],
                test_case["test_name"]
            )
            time.sleep(1)  # Prevent API rate limiting
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 70)
        print("🎯 TEST RESULTS SUMMARY")
        print("=" * 70)
        
        total_tests = len(self.test_results)
        correct_answers = sum(1 for r in self.test_results if r["is_correct"])
        accuracy_rate = (correct_answers / total_tests) * 100 if total_tests > 0 else 0
        
        avg_processing_time = sum(r["processing_time"] for r in self.test_results) / total_tests if total_tests > 0 else 0
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Correct Answers: {correct_answers}")
        print(f"❌ Incorrect Answers: {total_tests - correct_answers}")
        print(f"🎯 Accuracy Rate: {accuracy_rate:.1f}%")
        print(f"⚡ Average Processing Time: {avg_processing_time:.2f}s")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 70)
        
        for i, result in enumerate(self.test_results, 1):
            status_icon = "✅" if result["is_correct"] else "❌"
            print(f"{i:2d}. {status_icon} {result['test_name']}")
            print(f"    Expected: {result['expected']} | Actual: {result['actual']} | Time: {result['processing_time']:.2f}s")
        
        # Failed tests analysis
        failed_tests = [r for r in self.test_results if not r["is_correct"]]
        if failed_tests:
            print(f"\n❌ FAILED TESTS ANALYSIS:")
            print("-" * 70)
            
            for test in failed_tests:
                print(f"\n🔍 {test['test_name']}")
                print(f"   Question: {test['question'][:60]}...")
                print(f"   Expected: {test['expected']}")
                print(f"   Actual: {test['actual']}")
                print(f"   AI Response Preview: {test['ai_response'][:100]}...")
        
        # Performance analysis
        print(f"\n⚡ PERFORMANCE ANALYSIS:")
        print("-" * 70)
        
        fastest = min(self.test_results, key=lambda r: r["processing_time"])
        slowest = max(self.test_results, key=lambda r: r["processing_time"])
        
        print(f"🚀 Fastest Test: {fastest['test_name']} ({fastest['processing_time']:.2f}s)")
        print(f"🐌 Slowest Test: {slowest['test_name']} ({slowest['processing_time']:.2f}s)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 70)
        
        if accuracy_rate < 80:
            print("⚠️  CRITICAL: Accuracy below 80% - Major improvements needed!")
            print("   - Review AI prompt engineering")
            print("   - Improve OCR preprocessing")
            print("   - Add context validation")
        elif accuracy_rate < 90:
            print("⚠️  WARNING: Accuracy below 90% - Improvements recommended")
            print("   - Fine-tune confidence thresholds")
            print("   - Add more validation rules")
        else:
            print("✅ EXCELLENT: High accuracy achieved!")
            print("   - Continue monitoring performance")
            print("   - Consider edge case testing")
        
        if avg_processing_time > 10:
            print("⚠️  PERFORMANCE: Processing time is high")
            print("   - Optimize AI provider settings")
            print("   - Implement caching improvements")
        
        print(f"\n🎊 TEST COMPLETED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    def export_results(self, filename: Optional[str] = None):
        """Export test results to JSON file"""
        import json
        
        if filename is None:
            filename = f"accuracy_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(self.test_results),
                "accuracy_rate": (sum(1 for r in self.test_results if r["is_correct"]) / len(self.test_results)) * 100,
                "avg_processing_time": sum(r["processing_time"] for r in self.test_results) / len(self.test_results)
            },
            "test_results": self.test_results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"📄 Results exported to: {filename}")

def main():
    """Main test runner"""
    try:
        tester = AccuracyTester()
        tester.run_comprehensive_tests()
        
        # Export results
        response = input("\n💾 Export results to file? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            tester.export_results()
        
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
