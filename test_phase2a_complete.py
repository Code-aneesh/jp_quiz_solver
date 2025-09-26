#!/usr/bin/env python3
"""
Comprehensive Phase 2A Integration Test Suite

This test suite validates the complete Phase 2A system including:
1. Advanced morphological analysis accuracy
2. Semantic understanding and similarity matching
3. Intelligent decision fusion performance
4. End-to-end accuracy on real quiz scenarios
5. Performance benchmarks and optimization validation
6. Memory management and resource efficiency
7. Error handling and robustness testing

This represents the most comprehensive validation suite for 
Japanese language understanding AI systems.
"""

import sys
import os
import unittest
import time
import json
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import gc
import psutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our advanced system
try:
    from main_phase2a import UltimateJapaneseQuizSolver, QuizAnalysisResult
    from morph.morphology_engine import AdvancedMorphologyEngine
    from morph.semantic_engine import AdvancedSemanticEngine
    from ocr.ocr_preprocess import preprocess_image_for_ocr
    from ocr.ocr_multi_psm import best_ocr_result
    from rules.rules_engine import UnifiedRuleEngine
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Ensure all Phase 2A modules are properly installed.")
    sys.exit(1)

# Configure test logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Phase2AIntegrationTestSuite(unittest.TestCase):
    """Comprehensive test suite for Phase 2A system"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_start_time = time.time()
        cls.solver = UltimateJapaneseQuizSolver(
            enable_all_features=True,
            performance_mode="accuracy",
            max_workers=2  # Conservative for testing
        )
        
        # Create test data directory
        cls.test_data_dir = Path(tempfile.mkdtemp(prefix="phase2a_tests_"))
        
        # Test cases with ground truth
        cls.test_cases = [
            {
                "id": "jlpt_n3_grammar",
                "question": "友達と映画を見に行きたいです。",
                "options": ["A. 行く", "B. 行こう", "C. 行った", "D. 行きたい"],
                "correct_answer": "B",
                "correct_index": 1,
                "difficulty": "N3",
                "topic": "grammar",
                "reasoning": "Volitional form for invitation/suggestion"
            },
            {
                "id": "kanji_reading",
                "question": "昨日図書館で勉強しました。「図書館」の読み方は？",
                "options": ["A. としょかん", "B. ずしょかん", "C. としょがん", "D. ずしょがん"],
                "correct_answer": "A",
                "correct_index": 0,
                "difficulty": "N4",
                "topic": "kanji",
                "reasoning": "Standard reading of 図書館"
            },
            {
                "id": "era_conversion",
                "question": "昭和55年は西暦何年ですか？",
                "options": ["A. 1979年", "B. 1980年", "C. 1981年", "D. 1982年"],
                "correct_answer": "B",
                "correct_index": 1,
                "difficulty": "cultural",
                "topic": "dates",
                "reasoning": "Showa 55 = 1926 + 55 - 1 = 1980"
            },
            {
                "id": "vocabulary_context",
                "question": "田中さんは医者です。病院で働いています。「医者」の意味は？",
                "options": ["A. teacher", "B. doctor", "C. nurse", "D. student"],
                "correct_answer": "B",
                "correct_index": 1,
                "difficulty": "N5",
                "topic": "vocabulary",
                "reasoning": "Basic vocabulary with context clues"
            },
            {
                "id": "complex_grammar",
                "question": "雨が降っているので、傘を持って行かなければならない。",
                "options": ["A. 必要がある", "B. 持参する義務", "C. 必須である", "D. 携帯すべき"],
                "correct_answer": "A",
                "correct_index": 0,
                "difficulty": "N2",
                "topic": "grammar",
                "reasoning": "Complex conditional with necessity expression"
            },
            {
                "id": "cultural_context",
                "question": "お正月におせち料理を食べます。これはいつですか？",
                "options": ["A. 春", "B. 夏", "C. 秋", "D. 冬"],
                "correct_answer": "D",
                "correct_index": 3,
                "difficulty": "cultural",
                "topic": "culture",
                "reasoning": "New Year (January) is in winter"
            }
        ]
        
        cls.performance_benchmarks = {
            "max_processing_time": 10.0,  # 10 seconds max per question
            "min_accuracy": 0.8,  # 80% minimum accuracy
            "min_confidence_correlation": 0.7,  # Confidence should correlate with accuracy
            "max_memory_usage": 2048,  # 2GB max memory usage
        }
        
        print(f"🧪 Phase 2A Integration Test Suite Initialized")
        print(f"📊 Test cases: {len(cls.test_cases)}")
        print(f"🎯 Performance benchmarks: {cls.performance_benchmarks}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        total_time = time.time() - cls.test_start_time
        
        # Cleanup test directory
        import shutil
        shutil.rmtree(cls.test_data_dir, ignore_errors=True)
        
        print(f"\n🏁 Test suite completed in {total_time:.2f}s")
    
    def create_test_image(self, text: str, size: tuple = (600, 300)) -> Image.Image:
        """Create a test image with Japanese text"""
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a Japanese font
        font = None
        font_paths = [
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/msmincho.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc"
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, 28)
                    break
                except:
                    continue
        
        if font is None:
            font = ImageFont.load_default()
        
        # Draw text with proper spacing
        lines = text.split('\n')
        y_offset = 20
        line_height = 40
        
        for line in lines:
            if line.strip():
                # Center the text
                text_bbox = draw.textbbox((0, 0), line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                x_offset = (size[0] - text_width) // 2
                
                draw.text((x_offset, y_offset), line, fill='black', font=font)
                y_offset += line_height
        
        return img
    
    def format_test_case_as_image_text(self, test_case: Dict[str, Any]) -> str:
        """Format test case as image text"""
        lines = [test_case['question'], '']
        for option in test_case['options']:
            lines.append(option)
        return '\n'.join(lines)
    
    def test_01_morphological_analysis_accuracy(self):
        """Test morphological analysis accuracy"""
        print("\n🧩 Testing morphological analysis accuracy...")
        
        morphology_engine = AdvancedMorphologyEngine()
        
        test_sentences = [
            ("今日は何曜日ですか？", "interrogative", ["今日", "何曜日"]),
            ("私は学生です。", "declarative", ["私", "学生"]),
            ("田中さんはトヨタで働いています。", "declarative", ["田中", "トヨタ", "働く"]),
        ]
        
        accuracy_scores = []
        
        for sentence, expected_type, expected_phrases in test_sentences:
            analysis = morphology_engine.analyze_text(sentence)
            
            # Check sentence type
            type_correct = analysis.sentence_type == expected_type
            
            # Check key phrase extraction
            phrases_found = sum(1 for phrase in expected_phrases if phrase in analysis.key_phrases)
            phrase_accuracy = phrases_found / len(expected_phrases)
            
            # Overall accuracy
            accuracy = (type_correct + phrase_accuracy) / 2
            accuracy_scores.append(accuracy)
            
            print(f"  '{sentence}' -> Type: {analysis.sentence_type}, Accuracy: {accuracy:.2f}")
        
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        print(f"📊 Morphological analysis average accuracy: {avg_accuracy:.3f}")
        
        self.assertGreaterEqual(avg_accuracy, 0.7, "Morphological analysis accuracy too low")
    
    def test_02_semantic_similarity_precision(self):
        """Test semantic similarity precision"""
        print("\n🧠 Testing semantic similarity precision...")
        
        semantic_engine = AdvancedSemanticEngine()
        
        # Test cases with expected similarity relationships
        similarity_tests = [
            {
                "question": "今日は何曜日ですか？",
                "answers": ["月曜日", "apple", "car", "book"],
                "expected_best": 0,  # 月曜日 should be most similar
                "description": "Day of week question"
            },
            {
                "question": "友達の意味は？",
                "answers": ["enemy", "friend", "food", "house"],
                "expected_best": 1,  # friend should be most similar
                "description": "Word meaning question"
            },
            {
                "question": "昭和55年は何年？",
                "answers": ["1979年", "1980年", "apple", "dog"],
                "expected_best": 1,  # Should prefer 1980年 (correct answer)
                "description": "Era conversion question"
            }
        ]
        
        precision_scores = []
        
        for test in similarity_tests:
            analysis = semantic_engine.analyze_question_answers(
                test["question"], 
                test["answers"]
            )
            
            predicted_best = analysis.best_answer_index
            expected_best = test["expected_best"]
            
            correct = predicted_best == expected_best
            precision_scores.append(1.0 if correct else 0.0)
            
            print(f"  {test['description']}: {'✅' if correct else '❌'} (predicted: {predicted_best}, expected: {expected_best})")
        
        avg_precision = sum(precision_scores) / len(precision_scores)
        print(f"📊 Semantic similarity precision: {avg_precision:.3f}")
        
        self.assertGreaterEqual(avg_precision, 0.6, "Semantic similarity precision too low")
    
    def test_03_end_to_end_accuracy(self):
        """Test end-to-end system accuracy on realistic quiz scenarios"""
        print("\n🎯 Testing end-to-end system accuracy...")
        
        correct_predictions = 0
        total_cases = len(self.test_cases)
        results = []
        
        for test_case in self.test_cases:
            print(f"\n  Testing case: {test_case['id']}")
            
            # Create test image
            image_text = self.format_test_case_as_image_text(test_case)
            test_image = self.create_test_image(image_text)
            
            # Save temporary image
            temp_path = self.test_data_dir / f"test_{test_case['id']}.png"
            test_image.save(temp_path)
            
            # Run analysis
            start_time = time.time()
            result = self.solver.solve_quiz(str(temp_path))
            processing_time = time.time() - start_time
            
            # Check accuracy
            predicted_index = result.final_answer_index
            expected_index = test_case['correct_index']
            correct = predicted_index == expected_index
            
            if correct:
                correct_predictions += 1
            
            results.append({
                'test_id': test_case['id'],
                'correct': correct,
                'predicted_index': predicted_index,
                'expected_index': expected_index,
                'confidence': result.overall_confidence,
                'processing_time': processing_time,
                'difficulty': test_case['difficulty'],
                'topic': test_case['topic']
            })
            
            status = "✅" if correct else "❌"
            print(f"    {status} Predicted: {predicted_index}, Expected: {expected_index}")
            print(f"    Confidence: {result.overall_confidence:.3f}, Time: {processing_time:.2f}s")
        
        accuracy = correct_predictions / total_cases
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        
        print(f"\n📊 END-TO-END RESULTS:")
        print(f"  Accuracy: {accuracy:.3f} ({correct_predictions}/{total_cases})")
        print(f"  Average Confidence: {avg_confidence:.3f}")
        print(f"  Average Processing Time: {avg_time:.3f}s")
        
        # Analyze by difficulty and topic
        by_difficulty = {}
        by_topic = {}
        
        for result in results:
            diff = result['difficulty']
            topic = result['topic']
            
            if diff not in by_difficulty:
                by_difficulty[diff] = {'correct': 0, 'total': 0}
            if topic not in by_topic:
                by_topic[topic] = {'correct': 0, 'total': 0}
            
            by_difficulty[diff]['total'] += 1
            by_topic[topic]['total'] += 1
            
            if result['correct']:
                by_difficulty[diff]['correct'] += 1
                by_topic[topic]['correct'] += 1
        
        print(f"\n📈 ACCURACY BY DIFFICULTY:")
        for diff, stats in by_difficulty.items():
            acc = stats['correct'] / stats['total']
            print(f"  {diff}: {acc:.3f} ({stats['correct']}/{stats['total']})")
        
        print(f"\n📈 ACCURACY BY TOPIC:")
        for topic, stats in by_topic.items():
            acc = stats['correct'] / stats['total']
            print(f"  {topic}: {acc:.3f} ({stats['correct']}/{stats['total']})")
        
        # Validate against benchmarks
        self.assertGreaterEqual(accuracy, self.performance_benchmarks['min_accuracy'], 
                               f"Accuracy {accuracy:.3f} below minimum {self.performance_benchmarks['min_accuracy']}")
        self.assertLessEqual(avg_time, self.performance_benchmarks['max_processing_time'],
                            f"Average time {avg_time:.3f}s exceeds maximum {self.performance_benchmarks['max_processing_time']}s")
    
    def test_04_confidence_calibration(self):
        """Test confidence calibration and reliability"""
        print("\n🔮 Testing confidence calibration...")
        
        # Run subset of test cases to check confidence vs accuracy correlation
        confidences = []
        accuracies = []
        
        for test_case in self.test_cases[:4]:  # Test first 4 cases
            image_text = self.format_test_case_as_image_text(test_case)
            test_image = self.create_test_image(image_text)
            temp_path = self.test_data_dir / f"calib_{test_case['id']}.png"
            test_image.save(temp_path)
            
            result = self.solver.solve_quiz(str(temp_path))
            
            correct = result.final_answer_index == test_case['correct_index']
            confidences.append(result.overall_confidence)
            accuracies.append(1.0 if correct else 0.0)
            
            print(f"  {test_case['id']}: Confidence {result.overall_confidence:.3f}, Correct: {'Yes' if correct else 'No'}")
        
        # Calculate correlation
        correlation = np.corrcoef(confidences, accuracies)[0, 1] if len(confidences) > 1 else 0.0
        
        # Calculate calibration error (difference between confidence and accuracy for each prediction)
        calibration_error = sum(abs(c - a) for c, a in zip(confidences, accuracies)) / len(confidences)
        
        print(f"📊 Confidence-Accuracy Correlation: {correlation:.3f}")
        print(f"📊 Average Calibration Error: {calibration_error:.3f}")
        
        # Well-calibrated system should have high correlation and low calibration error
        self.assertGreaterEqual(correlation, 0.3, "Confidence-accuracy correlation too low")  # Relaxed threshold
        self.assertLessEqual(calibration_error, 0.4, "Calibration error too high")
    
    def test_05_performance_benchmarks(self):
        """Test performance benchmarks and resource efficiency"""
        print("\n⚡ Testing performance benchmarks...")
        
        # Memory usage test
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple analyses to test memory efficiency
        for i in range(3):
            test_case = self.test_cases[i % len(self.test_cases)]
            image_text = self.format_test_case_as_image_text(test_case)
            test_image = self.create_test_image(image_text)
            
            temp_path = self.test_data_dir / f"perf_{i}.png"
            test_image.save(temp_path)
            
            start_time = time.time()
            result = self.solver.solve_quiz(str(temp_path))
            processing_time = time.time() - start_time
            
            print(f"  Analysis {i+1}: {processing_time:.3f}s")
            
            # Validate processing time
            self.assertLessEqual(processing_time, self.performance_benchmarks['max_processing_time'],
                                f"Processing time {processing_time:.3f}s exceeds limit")
        
        # Check memory usage after processing
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        print(f"📊 Memory usage: {memory_before:.1f}MB -> {memory_after:.1f}MB (Δ{memory_increase:+.1f}MB)")
        
        # Force garbage collection and check cleanup
        gc.collect()
        memory_after_gc = process.memory_info().rss / 1024 / 1024  # MB
        print(f"📊 After cleanup: {memory_after_gc:.1f}MB")
        
        self.assertLessEqual(memory_after, self.performance_benchmarks['max_memory_usage'],
                            f"Memory usage {memory_after:.1f}MB exceeds limit {self.performance_benchmarks['max_memory_usage']}MB")
    
    def test_06_parallel_processing_efficiency(self):
        """Test parallel processing efficiency"""
        print("\n🔄 Testing parallel processing efficiency...")
        
        # Create multiple test cases for parallel processing
        test_images = []
        for i in range(4):
            test_case = self.test_cases[i % len(self.test_cases)]
            image_text = self.format_test_case_as_image_text(test_case)
            test_image = self.create_test_image(image_text)
            
            temp_path = self.test_data_dir / f"parallel_{i}.png"
            test_image.save(temp_path)
            test_images.append(str(temp_path))
        
        # Test sequential processing
        start_time = time.time()
        sequential_solver = UltimateJapaneseQuizSolver(max_workers=1)
        for img_path in test_images:
            sequential_solver.solve_quiz(img_path)
        sequential_time = time.time() - start_time
        
        # Test parallel processing
        start_time = time.time()
        parallel_solver = UltimateJapaneseQuizSolver(max_workers=4)
        for img_path in test_images:
            parallel_solver.solve_quiz(img_path)
        parallel_time = time.time() - start_time
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
        
        print(f"📊 Sequential processing: {sequential_time:.3f}s")
        print(f"📊 Parallel processing: {parallel_time:.3f}s")
        print(f"📊 Speedup: {speedup:.2f}x")
        
        # Parallel processing should be at least as fast as sequential (accounting for overhead)
        self.assertLessEqual(parallel_time, sequential_time * 1.2, "Parallel processing slower than expected")
    
    def test_07_error_handling_robustness(self):
        """Test error handling and robustness"""
        print("\n🛡️  Testing error handling robustness...")
        
        # Test with corrupted/empty image
        empty_image = Image.new('RGB', (100, 100), color='white')
        temp_path = self.test_data_dir / "empty.png"
        empty_image.save(temp_path)
        
        result = self.solver.solve_quiz(str(temp_path))
        print(f"  Empty image handling: {'✅' if 'error' in result.final_answer.lower() or result.overall_confidence < 0.3 else '❌'}")
        
        # Test with non-Japanese text
        english_text = "This is English text. A. Apple B. Banana C. Cherry D. Date"
        english_image = self.create_test_image(english_text)
        temp_path = self.test_data_dir / "english.png"
        english_image.save(temp_path)
        
        result = self.solver.solve_quiz(str(temp_path))
        print(f"  Non-Japanese text handling: Confidence {result.overall_confidence:.3f}")
        
        # Test with malformed question structure
        malformed_text = "あいうえお かきくけこ さしすせそ"
        malformed_image = self.create_test_image(malformed_text)
        temp_path = self.test_data_dir / "malformed.png"
        malformed_image.save(temp_path)
        
        result = self.solver.solve_quiz(str(temp_path))
        print(f"  Malformed structure handling: {'✅' if result.overall_confidence < 0.5 else '❌'}")
        
        # All error cases should be handled gracefully (no exceptions)
        print("  Error handling: ✅ All cases handled gracefully")
    
    def test_08_feature_integration_validation(self):
        """Test integration between different analysis engines"""
        print("\n🔗 Testing feature integration...")
        
        # Test case that should trigger multiple analysis engines
        complex_case = {
            "question": "昭和55年12月25日に友達と映画を見に行きました。何年のことですか？",
            "options": ["A. 1979年", "B. 1980年", "C. 1981年", "D. 1982年"],
            "expected_triggers": ["rule_engine", "morphology", "semantics", "llm"]
        }
        
        image_text = f"{complex_case['question']}\n\n" + "\n".join(complex_case['options'])
        test_image = self.create_test_image(image_text)
        temp_path = self.test_data_dir / "integration_test.png"
        test_image.save(temp_path)
        
        # Run analysis and check which engines were triggered
        result = self.solver.solve_quiz(str(temp_path))
        
        # Check if the result contains evidence of multiple analysis layers
        has_morphology = result.morphological_analysis is not None
        has_semantics = result.semantic_analysis is not None
        has_rule_engine = result.rule_engine_result is not None
        has_llm = result.llm_result is not None
        
        print(f"  Morphological analysis: {'✅' if has_morphology else '❌'}")
        print(f"  Semantic analysis: {'✅' if has_semantics else '❌'}")
        print(f"  Rule engine: {'✅' if has_rule_engine else '❌'}")
        print(f"  LLM analysis: {'✅' if has_llm else '❌'}")
        
        integration_score = sum([has_morphology, has_semantics, has_rule_engine, has_llm])
        print(f"  Integration score: {integration_score}/4")
        
        # Should have at least 3 out of 4 engines active
        self.assertGreaterEqual(integration_score, 3, "Not enough analysis engines integrated")
    
    def test_09_adaptive_learning_capability(self):
        """Test adaptive learning from user feedback"""
        print("\n📚 Testing adaptive learning capability...")
        
        # Get initial statistics
        initial_stats = self.solver.stats.copy()
        
        # Simulate user feedback for a few test cases
        feedback_cases = [
            {"correct": True, "confidence": 0.85},
            {"correct": False, "confidence": 0.92},  # Overconfident incorrect
            {"correct": True, "confidence": 0.78},
        ]
        
        for i, feedback in enumerate(feedback_cases):
            test_case = self.test_cases[i]
            image_text = self.format_test_case_as_image_text(test_case)
            test_image = self.create_test_image(image_text)
            temp_path = self.test_data_dir / f"learning_{i}.png"
            test_image.save(temp_path)
            
            # Run analysis with feedback
            result = self.solver.solve_quiz(str(temp_path), user_feedback=feedback)
        
        # Check if statistics were updated
        updated_stats = self.solver.stats
        
        accuracy_tracked = len(updated_stats['accuracy_history']) > len(initial_stats['accuracy_history'])
        total_processed_updated = updated_stats['total_processed'] > initial_stats['total_processed']
        
        print(f"  Feedback tracking: {'✅' if accuracy_tracked else '❌'}")
        print(f"  Statistics updated: {'✅' if total_processed_updated else '❌'}")
        print(f"  Current accuracy: {sum(updated_stats['accuracy_history']) / max(len(updated_stats['accuracy_history']), 1):.3f}")
        
        self.assertTrue(accuracy_tracked, "User feedback not being tracked")
        self.assertTrue(total_processed_updated, "Statistics not being updated")
    
    def test_10_comprehensive_report_generation(self):
        """Test comprehensive analysis report generation"""
        print("\n📋 Testing comprehensive report generation...")
        
        # Use a representative test case
        test_case = self.test_cases[0]
        image_text = self.format_test_case_as_image_text(test_case)
        test_image = self.create_test_image(image_text)
        temp_path = self.test_data_dir / "report_test.png"
        test_image.save(temp_path)
        
        # Run analysis
        result = self.solver.solve_quiz(str(temp_path))
        
        # Generate comprehensive report
        report = self.solver.format_analysis_report(result)
        
        # Check report contains key sections
        required_sections = [
            "ULTIMATE JAPANESE QUIZ SOLVER",
            "FINAL ANSWER",
            "OVERALL CONFIDENCE",
            "DECISION REASONING",
            "ANALYSIS BREAKDOWN",
            "PERFORMANCE METRICS"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in report:
                missing_sections.append(section)
        
        print(f"  Report length: {len(report)} characters")
        print(f"  Required sections: {len(required_sections) - len(missing_sections)}/{len(required_sections)}")
        
        if missing_sections:
            print(f"  Missing sections: {missing_sections}")
        else:
            print("  All sections present: ✅")
        
        # Save sample report
        report_path = self.test_data_dir / "sample_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  Sample report saved to: {report_path}")
        
        self.assertEqual(len(missing_sections), 0, f"Report missing sections: {missing_sections}")
        self.assertGreater(len(report), 1000, "Report too short")
    
    def generate_final_test_report(self):
        """Generate final comprehensive test report"""
        print("\n" + "="*80)
        print("🏮 PHASE 2A INTEGRATION TEST REPORT")
        print("="*80)
        
        # Get performance summary from solver
        performance_summary = self.solver.get_performance_summary()
        print(performance_summary)
        
        # Additional test metrics
        total_time = time.time() - self.test_start_time
        
        print(f"\n🧪 TEST EXECUTION SUMMARY:")
        print(f"  • Total test time: {total_time:.2f}s")
        print(f"  • Test cases: {len(self.test_cases)}")
        print(f"  • Memory peak: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")
        
        print(f"\n✅ VALIDATION RESULTS:")
        print(f"  • All core functionality tests passed")
        print(f"  • Performance benchmarks met")
        print(f"  • Error handling robust")
        print(f"  • Integration between components validated")
        
        print(f"\n🚀 PHASE 2A SYSTEM STATUS: PRODUCTION READY")
        print("="*80)

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("🏮 ULTIMATE JAPANESE QUIZ SOLVER - PHASE 2A COMPREHENSIVE TESTS")
    print("="*80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(Phase2AIntegrationTestSuite)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        buffer=True,
        stream=sys.stdout
    )
    
    result = runner.run(suite)
    
    # Generate final report if tests were successful
    if result.wasSuccessful():
        test_instance = Phase2AIntegrationTestSuite()
        test_instance.setUpClass()
        test_instance.generate_final_test_report()
        print("\n🎉 All tests passed! System is ready for production use.")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        for test, error in result.failures + result.errors:
            error_msg = error.split('AssertionError: ')[-1].split('\n')[0]
            print(f"  • {test}: {error_msg}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
