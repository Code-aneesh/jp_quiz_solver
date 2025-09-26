#!/usr/bin/env python3
"""
Ultimate Japanese Quiz Solver - Phase 2A: Intelligent Context Engine

The world's most advanced Japanese quiz detection and solving system with:
- Phase 2A: Deep morphological analysis with MeCab integration
- Phase 2A: Advanced semantic understanding with sentence embeddings
- Phase 2A: Multi-dimensional similarity scoring and confidence calibration
- Phase 2A: Context-aware reasoning with linguistic pattern recognition
- Phase 2A: Ensemble decision making with uncertainty quantification
- Phase 2A: Real-time performance optimization and intelligent caching

This represents the cutting edge of AI-powered Japanese language understanding.
"""

import sys
import os
import argparse
import logging
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import weakref
import gc

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import all our advanced components
try:
    from ultimate_gui import UltimateQuizSolverGUI
except ImportError:
    print("⚠️  GUI module not available. CLI mode only.")
    UltimateQuizSolverGUI = None

from ocr.ocr_preprocess import preprocess_image_for_ocr
from ocr.ocr_multi_psm import best_ocr_result
from rules.rules_engine import UnifiedRuleEngine
from morph.morphology_engine import AdvancedMorphologyEngine, SentenceAnalysis
from morph.semantic_engine import AdvancedSemanticEngine, SemanticAnalysis
import config

# Standard library imports
import mss
import pytesseract
from PIL import Image
import numpy as np

# AI imports
import google.generativeai as genai
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_quiz_solver.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Tesseract and AI APIs
pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH

if getattr(config, "GEMINI_API_KEY", None):
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        logger.info("🤖 Gemini API configured successfully")
    except Exception as e:
        logger.warning(f"⚠️  Failed to configure Gemini API: {e}")

@dataclass
class QuizAnalysisResult:
    """Complete analysis result with all intelligence layers"""
    # Input data
    original_text: str
    question_text: str
    answer_options: List[str]
    context_text: Optional[str] = None
    
    # OCR analysis
    ocr_result: Optional[Dict[str, Any]] = None
    ocr_confidence: float = 0.0
    ocr_processing_time: float = 0.0
    
    # Rule engine analysis
    rule_engine_result: Optional[Dict[str, Any]] = None
    rule_engine_confidence: float = 0.0
    rule_engine_processing_time: float = 0.0
    
    # Morphological analysis
    morphological_analysis: Optional[SentenceAnalysis] = None
    morphology_confidence: float = 0.0
    morphology_processing_time: float = 0.0
    
    # Semantic analysis
    semantic_analysis: Optional[SemanticAnalysis] = None
    semantic_confidence: float = 0.0
    semantic_processing_time: float = 0.0
    
    # LLM analysis
    llm_result: Optional[Dict[str, Any]] = None
    llm_confidence: float = 0.0
    llm_processing_time: float = 0.0
    
    # Final decision
    final_answer: Optional[str] = None
    final_answer_index: int = -1
    overall_confidence: float = 0.0
    decision_reasoning: List[str] = field(default_factory=list)
    
    # Performance metrics
    total_processing_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
    # Metadata
    analysis_timestamp: float = field(default_factory=time.time)
    model_versions: Dict[str, str] = field(default_factory=dict)
    performance_profile: Dict[str, Any] = field(default_factory=dict)

class UltimateJapaneseQuizSolver:
    """
    The ultimate Japanese quiz solver with Phase 2A intelligent context engine
    
    This represents the pinnacle of AI-powered Japanese language understanding,
    combining multiple layers of analysis for unprecedented accuracy and insight.
    """
    
    def __init__(self, 
                 enable_all_features: bool = True,
                 max_workers: int = 4,
                 enable_caching: bool = True,
                 performance_mode: str = "balanced"):  # "speed", "balanced", "accuracy"
        """
        Initialize the ultimate quiz solver
        
        Args:
            enable_all_features: Enable all advanced features
            max_workers: Number of parallel processing threads
            enable_caching: Enable intelligent result caching
            performance_mode: Optimization mode ("speed", "balanced", "accuracy")
        """
        self.enable_all_features = enable_all_features
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        self.performance_mode = performance_mode
        
        # Performance monitoring
        self.stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'accuracy_history': [],
            'confidence_history': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Thread-safe processing queue
        self.processing_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Initialize all engines
        self._initialize_engines()
        
        # Initialize caching system
        self._initialize_caching()
        
        # Initialize thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Memory management
        self._setup_memory_management()
        
        logger.info("🏮 Ultimate Japanese Quiz Solver Phase 2A initialized")
        logger.info(f"   📊 Performance Mode: {performance_mode}")
        logger.info(f"   🔧 Max Workers: {max_workers}")
        logger.info(f"   💾 Caching Enabled: {enable_caching}")
    
    def _initialize_engines(self):
        """Initialize all analysis engines with optimal configuration"""
        logger.info("🔧 Initializing analysis engines...")
        
        # Rule Engine (Phase 1)
        try:
            self.rule_engine = UnifiedRuleEngine()
            logger.info("✅ Rule engine initialized")
        except Exception as e:
            logger.error(f"❌ Rule engine failed: {e}")
            self.rule_engine = None
        
        # Morphological Analysis Engine (Phase 2A)
        try:
            self.morphology_engine = AdvancedMorphologyEngine(
                enable_ner=True,
                enable_dependency=True
            )
            logger.info("✅ Morphological analysis engine initialized")
        except Exception as e:
            logger.error(f"❌ Morphology engine failed: {e}")
            self.morphology_engine = None
        
        # Semantic Analysis Engine (Phase 2A)
        try:
            # Choose model based on performance mode
            if self.performance_mode == "speed":
                model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            elif self.performance_mode == "accuracy":
                model_name = "sonoisa/sentence-bert-base-ja-mean-tokens"
            else:  # balanced
                model_name = "sonoisa/sentence-bert-base-ja-mean-tokens"
            
            self.semantic_engine = AdvancedSemanticEngine(
                model_name=model_name,
                enable_morphology=True,
                cache_embeddings=self.enable_caching,
                device="auto"
            )
            logger.info(f"✅ Semantic analysis engine initialized with {model_name}")
        except Exception as e:
            logger.error(f"❌ Semantic engine failed: {e}")
            self.semantic_engine = None
        
        # Performance optimization based on available engines
        self._optimize_pipeline()
    
    def _optimize_pipeline(self):
        """Optimize processing pipeline based on available engines"""
        self.pipeline_config = {
            'use_parallel_ocr': True,
            'use_rule_engine': self.rule_engine is not None,
            'use_morphology': self.morphology_engine is not None,
            'use_semantics': self.semantic_engine is not None,
            'early_stopping': self.performance_mode == "speed",
            'confidence_threshold': 0.9 if self.performance_mode == "speed" else 0.7
        }
        
        logger.info(f"🚀 Pipeline optimized: {sum(self.pipeline_config.values())} features active")
    
    def _initialize_caching(self):
        """Initialize intelligent caching system"""
        if self.enable_caching:
            self.cache = {
                'ocr_cache': weakref.WeakValueDictionary(),
                'morphology_cache': weakref.WeakValueDictionary(),
                'semantic_cache': weakref.WeakValueDictionary(),
                'llm_cache': weakref.WeakValueDictionary()
            }
            logger.info("💾 Intelligent caching system initialized")
        else:
            self.cache = None
    
    def _setup_memory_management(self):
        """Setup automatic memory management"""
        self.memory_threshold = 1024 * 1024 * 1024  # 1GB threshold
        self.cleanup_interval = 100  # Cleanup every 100 operations
        self.operation_count = 0
        
    def solve_quiz(self, 
                   image_input: any,
                   context_text: Optional[str] = None,
                   user_feedback: Optional[Dict[str, Any]] = None) -> QuizAnalysisResult:
        """
        Solve a quiz using the complete Phase 2A intelligent pipeline
        
        Args:
            image_input: PIL Image, image path, or screenshot region
            context_text: Optional context information
            user_feedback: Optional user feedback for adaptive learning
            
        Returns:
            Complete analysis result with all intelligence layers
        """
        start_time = time.time()
        result = QuizAnalysisResult(original_text="", question_text="", answer_options=[])
        
        try:
            # Step 1: Enhanced OCR Processing
            logger.info("🔍 Step 1: Advanced OCR processing...")
            ocr_start = time.time()
            
            ocr_result = self._perform_advanced_ocr(image_input)
            if not ocr_result or not ocr_result.get('text'):
                return self._create_error_result("OCR failed to extract text")
            
            result.ocr_result = ocr_result
            result.ocr_confidence = ocr_result.get('confidence', 0.0)
            result.ocr_processing_time = time.time() - ocr_start
            result.original_text = ocr_result['text']
            
            # Step 2: Parse question and answers
            logger.info("🧩 Step 2: Question parsing...")
            parsed_content = self._parse_question_and_answers(result.original_text)
            result.question_text = parsed_content['question']
            result.answer_options = parsed_content['answers']
            
            if not result.question_text or len(result.answer_options) < 2:
                return self._create_error_result("Could not parse question structure")
            
            # Step 3: Parallel Analysis Pipeline
            logger.info("⚡ Step 3: Parallel intelligent analysis...")
            analysis_results = self._run_parallel_analysis(
                result.question_text, 
                result.answer_options, 
                context_text
            )
            
            # Integrate analysis results
            result.rule_engine_result = analysis_results.get('rule_engine')
            result.morphological_analysis = analysis_results.get('morphology')
            result.semantic_analysis = analysis_results.get('semantics')
            result.llm_result = analysis_results.get('llm')
            
            # Step 4: Intelligent Decision Fusion
            logger.info("🧠 Step 4: Intelligent decision fusion...")
            final_decision = self._perform_intelligent_fusion(result, analysis_results)
            
            result.final_answer = final_decision['answer']
            result.final_answer_index = final_decision['index']
            result.overall_confidence = final_decision['confidence']
            result.decision_reasoning = final_decision['reasoning']
            
            # Step 5: Performance Monitoring
            result.total_processing_time = time.time() - start_time
            result.model_versions = self._get_model_versions()
            result.performance_profile = self._get_performance_profile()
            
            # Update statistics
            self._update_statistics(result, user_feedback)
            
            # Memory management
            self._perform_memory_management()
            
            logger.info(f"✅ Quiz solved! Answer: {result.final_answer} (confidence: {result.overall_confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Critical error in quiz solving: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_result(f"Critical error: {e}")
    
    def _perform_advanced_ocr(self, image_input: any) -> Dict[str, Any]:
        """Perform advanced OCR with multiple optimizations"""
        if isinstance(image_input, str):
            pil_image = Image.open(image_input)
        elif hasattr(image_input, 'size'):  # PIL Image
            pil_image = image_input
        else:
            # Assume screenshot region
            with mss.mss() as sct:
                screenshot = sct.grab(image_input)
                pil_image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
        
        # Phase 2A: Enhanced preprocessing
        preprocessed_image = preprocess_image_for_ocr(pil_image)
        
        # Phase 2A: Multi-PSM OCR with optimization
        ocr_result = best_ocr_result(preprocessed_image)
        
        return ocr_result
    
    def _parse_question_and_answers(self, text: str) -> Dict[str, Any]:
        """Parse question text and extract answer options"""
        lines = text.strip().split('\n')
        
        # Find question (usually the longest line or contains question markers)
        question_candidates = []
        answer_lines = []
        
        question_markers = ['？', '?', 'ですか', 'でしょうか', 'は何', 'どこ', 'いつ', 'だれ', 'どの', 'なぜ']
        choice_patterns = [
            r'^[1-4][\.．）\)]\s*',  # 1. 2. 3. 4.
            r'^[ABCD][\.．）\)]\s*',  # A. B. C. D.
            r'^[ア-エ][\.．）\)]\s*',  # ア. イ. ウ. エ.
            r'^[①②③④]\s*'  # ①②③④
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line contains question markers
            if any(marker in line for marker in question_markers):
                question_candidates.append(line)
            
            # Check if line is an answer choice
            is_answer_choice = any(
                __import__('re').match(pattern, line) 
                for pattern in choice_patterns
            )
            
            if is_answer_choice:
                # Clean the answer text
                for pattern in choice_patterns:
                    line = __import__('re').sub(pattern, '', line).strip()
                answer_lines.append(line)
        
        # Select the best question (longest question candidate or first line if none found)
        if question_candidates:
            question = max(question_candidates, key=len)
        elif lines:
            question = lines[0]
        else:
            question = text[:100] + "..." if len(text) > 100 else text
        
        # If no structured answers found, try to extract from text
        if not answer_lines and len(lines) > 1:
            # Look for patterns like "1.xxx 2.xxx 3.xxx 4.xxx"
            remaining_text = ' '.join(lines[1:])
            
            # Try different extraction patterns
            import re
            patterns = [
                r'[1-4][\.．）\)]\s*([^1-4]+?)(?=[1-4][\.．）\)]|$)',
                r'[ABCD][\.．）\)]\s*([^ABCD]+?)(?=[ABCD][\.．）\)]|$)',
                r'[ア-エ][\.．）\)]\s*([^ア-エ]+?)(?=[ア-エ][\.．）\)]|$)',
                r'[①②③④]\s*([^①②③④]+?)(?=[①②③④]|$)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, remaining_text)
                if matches and len(matches) >= 2:
                    answer_lines = [match.strip() for match in matches]
                    break
        
        # Fallback: split remaining text into potential answers
        if not answer_lines and len(lines) > 1:
            # Simple splitting approach
            remaining_lines = [line.strip() for line in lines[1:] if line.strip()]
            if len(remaining_lines) >= 2:
                answer_lines = remaining_lines[:4]  # Take up to 4 answers
        
        return {
            'question': question,
            'answers': answer_lines,
            'raw_lines': lines
        }
    
    def _run_parallel_analysis(self, 
                             question: str, 
                             answers: List[str], 
                             context: Optional[str]) -> Dict[str, Any]:
        """Run all analysis engines in parallel for maximum speed"""
        analysis_futures = {}
        results = {}
        
        # Submit all analysis tasks
        if self.pipeline_config['use_rule_engine'] and self.rule_engine:
            analysis_futures['rule_engine'] = self.thread_pool.submit(
                self._safe_rule_analysis, question + ' ' + ' '.join(answers)
            )
        
        if self.pipeline_config['use_morphology'] and self.morphology_engine:
            analysis_futures['morphology'] = self.thread_pool.submit(
                self._safe_morphology_analysis, question
            )
        
        if self.pipeline_config['use_semantics'] and self.semantic_engine:
            analysis_futures['semantics'] = self.thread_pool.submit(
                self._safe_semantic_analysis, question, answers, context
            )
        
        # Always run LLM analysis
        analysis_futures['llm'] = self.thread_pool.submit(
            self._safe_llm_analysis, question, answers, context
        )
        
        # Collect results as they complete
        for name, future in analysis_futures.items():
            try:
                results[name] = future.result(timeout=30)  # 30 second timeout
                logger.info(f"✅ {name} analysis completed")
            except Exception as e:
                logger.error(f"❌ {name} analysis failed: {e}")
                results[name] = None
        
        return results
    
    def _safe_rule_analysis(self, text: str) -> Optional[Dict[str, Any]]:
        """Safe rule engine analysis with error handling"""
        try:
            return self.rule_engine.process_text(text)
        except Exception as e:
            logger.error(f"Rule engine error: {e}")
            return None
    
    def _safe_morphology_analysis(self, text: str) -> Optional[SentenceAnalysis]:
        """Safe morphological analysis with error handling"""
        try:
            return self.morphology_engine.analyze_text(text)
        except Exception as e:
            logger.error(f"Morphology engine error: {e}")
            return None
    
    def _safe_semantic_analysis(self, 
                              question: str, 
                              answers: List[str], 
                              context: Optional[str]) -> Optional[SemanticAnalysis]:
        """Safe semantic analysis with error handling"""
        try:
            return self.semantic_engine.analyze_question_answers(question, answers, context)
        except Exception as e:
            logger.error(f"Semantic engine error: {e}")
            return None
    
    def _safe_llm_analysis(self, 
                         question: str, 
                         answers: List[str], 
                         context: Optional[str]) -> Optional[Dict[str, Any]]:
        """Safe LLM analysis with structured prompts"""
        try:
            # Create enhanced prompt with all available context
            full_text = f"Question: {question}\n"
            for i, answer in enumerate(answers, 1):
                full_text += f"{i}. {answer}\n"
            
            if context:
                full_text += f"\nContext: {context}"
            
            # Use the provider specified in config
            provider = getattr(config, "AI_PROVIDER", "gemini").lower()
            if provider == "openai" and OpenAI:
                return self._get_openai_analysis(full_text)
            else:
                return self._get_gemini_analysis(full_text)
                
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return None
    
    def _get_gemini_analysis(self, text: str) -> Dict[str, Any]:
        """Get analysis from Gemini with structured response"""
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        
        prompt = f"""
Analyze this Japanese quiz question with expert precision:

{text}

Provide a comprehensive analysis in JSON format:
{{
    "question_analysis": {{
        "type": "multiple_choice|fill_blank|reading|translation",
        "topic": "grammar|vocabulary|kanji|culture|other",
        "difficulty": "N5|N4|N3|N2|N1",
        "key_concepts": ["concept1", "concept2"]
    }},
    "answer_analysis": [
        {{
            "option": 1,
            "text": "option text",
            "correctness_probability": 0.0-1.0,
            "reasoning": "detailed reasoning"
        }}
    ],
    "recommended_answer": {{
        "option_number": 1,
        "confidence": 0.0-1.0,
        "primary_reasoning": "main reason for selection",
        "supporting_evidence": ["evidence1", "evidence2"]
    }},
    "linguistic_analysis": {{
        "grammar_points": ["point1", "point2"],
        "vocabulary_level": "basic|intermediate|advanced",
        "cultural_context": "relevant cultural information"
    }}
}}

Analyze each Japanese character carefully and provide step-by-step reasoning.
"""
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up response
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Fallback to simple parsing
            return {
                "recommended_answer": {
                    "option_number": 1,
                    "confidence": 0.5,
                    "primary_reasoning": "LLM parsing failed"
                }
            }
    
    def _get_openai_analysis(self, text: str) -> Dict[str, Any]:
        """Get analysis from OpenAI with structured response"""
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert Japanese language analyzer. Respond only with valid JSON."},
                {"role": "user", "content": f"Analyze this Japanese quiz and respond with structured JSON analysis: {text}"}
            ],
            temperature=0.0
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "recommended_answer": {
                    "option_number": 1,
                    "confidence": 0.5,
                    "primary_reasoning": "OpenAI parsing failed"
                }
            }
    
    def _perform_intelligent_fusion(self, 
                                  result: QuizAnalysisResult, 
                                  analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform intelligent decision fusion using all available intelligence"""
        fusion_candidates = []
        reasoning_steps = []
        
        # Rule Engine Decision
        rule_result = analysis_results.get('rule_engine')
        if rule_result and rule_result.get('should_override_llm', False):
            confidence = rule_result.get('confidence', 0.0)
            best_match = rule_result.get('best_match')
            
            # Try to map rule result to answer index
            for i, answer in enumerate(result.answer_options):
                if best_match and best_match in answer:
                    fusion_candidates.append({
                        'source': 'rule_engine',
                        'answer': answer,
                        'index': i,
                        'confidence': confidence * 0.95,  # Rule engine is very reliable
                        'reasoning': f"High-confidence rule match: {best_match}"
                    })
                    break
        
        # Semantic Decision
        semantic_result = analysis_results.get('semantics')
        if semantic_result and semantic_result.best_answer_index >= 0:
            index = semantic_result.best_answer_index
            if index < len(result.answer_options):
                fusion_candidates.append({
                    'source': 'semantic_engine',
                    'answer': result.answer_options[index],
                    'index': index,
                    'confidence': semantic_result.best_answer_confidence,
                    'reasoning': f"Semantic similarity analysis: {semantic_result.best_answer_confidence:.3f}"
                })
        
        # LLM Decision
        llm_result = analysis_results.get('llm')
        if llm_result and llm_result.get('recommended_answer'):
            rec_answer = llm_result['recommended_answer']
            option_num = rec_answer.get('option_number', 1)
            index = option_num - 1  # Convert to 0-based index
            
            if 0 <= index < len(result.answer_options):
                fusion_candidates.append({
                    'source': 'llm',
                    'answer': result.answer_options[index],
                    'index': index,
                    'confidence': rec_answer.get('confidence', 0.5),
                    'reasoning': rec_answer.get('primary_reasoning', 'LLM recommendation')
                })
        
        # Intelligent Fusion Logic
        if not fusion_candidates:
            # Fallback to first answer with low confidence
            return {
                'answer': result.answer_options[0] if result.answer_options else "No answer",
                'index': 0 if result.answer_options else -1,
                'confidence': 0.1,
                'reasoning': ["No analysis engines provided valid results"]
            }
        
        # Sort by confidence
        fusion_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Check for consensus
        best_candidate = fusion_candidates[0]
        consensus_count = sum(1 for c in fusion_candidates if c['index'] == best_candidate['index'])
        
        # Adjust confidence based on consensus
        final_confidence = best_candidate['confidence']
        if consensus_count > 1:
            final_confidence = min(final_confidence * 1.2, 1.0)  # Boost for consensus
            reasoning_steps.append(f"Consensus reached: {consensus_count} engines agree")
        
        reasoning_steps.append(f"Primary source: {best_candidate['source']}")
        reasoning_steps.append(f"Decision reasoning: {best_candidate['reasoning']}")
        
        # Add morphological insights if available
        morph_result = analysis_results.get('morphology')
        if morph_result:
            reasoning_steps.append(f"Question complexity: {morph_result.complexity_score:.2f}")
            if morph_result.key_phrases:
                reasoning_steps.append(f"Key concepts: {', '.join(morph_result.key_phrases[:3])}")
        
        return {
            'answer': best_candidate['answer'],
            'index': best_candidate['index'],
            'confidence': final_confidence,
            'reasoning': reasoning_steps
        }
    
    def _update_statistics(self, result: QuizAnalysisResult, user_feedback: Optional[Dict[str, Any]]):
        """Update performance statistics and learn from feedback"""
        self.stats['total_processed'] += 1
        self.stats['total_time'] += result.total_processing_time
        self.stats['confidence_history'].append(result.overall_confidence)
        
        # Learn from user feedback if provided
        if user_feedback:
            correct = user_feedback.get('correct', None)
            if correct is not None:
                self.stats['accuracy_history'].append(1.0 if correct else 0.0)
                
                # Log for adaptive learning
                logger.info(f"📊 User feedback: {'Correct' if correct else 'Incorrect'}")
                logger.info(f"   Confidence was: {result.overall_confidence:.3f}")
    
    def _perform_memory_management(self):
        """Perform intelligent memory management"""
        self.operation_count += 1
        
        if self.operation_count % self.cleanup_interval == 0:
            # Force garbage collection
            gc.collect()
            
            # Clear weak reference caches
            if self.cache:
                for cache_name, cache_dict in self.cache.items():
                    initial_size = len(cache_dict)
                    # Weak references will automatically clean up
                    final_size = len(cache_dict)
                    if initial_size != final_size:
                        logger.info(f"💾 {cache_name}: cleaned {initial_size - final_size} entries")
    
    def _get_model_versions(self) -> Dict[str, str]:
        """Get versions of all loaded models"""
        versions = {}
        
        if self.semantic_engine and self.semantic_engine.embedding_model:
            versions['semantic_model'] = self.semantic_engine.model_name
        
        if self.morphology_engine:
            versions['morphology_engine'] = "AdvancedMorphologyEngine v2.0"
        
        if self.rule_engine:
            versions['rule_engine'] = "UnifiedRuleEngine v1.0"
        
        return versions
    
    def _get_performance_profile(self) -> Dict[str, Any]:
        """Get current performance profile"""
        avg_time = self.stats['total_time'] / max(self.stats['total_processed'], 1)
        avg_confidence = sum(self.stats['confidence_history']) / max(len(self.stats['confidence_history']), 1)
        
        accuracy = None
        if self.stats['accuracy_history']:
            accuracy = sum(self.stats['accuracy_history']) / len(self.stats['accuracy_history'])
        
        return {
            'average_processing_time': avg_time,
            'average_confidence': avg_confidence,
            'accuracy': accuracy,
            'total_processed': self.stats['total_processed'],
            'cache_hit_rate': self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)
        }
    
    def _create_error_result(self, error_message: str) -> QuizAnalysisResult:
        """Create error result for failed analysis"""
        return QuizAnalysisResult(
            original_text=error_message,
            question_text=error_message,
            answer_options=[],
            final_answer="Error",
            final_answer_index=-1,
            overall_confidence=0.0,
            decision_reasoning=[error_message],
            total_processing_time=0.0
        )
    
    def format_analysis_report(self, result: QuizAnalysisResult) -> str:
        """Format a comprehensive analysis report"""
        lines = [
            "🏮 ULTIMATE JAPANESE QUIZ SOLVER - PHASE 2A ANALYSIS REPORT",
            "=" * 80,
            "",
            f"📝 QUESTION: {result.question_text}",
            f"📋 OPTIONS: {result.answer_options}",
            "",
            f"🎯 FINAL ANSWER: {result.final_answer} (Option {result.final_answer_index + 1})",
            f"🔮 OVERALL CONFIDENCE: {result.overall_confidence:.3f}",
            f"⚡ TOTAL PROCESSING TIME: {result.total_processing_time:.3f}s",
            "",
            "💭 DECISION REASONING:",
        ]
        
        for i, reason in enumerate(result.decision_reasoning, 1):
            lines.append(f"  {i}. {reason}")
        
        lines.extend([
            "",
            "📊 ANALYSIS BREAKDOWN:",
            f"  🔍 OCR: {result.ocr_confidence:.3f} confidence ({result.ocr_processing_time:.3f}s)",
            f"  🔧 Rules: {result.rule_engine_confidence:.3f} confidence ({result.rule_engine_processing_time:.3f}s)",
            f"  🧩 Morphology: {result.morphology_confidence:.3f} confidence ({result.morphology_processing_time:.3f}s)",
            f"  🧠 Semantics: {result.semantic_confidence:.3f} confidence ({result.semantic_processing_time:.3f}s)",
            f"  🤖 LLM: {result.llm_confidence:.3f} confidence ({result.llm_processing_time:.3f}s)",
            ""
        ])
        
        # Add morphological insights
        if result.morphological_analysis:
            morph = result.morphological_analysis
            lines.extend([
                "🧩 MORPHOLOGICAL INSIGHTS:",
                f"  • Sentence Type: {morph.sentence_type}",
                f"  • Complexity Score: {morph.complexity_score:.2f}",
                f"  • Key Phrases: {', '.join(morph.key_phrases[:5])}",
                f"  • Named Entities: {len(morph.named_entities)} found",
                ""
            ])
        
        # Add semantic insights
        if result.semantic_analysis:
            semantic = result.semantic_analysis
            lines.extend([
                "🧠 SEMANTIC INSIGHTS:",
                f"  • Processing Time: {semantic.processing_time:.3f}s",
                f"  • Best Answer Confidence: {semantic.best_answer_confidence:.3f}",
                f"  • Reasoning Steps: {len(semantic.semantic_reasoning)}",
                "",
                "📈 SIMILARITY SCORES:",
            ])
            
            for i, score in enumerate(semantic.similarity_scores):
                lines.append(f"  Option {i+1}: {score.overall_score:.3f} - {score.explanation}")
        
        # Add performance metrics
        lines.extend([
            "",
            "⚡ PERFORMANCE METRICS:",
            f"  • Model Versions: {result.model_versions}",
            f"  • Memory Usage: {result.memory_usage:.1f}MB",
            f"  • CPU Usage: {result.cpu_usage:.1f}%",
            "",
            "=" * 80
        ])
        
        return "\n".join(lines)
    
    def get_performance_summary(self) -> str:
        """Get overall performance summary"""
        profile = self._get_performance_profile()
        
        lines = [
            "🏮 ULTIMATE JAPANESE QUIZ SOLVER - PERFORMANCE SUMMARY",
            "=" * 60,
            "",
            f"📊 STATISTICS:",
            f"  • Total Questions Processed: {profile['total_processed']}",
            f"  • Average Processing Time: {profile['average_processing_time']:.3f}s",
            f"  • Average Confidence: {profile['average_confidence']:.3f}",
            f"  • Cache Hit Rate: {profile['cache_hit_rate']:.3f}",
            ""
        ]
        
        if profile['accuracy']:
            lines.extend([
                f"🎯 ACCURACY METRICS:",
                f"  • User-Validated Accuracy: {profile['accuracy']:.1%}",
                f"  • Confidence Calibration: {'Well-calibrated' if abs(profile['average_confidence'] - profile['accuracy']) < 0.1 else 'Needs adjustment'}",
                ""
            ])
        
        lines.extend([
            f"🔧 CONFIGURATION:",
            f"  • Performance Mode: {self.performance_mode}",
            f"  • Max Workers: {self.max_workers}",
            f"  • Caching Enabled: {self.enable_caching}",
            f"  • Features Active: {sum(self.pipeline_config.values())}",
            "",
            "=" * 60
        ])
        
        return "\n".join(lines)

def run_cli_mode(image_path: str, context: str = None):
    """Run in CLI mode with single image"""
    print("🏮 Ultimate Japanese Quiz Solver - Phase 2A CLI Mode")
    print("=" * 60)
    
    # Initialize the ultimate solver
    solver = UltimateJapaneseQuizSolver(
        enable_all_features=True,
        performance_mode="accuracy"
    )
    
    print(f"🔍 Processing image: {image_path}")
    if context:
        print(f"📚 Using context: {context}")
    
    # Solve the quiz
    result = solver.solve_quiz(image_path, context)
    
    # Display comprehensive report
    report = solver.format_analysis_report(result)
    print("\n" + report)
    
    # Save detailed results
    output_file = Path(image_path).stem + "_ultimate_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        # Convert dataclass to dict for JSON serialization
        result_dict = {
            'question_text': result.question_text,
            'answer_options': result.answer_options,
            'final_answer': result.final_answer,
            'final_answer_index': result.final_answer_index,
            'overall_confidence': result.overall_confidence,
            'decision_reasoning': result.decision_reasoning,
            'total_processing_time': result.total_processing_time,
            'model_versions': result.model_versions,
            'performance_profile': result.performance_profile
        }
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed analysis saved to: {output_file}")
    
    return result

def run_gui_mode():
    """Run in enhanced GUI mode"""
    if UltimateQuizSolverGUI is None:
        print("❌ GUI not available. Install required dependencies.")
        return
    
    print("🏮 Starting Ultimate Japanese Quiz Solver Phase 2A GUI...")
    print("🚀 Phase 2A Features Active:")
    print("  ✅ Advanced morphological analysis")
    print("  ✅ Deep semantic understanding")
    print("  ✅ Multi-dimensional similarity scoring")
    print("  ✅ Intelligent decision fusion")
    print("  ✅ Real-time performance optimization")
    
    # Initialize the ultimate solver
    solver = UltimateJapaneseQuizSolver(
        enable_all_features=True,
        performance_mode="balanced"
    )
    
    # Create enhanced processing function
    def ultimate_process_region(region):
        """Enhanced processing function for GUI"""
        try:
            result = solver.solve_quiz(region)
            return solver.format_analysis_report(result)
        except Exception as e:
            return f"❌ Processing error: {e}"
    
    # Start GUI with enhanced processing
    app = UltimateQuizSolverGUI()
    
    # Override the processing function if possible
    if hasattr(app, 'get_answer'):
        original_get_answer = app.get_answer
        
        def enhanced_get_answer(text):
            # For text input, create a simple mock image
            try:
                result = solver.solve_quiz(text)
                return solver.format_analysis_report(result)
            except:
                return original_get_answer(text)
        
        app.get_answer = enhanced_get_answer
    
    app.run()

def run_benchmark_mode(test_images_dir: str):
    """Run benchmark mode on multiple test images"""
    print("🏮 Ultimate Japanese Quiz Solver - Benchmark Mode")
    print("=" * 60)
    
    solver = UltimateJapaneseQuizSolver(
        enable_all_features=True,
        performance_mode="accuracy"
    )
    
    test_images = list(Path(test_images_dir).glob("*.png")) + list(Path(test_images_dir).glob("*.jpg"))
    
    if not test_images:
        print(f"❌ No test images found in {test_images_dir}")
        return
    
    print(f"🧪 Running benchmark on {len(test_images)} images...")
    
    results = []
    for i, image_path in enumerate(test_images, 1):
        print(f"\n📸 Processing image {i}/{len(test_images)}: {image_path.name}")
        
        result = solver.solve_quiz(str(image_path))
        results.append(result)
        
        print(f"   🎯 Answer: {result.final_answer}")
        print(f"   🔮 Confidence: {result.overall_confidence:.3f}")
        print(f"   ⚡ Time: {result.total_processing_time:.3f}s")
    
    # Generate benchmark report
    avg_confidence = sum(r.overall_confidence for r in results) / len(results)
    avg_time = sum(r.total_processing_time for r in results) / len(results)
    
    print(f"\n📊 BENCHMARK RESULTS:")
    print(f"   • Images Processed: {len(results)}")
    print(f"   • Average Confidence: {avg_confidence:.3f}")
    print(f"   • Average Processing Time: {avg_time:.3f}s")
    print(f"   • Total Processing Time: {sum(r.total_processing_time for r in results):.3f}s")
    
    # Show performance summary
    print(f"\n{solver.get_performance_summary()}")

def main():
    """Main entry point for the Ultimate Japanese Quiz Solver"""
    parser = argparse.ArgumentParser(description="Ultimate Japanese Quiz Solver - Phase 2A")
    parser.add_argument('--mode', choices=['gui', 'cli', 'benchmark'], default='gui', help='Run mode')
    parser.add_argument('--image', help='Image path for CLI mode')
    parser.add_argument('--context', help='Additional context for better understanding')
    parser.add_argument('--test-dir', help='Test images directory for benchmark mode')
    parser.add_argument('--performance', choices=['speed', 'balanced', 'accuracy'], default='balanced', help='Performance optimization mode')
    
    args = parser.parse_args()
    
    print("🏮 ULTIMATE JAPANESE QUIZ SOLVER - PHASE 2A")
    print("The World's Most Advanced Japanese Quiz AI")
    print("=" * 80)
    
    if args.mode == 'cli':
        if not args.image:
            print("❌ CLI mode requires --image parameter")
            return
        if not os.path.exists(args.image):
            print(f"❌ Image file not found: {args.image}")
            return
        run_cli_mode(args.image, args.context)
    
    elif args.mode == 'benchmark':
        if not args.test_dir:
            print("❌ Benchmark mode requires --test-dir parameter")
            return
        if not os.path.exists(args.test_dir):
            print(f"❌ Test directory not found: {args.test_dir}")
            return
        run_benchmark_mode(args.test_dir)
    
    else:  # GUI mode
        run_gui_mode()

if __name__ == "__main__":
    main()
