#!/usr/bin/env python3
"""
Phase 2A: Advanced Semantic Analysis Engine

This module provides sophisticated semantic understanding for Japanese text
using sentence embeddings, similarity matching, and context-aware analysis
for superior quiz answer selection and validation.

Key Features:
- Sentence-BERT embeddings for Japanese (sonoisa/sentence-bert-base-ja-mean-tokens)
- Semantic similarity scoring between questions and answer options
- Context-aware embedding generation with morphological enhancement
- Cross-reference with knowledge graphs and semantic databases
- Multi-dimensional similarity analysis (lexical, syntactic, semantic)
- Answer confidence calibration using embedding distances
"""

import sys
import os
import re
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

# Import embedding libraries with fallback handling
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. Install with: pip install torch")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available. Install with: pip install scikit-learn")

# Import morphological analysis components
try:
    from .morphology_engine import AdvancedMorphologyEngine, SentenceAnalysis, MorphToken
    MORPHOLOGY_AVAILABLE = True
except ImportError:
    MORPHOLOGY_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SemanticVector:
    """Rich semantic vector representation with metadata"""
    text: str
    embedding: np.ndarray
    morphological_features: Optional[SentenceAnalysis] = None
    semantic_type: str = "general"  # question, answer_option, context, etc.
    confidence: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimilarityScore:
    """Comprehensive similarity scoring with multiple dimensions"""
    overall_score: float
    semantic_similarity: float  # Embedding-based similarity
    lexical_similarity: float  # Surface form similarity
    syntactic_similarity: float  # Structural similarity
    morphological_similarity: float  # POS and morphology similarity
    confidence: float
    explanation: str
    evidence: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticAnalysis:
    """Complete semantic analysis result"""
    question_vector: SemanticVector
    answer_vectors: List[SemanticVector]
    similarity_scores: List[SimilarityScore]
    best_answer_index: int
    best_answer_confidence: float
    semantic_reasoning: List[str]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedSemanticEngine:
    """
    Advanced semantic analysis engine for Japanese text understanding
    
    Provides comprehensive semantic matching using multiple embedding models,
    morphological analysis integration, and multi-dimensional similarity scoring
    for superior quiz answer selection.
    """
    
    def __init__(self, 
                 model_name: str = "sonoisa/sentence-bert-base-ja-mean-tokens",
                 enable_morphology: bool = True,
                 cache_embeddings: bool = True,
                 device: str = "auto"):
        """
        Initialize the semantic analysis engine
        
        Args:
            model_name: Sentence transformer model name for Japanese
            enable_morphology: Enable morphological analysis integration
            cache_embeddings: Enable embedding caching for performance
            device: Device for model inference ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.enable_morphology = enable_morphology
        self.cache_embeddings = cache_embeddings
        self.device = self._determine_device(device)
        
        # Initialize core components
        self._initialize_embedding_model()
        self._initialize_morphology_engine()
        self._initialize_similarity_components()
        self._initialize_knowledge_base()
        
        # Initialize caching system
        self.embedding_cache = {} if cache_embeddings else None
        
        logger.info(f"Semantic engine initialized - Model: {model_name}, Device: {self.device}")
    
    def _determine_device(self, device: str) -> str:
        """Determine optimal device for model inference"""
        if device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _initialize_embedding_model(self):
        """Initialize sentence transformer model for Japanese"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("SentenceTransformers not available - semantic analysis disabled")
            self.embedding_model = None
            return
        
        try:
            # Try to load the specified model
            self.embedding_model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded embedding model: {self.model_name}")
            
            # Test the model with a simple sentence
            test_embedding = self.embedding_model.encode("テスト", show_progress_bar=False)
            logger.info(f"Model test successful - Embedding dimension: {len(test_embedding)}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            
            # Try fallback models
            fallback_models = [
                "cl-tohoku/sbert-base-ja-mean-tokens",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "sentence-transformers/distiluse-base-multilingual-cased"
            ]
            
            self.embedding_model = None
            for fallback_model in fallback_models:
                try:
                    self.embedding_model = SentenceTransformer(fallback_model, device=self.device)
                    logger.warning(f"Using fallback model: {fallback_model}")
                    break
                except Exception as fallback_e:
                    logger.debug(f"Fallback model {fallback_model} failed: {fallback_e}")
                    continue
            
            if self.embedding_model is None:
                logger.error("No embedding model could be loaded - semantic analysis disabled")
    
    def _initialize_morphology_engine(self):
        """Initialize morphological analysis engine"""
        if self.enable_morphology and MORPHOLOGY_AVAILABLE:
            try:
                self.morphology_engine = AdvancedMorphologyEngine(
                    enable_ner=True, 
                    enable_dependency=True
                )
                logger.info("Morphological analysis engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize morphology engine: {e}")
                self.morphology_engine = None
        else:
            self.morphology_engine = None
    
    def _initialize_similarity_components(self):
        """Initialize additional similarity analysis components"""
        # Initialize TF-IDF vectorizer for lexical similarity
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                analyzer='char',  # Character-level for Japanese
                ngram_range=(1, 3),
                max_features=10000
            )
            logger.info("TF-IDF vectorizer initialized for lexical similarity")
        else:
            self.tfidf_vectorizer = None
        
        # Initialize fuzzy matching for surface similarity
        try:
            from difflib import SequenceMatcher
            self.sequence_matcher = SequenceMatcher
            logger.info("Sequence matcher initialized for fuzzy similarity")
        except ImportError:
            self.sequence_matcher = None
    
    def _initialize_knowledge_base(self):
        """Initialize knowledge base for semantic enhancement"""
        # Load Japanese-specific semantic patterns and knowledge
        self.semantic_patterns = {
            'question_indicators': [
                '何', 'どこ', 'いつ', 'だれ', 'どの', 'どう', 'なぜ', 'いくつ', 'いくら',
                'ですか', 'でしょうか', 'か？', '？'
            ],
            'answer_patterns': {
                'choice_markers': ['A', 'B', 'C', 'D', '1', '2', '3', '4', 'ア', 'イ', 'ウ', 'エ'],
                'temporal_markers': ['とき', 'とき', '時', '時間', '日', '月', '年'],
                'location_markers': ['で', 'に', 'から', 'まで', 'へ'],
                'quantity_markers': ['つ', '個', '本', '枚', '匹', '人', '台']
            },
            'semantic_relations': {
                'cause_effect': ['ので', 'から', 'ため', 'によって', 'により'],
                'contrast': ['しかし', 'でも', 'が', 'けれど', 'ところが'],
                'similarity': ['のように', 'みたい', '同じ', '似ている'],
                'sequence': ['そして', 'それから', 'つぎに', '最初', '最後']
            }
        }
        
        logger.info("Semantic knowledge base initialized")
    
    def analyze_question_answers(self, 
                                question_text: str, 
                                answer_options: List[str],
                                context: Optional[str] = None) -> SemanticAnalysis:
        """
        Perform comprehensive semantic analysis on question and answer options
        
        Args:
            question_text: The question to analyze
            answer_options: List of possible answer options
            context: Optional context text for enhanced understanding
            
        Returns:
            SemanticAnalysis object with complete semantic reasoning
        """
        start_time = time.time()
        
        if not self.embedding_model:
            return self._create_fallback_analysis(question_text, answer_options)
        
        try:
            # Step 1: Generate semantic vectors for all texts
            question_vector = self._generate_semantic_vector(question_text, "question")
            answer_vectors = [
                self._generate_semantic_vector(option, "answer_option", index=i) 
                for i, option in enumerate(answer_options)
            ]
            
            # Add context vector if provided
            context_vector = None
            if context:
                context_vector = self._generate_semantic_vector(context, "context")
            
            # Step 2: Calculate multi-dimensional similarities
            similarity_scores = []
            for i, answer_vector in enumerate(answer_vectors):
                similarity_score = self._calculate_comprehensive_similarity(
                    question_vector, answer_vector, context_vector
                )
                similarity_scores.append(similarity_score)
            
            # Step 3: Determine best answer based on semantic analysis
            best_answer_index, best_confidence = self._select_best_answer(
                question_vector, answer_vectors, similarity_scores
            )
            
            # Step 4: Generate semantic reasoning
            semantic_reasoning = self._generate_semantic_reasoning(
                question_vector, answer_vectors, similarity_scores, best_answer_index
            )
            
            processing_time = time.time() - start_time
            
            return SemanticAnalysis(
                question_vector=question_vector,
                answer_vectors=answer_vectors,
                similarity_scores=similarity_scores,
                best_answer_index=best_answer_index,
                best_answer_confidence=best_confidence,
                semantic_reasoning=semantic_reasoning,
                processing_time=processing_time,
                metadata={
                    'model_used': self.model_name,
                    'morphology_enabled': self.enable_morphology,
                    'context_provided': context is not None,
                    'num_answer_options': len(answer_options)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return self._create_fallback_analysis(question_text, answer_options)
    
    def _generate_semantic_vector(self, text: str, semantic_type: str, index: int = -1) -> SemanticVector:
        """Generate rich semantic vector with morphological enhancement"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{text}_{semantic_type}_{self.model_name}"
        if self.embedding_cache and cache_key in self.embedding_cache:
            cached_vector = self.embedding_cache[cache_key]
            logger.debug(f"Using cached embedding for: {text[:30]}...")
            return cached_vector
        
        # Generate base embedding
        embedding = self.embedding_model.encode([text], show_progress_bar=False)[0]
        
        # Enhance with morphological analysis
        morphological_features = None
        if self.morphology_engine:
            morphological_features = self.morphology_engine.analyze_text(text)
        
        # Calculate confidence based on text quality and analysis completeness
        confidence = self._calculate_vector_confidence(text, embedding, morphological_features)
        
        processing_time = time.time() - start_time
        
        # Create semantic vector
        vector = SemanticVector(
            text=text,
            embedding=embedding,
            morphological_features=morphological_features,
            semantic_type=semantic_type,
            confidence=confidence,
            processing_time=processing_time,
            metadata={
                'index': index,
                'embedding_dimension': len(embedding),
                'morphology_available': morphological_features is not None
            }
        )
        
        # Cache the vector
        if self.embedding_cache:
            self.embedding_cache[cache_key] = vector
        
        return vector
    
    def _calculate_vector_confidence(self, text: str, embedding: np.ndarray, morph_analysis: Optional[SentenceAnalysis]) -> float:
        """Calculate confidence score for semantic vector"""
        confidence = 0.5  # Base confidence
        
        # Text quality factors
        if len(text) >= 3:  # Reasonable length
            confidence += 0.1
        if any(char in text for char in self.semantic_patterns['question_indicators']):
            confidence += 0.1  # Contains semantic indicators
        
        # Embedding quality factors
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0.1:  # Non-trivial embedding
            confidence += 0.1
        
        # Morphological analysis factors
        if morph_analysis:
            confidence += 0.1
            if morph_analysis.confidence > 0.7:
                confidence += 0.1
            if len(morph_analysis.key_phrases) > 0:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_comprehensive_similarity(self, 
                                         question_vector: SemanticVector,
                                         answer_vector: SemanticVector,
                                         context_vector: Optional[SemanticVector] = None) -> SimilarityScore:
        """Calculate comprehensive multi-dimensional similarity"""
        
        # 1. Semantic similarity (embedding-based)
        semantic_sim = self._calculate_semantic_similarity(question_vector, answer_vector)
        
        # 2. Lexical similarity (surface form)
        lexical_sim = self._calculate_lexical_similarity(question_vector.text, answer_vector.text)
        
        # 3. Syntactic similarity (structural)
        syntactic_sim = self._calculate_syntactic_similarity(question_vector, answer_vector)
        
        # 4. Morphological similarity (POS patterns)
        morphological_sim = self._calculate_morphological_similarity(question_vector, answer_vector)
        
        # 5. Context-enhanced similarity (if context available)
        context_enhancement = 0.0
        if context_vector:
            context_enhancement = self._calculate_context_enhancement(
                question_vector, answer_vector, context_vector
            )
        
        # Weighted combination of similarity dimensions
        weights = {
            'semantic': 0.4,
            'lexical': 0.2,
            'syntactic': 0.2,
            'morphological': 0.15,
            'context': 0.05
        }
        
        overall_score = (
            weights['semantic'] * semantic_sim +
            weights['lexical'] * lexical_sim +
            weights['syntactic'] * syntactic_sim +
            weights['morphological'] * morphological_sim +
            weights['context'] * context_enhancement
        )
        
        # Calculate confidence based on consistency across dimensions
        score_variance = np.var([semantic_sim, lexical_sim, syntactic_sim, morphological_sim])
        confidence = max(0.3, 1.0 - score_variance)
        
        # Generate explanation
        explanation = self._generate_similarity_explanation(
            overall_score, semantic_sim, lexical_sim, syntactic_sim, morphological_sim
        )
        
        return SimilarityScore(
            overall_score=overall_score,
            semantic_similarity=semantic_sim,
            lexical_similarity=lexical_sim,
            syntactic_similarity=syntactic_sim,
            morphological_similarity=morphological_sim,
            confidence=confidence,
            explanation=explanation,
            evidence={
                'context_enhancement': context_enhancement,
                'score_variance': score_variance,
                'weights_used': weights
            }
        )
    
    def _calculate_semantic_similarity(self, vector1: SemanticVector, vector2: SemanticVector) -> float:
        """Calculate embedding-based semantic similarity"""
        if SKLEARN_AVAILABLE:
            similarity = cosine_similarity([vector1.embedding], [vector2.embedding])[0][0]
            # Normalize to 0-1 range
            return max(0.0, min(1.0, (similarity + 1) / 2))
        else:
            # Fallback dot product similarity
            dot_product = np.dot(vector1.embedding, vector2.embedding)
            norms = np.linalg.norm(vector1.embedding) * np.linalg.norm(vector2.embedding)
            return max(0.0, min(1.0, dot_product / norms if norms > 0 else 0.0))
    
    def _calculate_lexical_similarity(self, text1: str, text2: str) -> float:
        """Calculate surface form lexical similarity"""
        if self.sequence_matcher:
            similarity = self.sequence_matcher(None, text1, text2).ratio()
            return similarity
        else:
            # Simple character overlap fallback
            chars1 = set(text1)
            chars2 = set(text2)
            if not chars1 and not chars2:
                return 1.0
            elif not chars1 or not chars2:
                return 0.0
            else:
                intersection = len(chars1 & chars2)
                union = len(chars1 | chars2)
                return intersection / union
    
    def _calculate_syntactic_similarity(self, vector1: SemanticVector, vector2: SemanticVector) -> float:
        """Calculate structural/syntactic similarity"""
        if not vector1.morphological_features or not vector2.morphological_features:
            return 0.5  # Neutral if no morphological data
        
        morph1 = vector1.morphological_features
        morph2 = vector2.morphological_features
        
        similarity_factors = []
        
        # Sentence type similarity
        if morph1.sentence_type == morph2.sentence_type:
            similarity_factors.append(1.0)
        else:
            similarity_factors.append(0.3)
        
        # Complexity similarity
        complexity_diff = abs(morph1.complexity_score - morph2.complexity_score)
        complexity_sim = max(0.0, 1.0 - complexity_diff)
        similarity_factors.append(complexity_sim)
        
        # Token count similarity
        token_ratio = min(len(morph1.tokens), len(morph2.tokens)) / max(len(morph1.tokens), len(morph2.tokens))
        similarity_factors.append(token_ratio)
        
        return sum(similarity_factors) / len(similarity_factors)
    
    def _calculate_morphological_similarity(self, vector1: SemanticVector, vector2: SemanticVector) -> float:
        """Calculate part-of-speech and morphological similarity"""
        if not vector1.morphological_features or not vector2.morphological_features:
            return 0.5  # Neutral if no morphological data
        
        morph1 = vector1.morphological_features
        morph2 = vector2.morphological_features
        
        # Extract POS patterns
        pos1 = [token.part_of_speech for token in morph1.tokens if token.part_of_speech]
        pos2 = [token.part_of_speech for token in morph2.tokens if token.part_of_speech]
        
        if not pos1 and not pos2:
            return 1.0
        elif not pos1 or not pos2:
            return 0.0
        
        # Calculate POS overlap
        pos_set1 = set(pos1)
        pos_set2 = set(pos2)
        
        intersection = len(pos_set1 & pos_set2)
        union = len(pos_set1 | pos_set2)
        
        pos_similarity = intersection / union if union > 0 else 0.0
        
        # Calculate key phrase overlap
        phrases1 = set(morph1.key_phrases)
        phrases2 = set(morph2.key_phrases)
        
        phrase_intersection = len(phrases1 & phrases2)
        phrase_union = len(phrases1 | phrases2)
        
        phrase_similarity = phrase_intersection / phrase_union if phrase_union > 0 else 0.0
        
        # Weighted combination
        return 0.7 * pos_similarity + 0.3 * phrase_similarity
    
    def _calculate_context_enhancement(self, 
                                     question_vector: SemanticVector,
                                     answer_vector: SemanticVector,
                                     context_vector: SemanticVector) -> float:
        """Calculate context-based similarity enhancement"""
        # Calculate how well the answer fits within the provided context
        context_question_sim = self._calculate_semantic_similarity(context_vector, question_vector)
        context_answer_sim = self._calculate_semantic_similarity(context_vector, answer_vector)
        
        # Enhancement factor based on context coherence
        enhancement = (context_question_sim + context_answer_sim) / 2
        
        return enhancement * 0.5  # Scale down as it's an enhancement factor
    
    def _generate_similarity_explanation(self, overall: float, semantic: float, lexical: float, 
                                       syntactic: float, morphological: float) -> str:
        """Generate human-readable explanation of similarity scoring"""
        explanations = []
        
        if overall > 0.8:
            explanations.append("Very high semantic match")
        elif overall > 0.6:
            explanations.append("Good semantic match")
        elif overall > 0.4:
            explanations.append("Moderate semantic match")
        else:
            explanations.append("Low semantic match")
        
        if semantic > 0.7:
            explanations.append("strong embedding similarity")
        if lexical > 0.5:
            explanations.append("good surface form overlap")
        if syntactic > 0.6:
            explanations.append("similar sentence structure")
        if morphological > 0.6:
            explanations.append("compatible grammatical patterns")
        
        return " - ".join(explanations)
    
    def _select_best_answer(self, 
                          question_vector: SemanticVector,
                          answer_vectors: List[SemanticVector],
                          similarity_scores: List[SimilarityScore]) -> Tuple[int, float]:
        """Select best answer based on comprehensive semantic analysis"""
        if not similarity_scores:
            return -1, 0.0
        
        # Primary selection: highest overall similarity score
        best_index = max(range(len(similarity_scores)), 
                        key=lambda i: similarity_scores[i].overall_score)
        best_score = similarity_scores[best_index]
        
        # Confidence adjustment based on score gap and consistency
        score_values = [score.overall_score for score in similarity_scores]
        score_gap = best_score.overall_score - sorted(score_values)[-2] if len(score_values) > 1 else 0.5
        
        # Higher confidence if there's a clear winner
        confidence = best_score.confidence * (1.0 + score_gap)
        confidence = min(confidence, 1.0)
        
        return best_index, confidence
    
    def _generate_semantic_reasoning(self, 
                                   question_vector: SemanticVector,
                                   answer_vectors: List[SemanticVector],
                                   similarity_scores: List[SimilarityScore],
                                   best_answer_index: int) -> List[str]:
        """Generate step-by-step semantic reasoning"""
        reasoning = []
        
        # Question analysis
        if question_vector.morphological_features:
            morph = question_vector.morphological_features
            reasoning.append(f"Question analysis: {morph.sentence_type} sentence with {len(morph.tokens)} tokens")
            if morph.key_phrases:
                reasoning.append(f"Key concepts: {', '.join(morph.key_phrases[:3])}")
        
        # Answer option analysis
        if best_answer_index >= 0:
            best_answer = answer_vectors[best_answer_index]
            best_similarity = similarity_scores[best_answer_index]
            
            reasoning.append(f"Best answer selected: Option {best_answer_index + 1}")
            reasoning.append(f"Semantic similarity: {best_similarity.semantic_similarity:.3f}")
            reasoning.append(f"Overall match score: {best_similarity.overall_score:.3f}")
            reasoning.append(f"Reasoning: {best_similarity.explanation}")
        
        # Comparative analysis
        if len(similarity_scores) > 1:
            scores = [(i, score.overall_score) for i, score in enumerate(similarity_scores)]
            scores.sort(key=lambda x: x[1], reverse=True)
            
            reasoning.append(f"Ranking: " + ", ".join([f"Option {i+1}({score:.2f})" for i, score in scores[:3]]))
        
        return reasoning
    
    def _create_fallback_analysis(self, question_text: str, answer_options: List[str]) -> SemanticAnalysis:
        """Create fallback analysis when embedding model is unavailable"""
        # Simple keyword matching fallback
        question_words = set(question_text.lower().split())
        
        similarity_scores = []
        for i, option in enumerate(answer_options):
            option_words = set(option.lower().split())
            overlap = len(question_words & option_words)
            total = len(question_words | option_words)
            similarity = overlap / total if total > 0 else 0.0
            
            similarity_scores.append(SimilarityScore(
                overall_score=similarity,
                semantic_similarity=similarity,
                lexical_similarity=similarity,
                syntactic_similarity=0.5,
                morphological_similarity=0.5,
                confidence=0.3,  # Low confidence for fallback
                explanation="Simple keyword matching (embedding model unavailable)",
                evidence={"fallback_mode": True}
            ))
        
        # Select best based on keyword overlap
        if similarity_scores:
            best_index = max(range(len(similarity_scores)), 
                           key=lambda i: similarity_scores[i].overall_score)
            best_confidence = similarity_scores[best_index].confidence
        else:
            best_index = -1
            best_confidence = 0.0
        
        # Create fallback vectors
        question_vector = SemanticVector(
            text=question_text,
            embedding=np.zeros(384),  # Dummy embedding
            confidence=0.3
        )
        
        answer_vectors = [
            SemanticVector(text=option, embedding=np.zeros(384), confidence=0.3)
            for option in answer_options
        ]
        
        return SemanticAnalysis(
            question_vector=question_vector,
            answer_vectors=answer_vectors,
            similarity_scores=similarity_scores,
            best_answer_index=best_index,
            best_answer_confidence=best_confidence,
            semantic_reasoning=["Fallback analysis - embedding model unavailable"],
            processing_time=0.001,
            metadata={"fallback_mode": True}
        )

def main():
    """Test the semantic engine"""
    print("🧠 Testing Advanced Semantic Engine...")
    
    # Initialize engine
    engine = AdvancedSemanticEngine(
        enable_morphology=True,
        cache_embeddings=True
    )
    
    # Test questions with multiple choice answers
    test_cases = [
        {
            "question": "今日は何曜日ですか？",
            "answers": ["月曜日", "火曜日", "水曜日", "木曜日"],
            "context": "今日は2025年1月15日です。"
        },
        {
            "question": "次の単語の意味は？ 友達",
            "answers": ["family", "friend", "teacher", "student"],
            "context": None
        },
        {
            "question": "昭和55年は西暦何年ですか？",
            "answers": ["1979年", "1980年", "1981年", "1982年"],
            "context": "昭和は1926年から1989年まで続いた時代です。"
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"🔍 Test Case {i+1}: {test_case['question']}")
        print(f"📋 Options: {test_case['answers']}")
        print("-" * 60)
        
        # Perform semantic analysis
        analysis = engine.analyze_question_answers(
            test_case['question'],
            test_case['answers'],
            test_case['context']
        )
        
        print(f"🎯 Best Answer: Option {analysis.best_answer_index + 1} - '{test_case['answers'][analysis.best_answer_index]}'")
        print(f"🔮 Confidence: {analysis.best_answer_confidence:.3f}")
        print(f"⚡ Processing Time: {analysis.processing_time:.3f}s")
        
        print("\n💭 Semantic Reasoning:")
        for j, reason in enumerate(analysis.semantic_reasoning):
            print(f"  {j+1}. {reason}")
        
        print("\n📊 Similarity Scores:")
        for j, score in enumerate(analysis.similarity_scores):
            print(f"  Option {j+1}: {score.overall_score:.3f} ({score.explanation})")
            print(f"    Semantic: {score.semantic_similarity:.3f}, Lexical: {score.lexical_similarity:.3f}")
            print(f"    Syntactic: {score.syntactic_similarity:.3f}, Morphological: {score.morphological_similarity:.3f}")

if __name__ == "__main__":
    main()
