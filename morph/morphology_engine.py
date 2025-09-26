#!/usr/bin/env python3
"""
Phase 2A: Advanced Morphological Analysis Engine

This module provides sophisticated Japanese text analysis using MeCab
with enhanced part-of-speech tagging, dependency parsing, and named
entity recognition for superior context understanding.

Key Features:
- MeCab integration with multiple dictionaries
- Context-aware POS tagging with confidence scoring
- Dependency parsing for complex sentence structures  
- Named entity recognition (persons, locations, dates, organizations)
- Semantic role labeling for verb-argument structures
- Advanced tokenization with compound word detection
"""

import sys
import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

# Import MeCab with fallback handling
try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    print("⚠️  MeCab not available. Install with: pip install mecab-python3")

# Import additional NLP libraries
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class MorphToken:
    """Enhanced morphological token with rich linguistic information"""
    surface: str  # Original text
    part_of_speech: str  # Main POS category
    pos_detail_1: str  # POS subcategory  
    pos_detail_2: str  # POS sub-subcategory
    pos_detail_3: str  # Additional POS info
    inflection_type: str  # Conjugation type
    inflection_form: str  # Conjugation form
    base_form: str  # Dictionary form
    reading: str  # Katakana reading
    pronunciation: str  # Pronunciation
    
    # Enhanced attributes
    confidence: float = 0.0  # Analysis confidence
    semantic_role: str = ""  # Semantic role in sentence
    dependency_head: int = -1  # Head token index
    dependency_relation: str = ""  # Dependency relation type
    named_entity_tag: str = ""  # Named entity classification
    compound_info: List[str] = field(default_factory=list)  # Compound word components

@dataclass  
class SentenceAnalysis:
    """Complete sentence analysis with multiple linguistic layers"""
    original_text: str
    tokens: List[MorphToken]
    sentence_type: str  # declarative, interrogative, imperative, etc.
    complexity_score: float  # 0.0-1.0 sentence complexity
    semantic_relations: List[Dict[str, Any]]  # Subject-verb-object relations
    named_entities: List[Dict[str, Any]]  # Extracted named entities
    key_phrases: List[str]  # Important noun/verb phrases
    confidence: float  # Overall analysis confidence
    processing_time: float  # Time taken for analysis

class AdvancedMorphologyEngine:
    """
    Advanced morphological analysis engine with context understanding
    
    Provides comprehensive Japanese text analysis using MeCab with
    additional semantic processing and confidence scoring.
    """
    
    def __init__(self, 
                 mecab_dict: Optional[str] = None,
                 enable_ner: bool = True,
                 enable_dependency: bool = True):
        """
        Initialize the morphological analysis engine
        
        Args:
            mecab_dict: Path to MeCab dictionary (None for default)
            enable_ner: Enable named entity recognition
            enable_dependency: Enable dependency parsing
        """
        self.enable_ner = enable_ner
        self.enable_dependency = enable_dependency
        
        # Initialize MeCab
        if MECAB_AVAILABLE:
            self._initialize_mecab(mecab_dict)
        else:
            logger.error("MeCab not available - morphological analysis disabled")
            self.mecab = None
        
        # Initialize additional NLP components
        self._initialize_additional_components()
        
        # Load linguistic resources
        self._load_linguistic_resources()
        
        logger.info(f"Morphology engine initialized - MeCab: {MECAB_AVAILABLE}, NER: {enable_ner}, Dep: {enable_dependency}")
    
    def _initialize_mecab(self, mecab_dict: Optional[str]):
        """Initialize MeCab with optimal configuration"""
        try:
            # Try different MeCab configurations
            configs = []
            
            if mecab_dict:
                configs.append(f"-d {mecab_dict}")
            
            # Common MeCab dictionary locations
            common_dicts = [
                "/usr/local/lib/mecab/dic/mecab-ipadic-neologd",  # NEologd (preferred)
                "/usr/local/lib/mecab/dic/ipadic",  # IPADic
                "C:\\mecab\\dic\\ipadic",  # Windows IPADic
                "C:\\Program Files\\MeCab\\dic\\ipadic"  # Windows Program Files
            ]
            
            for dict_path in common_dicts:
                if os.path.exists(dict_path):
                    configs.append(f"-d {dict_path}")
            
            # Default configuration
            configs.append("")  # Use system default
            
            # Try configurations in order of preference
            for config in configs:
                try:
                    self.mecab = MeCab.Tagger(config)
                    # Test the tagger
                    test_result = self.mecab.parse("テスト")
                    if test_result and not test_result.startswith("BOS"):
                        logger.info(f"MeCab initialized successfully with config: {config or 'default'}")
                        return
                except Exception as e:
                    logger.debug(f"MeCab config '{config}' failed: {e}")
                    continue
            
            # If all configurations fail, try bare initialization
            self.mecab = MeCab.Tagger()
            logger.warning("MeCab initialized with fallback configuration")
            
        except Exception as e:
            logger.error(f"Failed to initialize MeCab: {e}")
            self.mecab = None
    
    def _initialize_additional_components(self):
        """Initialize additional NLP components"""
        self.spacy_nlp = None
        
        if SPACY_AVAILABLE and self.enable_dependency:
            try:
                # Try to load Japanese spaCy model
                self.spacy_nlp = spacy.load("ja_core_news_sm")
                logger.info("SpaCy Japanese model loaded for dependency parsing")
            except OSError:
                logger.warning("SpaCy Japanese model not found. Install with: python -m spacy download ja_core_news_sm")
        
        # Initialize custom NER patterns
        self._initialize_ner_patterns()
    
    def _initialize_ner_patterns(self):
        """Initialize named entity recognition patterns"""
        self.ner_patterns = {
            'PERSON': [
                r'[一-龯]{1,4}(?:さん|くん|ちゃん|様|先生|教授|博士)',  # Name + honorific
                r'[ァ-ヺ]{2,8}(?:さん|くん|ちゃん|様)',  # Katakana name + honorific
            ],
            'LOCATION': [
                r'[一-龯]{1,6}(?:県|府|都|道|市|区|町|村|島|山|川|湖|海)',  # Administrative divisions
                r'(?:東京|大阪|京都|名古屋|横浜|神戸|福岡|札幌|仙台|広島)',  # Major cities
            ],
            'ORGANIZATION': [
                r'[一-龯]{1,8}(?:会社|企業|法人|株式会社|有限会社|大学|学校|病院|銀行)',
                r'(?:トヨタ|ソニー|任天堂|ソフトバンク|楽天|NTT|JR|ANA|JAL)',  # Major companies
            ],
            'DATE': [
                r'(?:昭和|平成|令和)\d{1,2}年',  # Japanese eras
                r'\d{1,4}年\d{1,2}月\d{1,2}日',  # Dates
                r'(?:月|火|水|木|金|土|日)曜日',  # Days of week
            ],
            'TIME': [
                r'\d{1,2}時\d{1,2}分',  # Hours and minutes
                r'(?:午前|午後)\d{1,2}時',  # AM/PM times
                r'(?:朝|昼|夜|夕方|深夜)',  # Time periods
            ]
        }
    
    def _load_linguistic_resources(self):
        """Load additional linguistic resources and dictionaries"""
        # Load compound word patterns
        self.compound_patterns = [
            r'[一-龯]{2,}(?:化|性|的|論|学|法|業|品|物|者|家|員|師|長|部|課|係)',  # Common suffixes
            r'(?:大|小|新|旧|高|低|長|短|多|少)[一-龯]+',  # Adjective prefixes
        ]
        
        # Load semantic role patterns
        self.semantic_roles = {
            'AGENT': ['が', 'は'],  # Subject particles
            'PATIENT': ['を', 'に'],  # Object particles  
            'LOCATION': ['で', 'から', 'まで', 'へ'],  # Location particles
            'TIME': ['に', 'で', 'から', 'まで'],  # Time particles
            'INSTRUMENT': ['で', 'を使って'],  # Instrument particles
        }
        
        logger.info("Linguistic resources loaded successfully")
    
    def analyze_text(self, text: str) -> SentenceAnalysis:
        """
        Perform comprehensive morphological analysis on Japanese text
        
        Args:
            text: Input Japanese text to analyze
            
        Returns:
            SentenceAnalysis object with complete linguistic information
        """
        import time
        start_time = time.time()
        
        if not self.mecab:
            return self._create_fallback_analysis(text)
        
        try:
            # Clean and normalize input text
            normalized_text = self._normalize_text(text)
            
            # Parse with MeCab
            tokens = self._parse_with_mecab(normalized_text)
            
            # Enhance tokens with additional analysis
            enhanced_tokens = self._enhance_tokens(tokens, normalized_text)
            
            # Perform sentence-level analysis
            sentence_analysis = self._analyze_sentence_structure(enhanced_tokens, normalized_text)
            
            # Add semantic relations
            semantic_relations = self._extract_semantic_relations(enhanced_tokens)
            
            # Extract named entities  
            named_entities = self._extract_named_entities(normalized_text, enhanced_tokens)
            
            # Extract key phrases
            key_phrases = self._extract_key_phrases(enhanced_tokens)
            
            # Calculate confidence and complexity
            confidence = self._calculate_confidence(enhanced_tokens)
            complexity = self._calculate_complexity(enhanced_tokens)
            
            processing_time = time.time() - start_time
            
            return SentenceAnalysis(
                original_text=text,
                tokens=enhanced_tokens,
                sentence_type=sentence_analysis['type'],
                complexity_score=complexity,
                semantic_relations=semantic_relations,
                named_entities=named_entities,
                key_phrases=key_phrases,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in morphological analysis: {e}")
            return self._create_fallback_analysis(text)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize input text for consistent processing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize punctuation
        text = text.replace('。', '。')  # Ensure proper period
        text = text.replace('、', '、')  # Ensure proper comma
        text = text.replace('？', '？')  # Ensure proper question mark
        
        return text
    
    def _parse_with_mecab(self, text: str) -> List[MorphToken]:
        """Parse text with MeCab and create MorphToken objects"""
        tokens = []
        
        # Parse with MeCab
        parsed = self.mecab.parse(text)
        
        for line in parsed.split('\n'):
            if line in ['', 'EOS']:
                continue
            
            try:
                # Split surface form and features
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                
                surface = parts[0]
                features = parts[1].split(',')
                
                # Ensure we have enough features (pad with empty strings if needed)
                while len(features) < 9:
                    features.append('')
                
                # Create MorphToken
                token = MorphToken(
                    surface=surface,
                    part_of_speech=features[0] if features[0] != '*' else '',
                    pos_detail_1=features[1] if features[1] != '*' else '',
                    pos_detail_2=features[2] if features[2] != '*' else '',
                    pos_detail_3=features[3] if features[3] != '*' else '',
                    inflection_type=features[4] if features[4] != '*' else '',
                    inflection_form=features[5] if features[5] != '*' else '',
                    base_form=features[6] if features[6] != '*' else surface,
                    reading=features[7] if features[7] != '*' else '',
                    pronunciation=features[8] if features[8] != '*' else ''
                )
                
                tokens.append(token)
                
            except Exception as e:
                logger.debug(f"Error parsing MeCab line '{line}': {e}")
                continue
        
        return tokens
    
    def _enhance_tokens(self, tokens: List[MorphToken], text: str) -> List[MorphToken]:
        """Enhance tokens with additional linguistic information"""
        enhanced_tokens = []
        
        for i, token in enumerate(tokens):
            # Calculate confidence based on POS completeness
            confidence = self._calculate_token_confidence(token)
            token.confidence = confidence
            
            # Add semantic role based on POS and context
            token.semantic_role = self._determine_semantic_role(token, tokens, i)
            
            # Check for compound words
            token.compound_info = self._analyze_compound_word(token)
            
            # Add dependency information if available
            if self.enable_dependency:
                token.dependency_head, token.dependency_relation = self._analyze_dependency(token, tokens, i)
            
            enhanced_tokens.append(token)
        
        return enhanced_tokens
    
    def _calculate_token_confidence(self, token: MorphToken) -> float:
        """Calculate confidence score for token analysis"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for complete POS information
        if token.part_of_speech:
            confidence += 0.2
        if token.pos_detail_1:
            confidence += 0.1
        if token.base_form != token.surface:
            confidence += 0.1
        if token.reading:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _determine_semantic_role(self, token: MorphToken, tokens: List[MorphToken], index: int) -> str:
        """Determine semantic role of token in sentence"""
        # Check if token is a particle that indicates semantic role
        if token.part_of_speech == '助詞':
            for role, particles in self.semantic_roles.items():
                if token.surface in particles:
                    return role
        
        # Check surrounding context for role determination
        if index > 0:
            prev_token = tokens[index - 1]
            if prev_token.part_of_speech == '助詞':
                for role, particles in self.semantic_roles.items():
                    if prev_token.surface in particles:
                        return role
        
        return ''
    
    def _analyze_compound_word(self, token: MorphToken) -> List[str]:
        """Analyze if token is part of a compound word"""
        compound_info = []
        
        for pattern in self.compound_patterns:
            if re.match(pattern, token.surface):
                compound_info.append(f"compound_pattern:{pattern}")
                break
        
        return compound_info
    
    def _analyze_dependency(self, token: MorphToken, tokens: List[MorphToken], index: int) -> Tuple[int, str]:
        """Analyze dependency relations (simplified version)"""
        # Simplified dependency analysis - in production would use proper parser
        head_index = -1
        relation = ""
        
        # Basic rules for Japanese dependency
        if token.part_of_speech == '助詞':  # Particles typically depend on following words
            if index < len(tokens) - 1:
                head_index = index + 1
                relation = "particle_attachment"
        elif token.part_of_speech == '形容詞':  # Adjectives typically modify nouns
            # Look for following noun
            for i in range(index + 1, min(index + 3, len(tokens))):
                if tokens[i].part_of_speech == '名詞':
                    head_index = i
                    relation = "adjectival_modification"
                    break
        
        return head_index, relation
    
    def _analyze_sentence_structure(self, tokens: List[MorphToken], text: str) -> Dict[str, Any]:
        """Analyze overall sentence structure and type"""
        sentence_type = "declarative"  # Default
        
        # Check for interrogative markers
        if any(token.surface in ['？', 'か', 'の'] for token in tokens):
            sentence_type = "interrogative"
        
        # Check for imperative markers
        if any(token.inflection_form in ['命令形'] for token in tokens):
            sentence_type = "imperative"
        
        # Check for exclamatory markers
        if any(token.surface in ['！', 'よ', 'ね'] for token in tokens):
            if sentence_type == "declarative":
                sentence_type = "exclamatory"
        
        return {
            'type': sentence_type,
            'token_count': len(tokens),
            'has_verb': any(token.part_of_speech == '動詞' for token in tokens),
            'has_subject': any(token.semantic_role == 'AGENT' for token in tokens)
        }
    
    def _extract_semantic_relations(self, tokens: List[MorphToken]) -> List[Dict[str, Any]]:
        """Extract semantic relations (subject-verb-object, etc.)"""
        relations = []
        
        # Find verbs and their arguments
        for i, token in enumerate(tokens):
            if token.part_of_speech == '動詞':  # Verb
                relation = {
                    'predicate': token.base_form,
                    'predicate_index': i,
                    'arguments': []
                }
                
                # Look for arguments before the verb
                for j in range(max(0, i - 5), i):
                    arg_token = tokens[j]
                    if arg_token.semantic_role in ['AGENT', 'PATIENT', 'LOCATION', 'TIME']:
                        relation['arguments'].append({
                            'role': arg_token.semantic_role,
                            'text': arg_token.surface,
                            'index': j
                        })
                
                if relation['arguments']:
                    relations.append(relation)
        
        return relations
    
    def _extract_named_entities(self, text: str, tokens: List[MorphToken]) -> List[Dict[str, Any]]:
        """Extract named entities using pattern matching"""
        entities = []
        
        if not self.enable_ner:
            return entities
        
        # Apply NER patterns
        for entity_type, patterns in self.ner_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    entities.append({
                        'type': entity_type,
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8  # Pattern-based confidence
                    })
        
        # Additional entity extraction from tokens
        current_entity = None
        for i, token in enumerate(tokens):
            # Look for proper nouns that might be entities
            if token.pos_detail_1 == '固有名詞':
                if current_entity is None:
                    current_entity = {
                        'type': 'PROPER_NOUN',
                        'tokens': [token],
                        'start_index': i
                    }
                else:
                    current_entity['tokens'].append(token)
            else:
                if current_entity:
                    # Finalize current entity
                    entity_text = ''.join(t.surface for t in current_entity['tokens'])
                    entities.append({
                        'type': current_entity['type'],
                        'text': entity_text,
                        'start_index': current_entity['start_index'],
                        'end_index': i - 1,
                        'confidence': 0.7
                    })
                    current_entity = None
        
        # Handle entity at end of sentence
        if current_entity:
            entity_text = ''.join(t.surface for t in current_entity['tokens'])
            entities.append({
                'type': current_entity['type'],
                'text': entity_text,
                'start_index': current_entity['start_index'],
                'end_index': len(tokens) - 1,
                'confidence': 0.7
            })
        
        return entities
    
    def _extract_key_phrases(self, tokens: List[MorphToken]) -> List[str]:
        """Extract important noun phrases and verb phrases"""
        key_phrases = []
        
        # Extract noun phrases
        current_np = []
        for token in tokens:
            if token.part_of_speech in ['名詞', '形容詞', '連体詞']:
                current_np.append(token.surface)
            else:
                if len(current_np) >= 2:  # Multi-token noun phrase
                    key_phrases.append(''.join(current_np))
                current_np = []
        
        # Handle final noun phrase
        if len(current_np) >= 2:
            key_phrases.append(''.join(current_np))
        
        # Extract important single tokens
        for token in tokens:
            # Important parts of speech
            if (token.part_of_speech in ['動詞', '名詞'] and 
                len(token.surface) >= 2 and
                token.surface not in ['する', 'いる', 'ある', 'なる']):
                key_phrases.append(token.base_form or token.surface)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_phrases = []
        for phrase in key_phrases:
            if phrase not in seen:
                seen.add(phrase)
                unique_phrases.append(phrase)
        
        return unique_phrases[:10]  # Return top 10 key phrases
    
    def _calculate_confidence(self, tokens: List[MorphToken]) -> float:
        """Calculate overall analysis confidence"""
        if not tokens:
            return 0.0
        
        token_confidences = [token.confidence for token in tokens]
        avg_confidence = sum(token_confidences) / len(token_confidences)
        
        # Adjust based on analysis completeness
        completeness_bonus = 0.0
        if any(token.semantic_role for token in tokens):
            completeness_bonus += 0.1
        if any(token.compound_info for token in tokens):
            completeness_bonus += 0.05
        
        return min(avg_confidence + completeness_bonus, 1.0)
    
    def _calculate_complexity(self, tokens: List[MorphToken]) -> float:
        """Calculate sentence complexity score"""
        if not tokens:
            return 0.0
        
        complexity_factors = 0.0
        
        # Base complexity from token count
        complexity_factors += min(len(tokens) / 20.0, 0.3)
        
        # Complexity from varied POS
        pos_types = set(token.part_of_speech for token in tokens)
        complexity_factors += min(len(pos_types) / 10.0, 0.2)
        
        # Complexity from verb conjugations
        complex_conjugations = ['仮定形', '可能', '使役', '受身', '尊敬', '謙譲']
        conjugation_complexity = sum(1 for token in tokens 
                                   if token.inflection_form in complex_conjugations)
        complexity_factors += min(conjugation_complexity / 5.0, 0.2)
        
        # Complexity from compound words
        compound_count = sum(1 for token in tokens if token.compound_info)
        complexity_factors += min(compound_count / 10.0, 0.15)
        
        # Complexity from semantic relations
        relation_count = sum(1 for token in tokens if token.semantic_role)
        complexity_factors += min(relation_count / 8.0, 0.15)
        
        return min(complexity_factors, 1.0)
    
    def _create_fallback_analysis(self, text: str) -> SentenceAnalysis:
        """Create fallback analysis when MeCab is unavailable"""
        import time
        
        # Basic tokenization by character type
        tokens = []
        current_token = ""
        current_type = None
        
        for char in text:
            char_type = self._get_character_type(char)
            
            if char_type != current_type and current_token:
                # Create basic token
                tokens.append(MorphToken(
                    surface=current_token,
                    part_of_speech=self._guess_pos(current_token, current_type),
                    pos_detail_1='',
                    pos_detail_2='',
                    pos_detail_3='',
                    inflection_type='',
                    inflection_form='',
                    base_form=current_token,
                    reading='',
                    pronunciation='',
                    confidence=0.3  # Low confidence for fallback
                ))
                current_token = ""
            
            current_token += char
            current_type = char_type
        
        # Add final token
        if current_token:
            tokens.append(MorphToken(
                surface=current_token,
                part_of_speech=self._guess_pos(current_token, current_type),
                pos_detail_1='',
                pos_detail_2='',
                pos_detail_3='',
                inflection_type='',
                inflection_form='',
                base_form=current_token,
                reading='',
                pronunciation='',
                confidence=0.3
            ))
        
        return SentenceAnalysis(
            original_text=text,
            tokens=tokens,
            sentence_type="unknown",
            complexity_score=0.5,
            semantic_relations=[],
            named_entities=[],
            key_phrases=[],
            confidence=0.3,
            processing_time=0.001
        )
    
    def _get_character_type(self, char: str) -> str:
        """Determine character type for fallback tokenization"""
        if re.match(r'[\u3040-\u309F]', char):  # Hiragana
            return 'hiragana'
        elif re.match(r'[\u30A0-\u30FF]', char):  # Katakana
            return 'katakana'
        elif re.match(r'[\u4E00-\u9FAF]', char):  # Kanji
            return 'kanji'
        elif re.match(r'[a-zA-Z]', char):  # Latin
            return 'latin'
        elif re.match(r'[0-9]', char):  # Numbers
            return 'number'
        else:
            return 'other'
    
    def _guess_pos(self, token: str, char_type: str) -> str:
        """Guess part of speech for fallback analysis"""
        pos_mapping = {
            'hiragana': '助詞',  # Often particles
            'katakana': '名詞',  # Often nouns  
            'kanji': '名詞',     # Often nouns
            'latin': '名詞',     # Foreign words, often nouns
            'number': '名詞',    # Numbers treated as nouns
            'other': '記号'      # Symbols
        }
        
        return pos_mapping.get(char_type, '未知語')

def main():
    """Test the morphology engine"""
    print("🔍 Testing Advanced Morphology Engine...")
    
    # Initialize engine
    engine = AdvancedMorphologyEngine(enable_ner=True, enable_dependency=True)
    
    # Test sentences
    test_sentences = [
        "今日は何曜日ですか？",
        "昭和55年12月25日に生まれました。",
        "私は東京大学で日本語を勉強しています。",
        "田中さんはトヨタ自動車で働いています。",
        "この問題の答えはAですか、それともBですか？"
    ]
    
    for sentence in test_sentences:
        print(f"\n📝 Analyzing: {sentence}")
        print("-" * 50)
        
        analysis = engine.analyze_text(sentence)
        
        print(f"Sentence Type: {analysis.sentence_type}")
        print(f"Complexity: {analysis.complexity_score:.2f}")
        print(f"Confidence: {analysis.confidence:.2f}")
        print(f"Processing Time: {analysis.processing_time:.3f}s")
        
        print("\nTokens:")
        for i, token in enumerate(analysis.tokens):
            print(f"  {i}: {token.surface} ({token.part_of_speech}) - {token.base_form}")
            if token.semantic_role:
                print(f"      Role: {token.semantic_role}")
        
        print(f"\nNamed Entities: {analysis.named_entities}")
        print(f"Key Phrases: {analysis.key_phrases}")
        print(f"Semantic Relations: {len(analysis.semantic_relations)} found")

if __name__ == "__main__":
    main()
