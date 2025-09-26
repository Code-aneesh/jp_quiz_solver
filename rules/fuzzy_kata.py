"""
Katakana Fuzzy Matching Module

This module provides fuzzy string matching specifically optimized for katakana
text that may contain OCR errors. Uses difflib and Levenshtein distance
to find the best matches when OCR produces noisy katakana.

Key features:
- Katakana-specific similarity scoring
- OCR error pattern recognition
- Multiple similarity algorithms (difflib, Levenshtein)
- Confidence thresholding for quality control
- Common katakana OCR error mappings
"""

import re
import difflib
import logging
from typing import List, Optional, Tuple, Dict, NamedTuple
from dataclasses import dataclass

# Try to import Levenshtein for better fuzzy matching
try:
    from Levenshtein import distance as levenshtein_distance
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    logging.warning("Levenshtein package not available. Using difflib only.")

logger = logging.getLogger(__name__)


@dataclass
class FuzzyMatch:
    """Result from fuzzy matching"""
    original_text: str
    matched_option: str
    similarity_score: float
    algorithm_used: str
    confidence: float
    error_patterns: List[str]


class KatakanaFuzzyMatcher:
    """
    Fuzzy matcher specialized for katakana text with OCR errors
    """
    
    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold
        self.common_ocr_errors = self._init_ocr_error_patterns()
        logger.info(f"Katakana Fuzzy Matcher initialized (threshold: {similarity_threshold})")
    
    def _init_ocr_error_patterns(self) -> Dict[str, str]:
        """
        Initialize common katakana OCR error patterns.
        Maps commonly misread characters to correct forms.
        """
        return {
            # Similar looking katakana characters
            "ソ": "ン",  # ソ (so) often misread as ン (n)
            "ン": "ソ",  # And vice versa
            "シ": "ツ",  # シ (shi) vs ツ (tsu) 
            "ツ": "シ",
            "ニ": "二",  # ニ (ni) vs 二 (number 2)
            "二": "ニ",
            "ロ": "コ",  # ロ (ro) vs コ (ko)
            "コ": "ロ",
            "ヘ": "ペ",  # ヘ (he) vs ペ (pe) - dakuten confusion
            "ペ": "ヘ",
            "ケ": "ゲ",  # ケ (ke) vs ゲ (ge)
            "ゲ": "ケ",
            "タ": "ダ",  # タ (ta) vs ダ (da)
            "ダ": "タ",
            "カ": "ガ",  # カ (ka) vs ガ (ga) 
            "ガ": "カ",
            "ハ": "バ",  # ハ (ha) vs バ (ba)
            "バ": "ハ",
            "パ": "ハ",  # パ (pa) vs ハ (ha) - dakuten/handakuten confusion
            "ハ": "パ",
            # Common character boundary errors
            "ア": "マ",  # ア (a) vs マ (ma)
            "マ": "ア",
            "ウ": "ラ",  # ウ (u) vs ラ (ra)
            "ラ": "ウ",
            "エ": "工",  # エ (e) vs 工 (工 - construction kanji)
            "工": "エ",
            # Small vs large tsu
            "ツ": "ッ",  # ツ vs small ッ
            "ッ": "ツ",
        }
    
    def find_best_katakana_match(self, katakana_text: str, options: List[str]) -> Optional[FuzzyMatch]:
        """
        Find the best fuzzy match for katakana text among the provided options.
        
        Args:
            katakana_text: OCR detected katakana text (may contain errors)
            options: List of option strings to match against
            
        Returns:
            FuzzyMatch object if a good match is found, None otherwise
        """
        if not katakana_text or not options:
            return None
        
        # Extract katakana from input text
        clean_katakana = self._extract_katakana(katakana_text)
        if not clean_katakana:
            logger.debug("No katakana found in input text")
            return None
        
        logger.info(f"Fuzzy matching katakana: '{clean_katakana}' against {len(options)} options")
        
        best_matches = []
        
        # Test each option
        for option in options:
            option_katakana = self._extract_katakana(option)
            if not option_katakana:
                continue
            
            # Calculate similarity using multiple algorithms
            similarities = self._calculate_similarities(clean_katakana, option_katakana)
            
            for algo_name, similarity in similarities.items():
                if similarity >= self.similarity_threshold:
                    # Detect error patterns
                    error_patterns = self._detect_error_patterns(clean_katakana, option_katakana)
                    
                    # Calculate confidence based on similarity and error patterns
                    confidence = self._calculate_confidence(similarity, error_patterns, clean_katakana, option_katakana)
                    
                    match = FuzzyMatch(
                        original_text=clean_katakana,
                        matched_option=option,
                        similarity_score=similarity,
                        algorithm_used=algo_name,
                        confidence=confidence,
                        error_patterns=error_patterns
                    )
                    best_matches.append(match)
                    
                    logger.debug(f"Match found: '{clean_katakana}' -> '{option}' "
                               f"(similarity: {similarity:.3f}, confidence: {confidence:.3f})")
        
        if not best_matches:
            logger.info("No fuzzy matches found above threshold")
            return None
        
        # Return best match (highest confidence, then highest similarity)
        best_match = max(best_matches, key=lambda m: (m.confidence, m.similarity_score))
        
        logger.info(f"Best katakana match: '{best_match.original_text}' -> '{best_match.matched_option}' "
                   f"(similarity: {best_match.similarity_score:.3f}, confidence: {best_match.confidence:.3f})")
        
        return best_match
    
    def _extract_katakana(self, text: str) -> str:
        """
        Extract katakana characters from mixed text.
        
        Args:
            text: Input text that may contain mixed characters
            
        Returns:
            String containing only katakana characters
        """
        # Katakana Unicode range: U+30A0-U+30FF
        katakana_pattern = r'[\u30A0-\u30FF]+'
        katakana_matches = re.findall(katakana_pattern, text)
        
        # Join all katakana segments
        return ''.join(katakana_matches)
    
    def _calculate_similarities(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Calculate similarity using multiple algorithms.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Dictionary of similarity scores by algorithm name
        """
        similarities = {}
        
        # Sequence matcher (difflib)
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        similarities['difflib'] = similarity
        
        # Levenshtein distance (if available)
        if LEVENSHTEIN_AVAILABLE:
            max_len = max(len(text1), len(text2))
            if max_len > 0:
                # Convert distance to similarity ratio
                distance = levenshtein_distance(text1, text2)
                similarity = 1.0 - (distance / max_len)
                similarities['levenshtein'] = similarity
        
        # Jaccard similarity (character-based)
        set1 = set(text1)
        set2 = set(text2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard = intersection / union if union > 0 else 0.0
        similarities['jaccard'] = jaccard
        
        # Jaro-Winkler similarity (approximation using difflib)
        jaro_sim = self._jaro_similarity(text1, text2)
        similarities['jaro'] = jaro_sim
        
        return similarities
    
    def _jaro_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate Jaro similarity (simplified implementation).
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Jaro similarity score (0.0 to 1.0)
        """
        if not s1 or not s2:
            return 0.0
        
        if s1 == s2:
            return 1.0
        
        # Simplified Jaro similarity calculation
        len1, len2 = len(s1), len(s2)
        max_distance = max(len1, len2) // 2 - 1
        
        if max_distance < 1:
            return 0.0
        
        # Find matches
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        matches = 0
        transpositions = 0
        
        # Identify matches
        for i in range(len1):
            start = max(0, i - max_distance)
            end = min(i + max_distance + 1, len2)
            
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        # Count transpositions
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
        return jaro
    
    def _detect_error_patterns(self, original: str, target: str) -> List[str]:
        """
        Detect common OCR error patterns between two katakana strings.
        
        Args:
            original: Original (potentially OCR'd) text
            target: Target correct text
            
        Returns:
            List of detected error patterns
        """
        error_patterns = []
        
        # Character-by-character comparison
        min_len = min(len(original), len(target))
        
        for i in range(min_len):
            orig_char = original[i]
            target_char = target[i]
            
            if orig_char != target_char:
                # Check if this is a known OCR error pattern
                if orig_char in self.common_ocr_errors and self.common_ocr_errors[orig_char] == target_char:
                    error_patterns.append(f"{orig_char} -> {target_char} (common_ocr_error)")
                elif target_char in self.common_ocr_errors and self.common_ocr_errors[target_char] == orig_char:
                    error_patterns.append(f"{orig_char} -> {target_char} (reverse_ocr_error)")
                else:
                    error_patterns.append(f"{orig_char} -> {target_char} (substitution)")
        
        # Length differences
        if len(original) != len(target):
            if len(original) > len(target):
                error_patterns.append("deletion")
            else:
                error_patterns.append("insertion")
        
        return error_patterns
    
    def _calculate_confidence(self, similarity: float, error_patterns: List[str], 
                            original: str, target: str) -> float:
        """
        Calculate confidence score based on similarity and error patterns.
        
        Args:
            similarity: Base similarity score
            error_patterns: List of detected error patterns
            original: Original text
            target: Target text
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Start with similarity as base confidence
        confidence = similarity
        
        # Adjust based on error patterns
        common_error_bonus = 0.0
        for pattern in error_patterns:
            if "common_ocr_error" in pattern:
                common_error_bonus += 0.1  # Boost for known OCR errors
            elif "reverse_ocr_error" in pattern:
                common_error_bonus += 0.05  # Smaller boost for reverse errors
        
        # Apply error pattern bonus (but cap at reasonable level)
        confidence += min(common_error_bonus, 0.2)
        
        # Length penalty for very different lengths
        length_diff = abs(len(original) - len(target))
        max_length = max(len(original), len(target))
        
        if max_length > 0:
            length_ratio = length_diff / max_length
            confidence *= (1.0 - length_ratio * 0.2)  # Up to 20% penalty
        
        # Ensure confidence stays in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def get_similarity_breakdown(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Get detailed similarity breakdown for analysis.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with detailed similarity metrics
        """
        katakana1 = self._extract_katakana(text1)
        katakana2 = self._extract_katakana(text2)
        
        if not katakana1 or not katakana2:
            return {"error": "No katakana found in one or both texts"}
        
        similarities = self._calculate_similarities(katakana1, katakana2)
        error_patterns = self._detect_error_patterns(katakana1, katakana2)
        confidence = self._calculate_confidence(similarities.get('difflib', 0), 
                                              error_patterns, katakana1, katakana2)
        
        return {
            "katakana1": katakana1,
            "katakana2": katakana2,
            "similarities": similarities,
            "error_patterns": error_patterns,
            "confidence": confidence,
            "above_threshold": any(sim >= self.similarity_threshold for sim in similarities.values())
        }
    
    def batch_match(self, katakana_texts: List[str], options: List[str]) -> List[Optional[FuzzyMatch]]:
        """
        Perform batch fuzzy matching for multiple katakana texts.
        
        Args:
            katakana_texts: List of katakana texts to match
            options: List of option strings
            
        Returns:
            List of FuzzyMatch objects (None for no matches)
        """
        results = []
        
        for text in katakana_texts:
            match = self.find_best_katakana_match(text, options)
            results.append(match)
        
        logger.info(f"Batch matching completed: {len([r for r in results if r])} / {len(katakana_texts)} matches found")
        
        return results


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize fuzzy matcher
    matcher = KatakanaFuzzyMatcher(similarity_threshold=0.75)
    
    # Test cases with common OCR errors
    test_cases = [
        # (OCR text, options, expected match)
        ("ソニー", ["ソニー", "トヨタ", "ホンダ", "ニンテンドー"]),  # Perfect match
        ("ンニー", ["ソニー", "トヨタ", "ホンダ", "ニンテンドー"]),  # ソ -> ン error
        ("ツカサ", ["ツカサ", "タカサ", "シカサ", "スカサ"]),      # Confusion between similar chars
        ("コーヒー", ["コーヒー", "ローヒー", "ゴーヒー", "ホーヒー"]),  # ロ/コ confusion
        ("ゲーム", ["ゲーム", "ケーム", "セーム", "デーム"]),       # ゲ/ケ dakuten error
    ]
    
    print("Katakana Fuzzy Matching Test Results:")
    print("=" * 50)
    
    for i, (ocr_text, options) in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{ocr_text}'")
        print(f"Options: {options}")
        
        match = matcher.find_best_katakana_match(ocr_text, options)
        
        if match:
            print(f"✅ Best match: '{match.matched_option}'")
            print(f"   Similarity: {match.similarity_score:.3f}")
            print(f"   Confidence: {match.confidence:.3f}")
            print(f"   Algorithm: {match.algorithm_used}")
            if match.error_patterns:
                print(f"   Error patterns: {match.error_patterns}")
        else:
            print("❌ No match found above threshold")
        
        # Show detailed breakdown
        if len(options) > 0:
            breakdown = matcher.get_similarity_breakdown(ocr_text, options[0])
            print(f"   Detailed breakdown with '{options[0]}':")
            print(f"   - Similarities: {breakdown.get('similarities', {})}")
    
    print("\nKatakana Fuzzy Matcher test completed!")
