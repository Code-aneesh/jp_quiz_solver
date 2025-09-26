"""
Unified Rule Engine

This module combines all rule-based processing systems into a single interface.
Provides priority-based rule matching and decision making for Japanese quiz
text processing BEFORE calling LLM services.

Key features:
- Unified interface for all rule types
- Priority-based rule application
- Confidence scoring and thresholding
- Rule performance tracking
- Extensible rule system architecture
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import time

from .rules_date import DateReadingRuleEngine, RuleMatch
from .fuzzy_kata import KatakanaFuzzyMatcher, FuzzyMatch

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of rules in the system"""
    DATE_READING = "date_reading"
    KATAKANA_FUZZY = "katakana_fuzzy"
    VOCABULARY = "vocabulary"
    PATTERN = "pattern"
    CUSTOM = "custom"


@dataclass
class UnifiedRuleResult:
    """Unified result from rule processing"""
    rule_type: RuleType
    matched_text: str
    suggested_answer: str
    confidence: float
    rule_name: str
    processing_time: float
    metadata: Dict[str, Any]
    should_override_llm: bool = False


class UnifiedRuleEngine:
    """
    Unified engine that coordinates all rule-based processing
    """
    
    def __init__(self, 
                 date_rule_confidence_threshold: float = 0.8,
                 katakana_fuzzy_threshold: float = 0.75,
                 enable_date_rules: bool = True,
                 enable_katakana_fuzzy: bool = True):
        
        self.date_rule_threshold = date_rule_confidence_threshold
        self.katakana_fuzzy_threshold = katakana_fuzzy_threshold
        self.enable_date_rules = enable_date_rules
        self.enable_katakana_fuzzy = enable_katakana_fuzzy
        
        # Initialize rule engines
        self.date_engine = DateReadingRuleEngine() if enable_date_rules else None
        self.katakana_matcher = KatakanaFuzzyMatcher(
            similarity_threshold=katakana_fuzzy_threshold
        ) if enable_katakana_fuzzy else None
        
        # Performance tracking
        self.rule_stats = {
            "total_queries": 0,
            "rule_hits": 0,
            "rule_type_counts": {},
            "average_processing_time": 0.0
        }
        
        logger.info("Unified Rule Engine initialized")
        self._log_configuration()
    
    def _log_configuration(self):
        """Log current configuration"""
        logger.info(f"Configuration:")
        logger.info(f"  - Date rules: {'enabled' if self.enable_date_rules else 'disabled'}")
        logger.info(f"  - Katakana fuzzy: {'enabled' if self.enable_katakana_fuzzy else 'disabled'}")
        logger.info(f"  - Date rule threshold: {self.date_rule_threshold}")
        logger.info(f"  - Katakana fuzzy threshold: {self.katakana_fuzzy_threshold}")
    
    def process_text(self, text: str, options: List[str]) -> Optional[UnifiedRuleResult]:
        """
        Process text through all enabled rule engines and return the best match.
        
        Args:
            text: OCR detected text
            options: List of multiple choice options
            
        Returns:
            UnifiedRuleResult if a rule match is found, None otherwise
        """
        start_time = time.time()
        self.rule_stats["total_queries"] += 1
        
        logger.info(f"Processing text with unified rules: '{text[:50]}...'")
        logger.debug(f"Options: {options}")
        
        all_results = []
        
        # Apply date/reading rules (highest priority)
        if self.enable_date_rules and self.date_engine:
            date_result = self._apply_date_rules(text, options)
            if date_result:
                all_results.append(date_result)
        
        # Apply katakana fuzzy matching (lower priority)
        if self.enable_katakana_fuzzy and self.katakana_matcher:
            katakana_result = self._apply_katakana_fuzzy(text, options)
            if katakana_result:
                all_results.append(katakana_result)
        
        # Select best result based on priority and confidence
        best_result = self._select_best_result(all_results) if all_results else None
        
        # Update statistics
        processing_time = time.time() - start_time
        self._update_stats(best_result, processing_time)
        
        if best_result:
            logger.info(f"Rule match found: {best_result.rule_type.value} - "
                       f"'{best_result.matched_text}' -> '{best_result.suggested_answer}' "
                       f"(confidence: {best_result.confidence:.3f})")
        else:
            logger.info("No rule matches found")
        
        return best_result
    
    def _apply_date_rules(self, text: str, options: List[str]) -> Optional[UnifiedRuleResult]:
        """Apply date/reading rule engine"""
        try:
            start_time = time.time()
            
            best_match = self.date_engine.find_best_match(text, options)
            
            if best_match and best_match.confidence >= self.date_rule_threshold:
                processing_time = time.time() - start_time
                
                # Determine if this should override LLM
                should_override = (best_match.confidence >= 0.9 and 
                                 best_match.rule_type in ["date", "reading"])
                
                result = UnifiedRuleResult(
                    rule_type=RuleType.DATE_READING,
                    matched_text=best_match.matched_text,
                    suggested_answer=best_match.correct_form,
                    confidence=best_match.confidence,
                    rule_name=f"date_{best_match.rule_type}",
                    processing_time=processing_time,
                    metadata={
                        "pattern_used": best_match.pattern_used,
                        "rule_subtype": best_match.rule_type
                    },
                    should_override_llm=should_override
                )
                
                logger.debug(f"Date rule match: {best_match.pattern_used} "
                           f"(confidence: {best_match.confidence:.3f})")
                return result
        
        except Exception as e:
            logger.warning(f"Date rule application failed: {e}")
        
        return None
    
    def _apply_katakana_fuzzy(self, text: str, options: List[str]) -> Optional[UnifiedRuleResult]:
        """Apply katakana fuzzy matching"""
        try:
            start_time = time.time()
            
            fuzzy_match = self.katakana_matcher.find_best_katakana_match(text, options)
            
            if fuzzy_match and fuzzy_match.confidence >= self.katakana_fuzzy_threshold:
                processing_time = time.time() - start_time
                
                # Katakana fuzzy matches rarely override LLM unless very high confidence
                should_override = fuzzy_match.confidence >= 0.95
                
                result = UnifiedRuleResult(
                    rule_type=RuleType.KATAKANA_FUZZY,
                    matched_text=fuzzy_match.original_text,
                    suggested_answer=fuzzy_match.matched_option,
                    confidence=fuzzy_match.confidence,
                    rule_name=f"katakana_{fuzzy_match.algorithm_used}",
                    processing_time=processing_time,
                    metadata={
                        "similarity_score": fuzzy_match.similarity_score,
                        "algorithm_used": fuzzy_match.algorithm_used,
                        "error_patterns": fuzzy_match.error_patterns
                    },
                    should_override_llm=should_override
                )
                
                logger.debug(f"Katakana fuzzy match: '{fuzzy_match.original_text}' -> "
                           f"'{fuzzy_match.matched_option}' "
                           f"(similarity: {fuzzy_match.similarity_score:.3f})")
                return result
        
        except Exception as e:
            logger.warning(f"Katakana fuzzy matching failed: {e}")
        
        return None
    
    def _select_best_result(self, results: List[UnifiedRuleResult]) -> Optional[UnifiedRuleResult]:
        """
        Select the best result from multiple rule matches.
        
        Args:
            results: List of UnifiedRuleResult objects
            
        Returns:
            Best UnifiedRuleResult based on priority and confidence
        """
        if not results:
            return None
        
        if len(results) == 1:
            return results[0]
        
        # Define rule type priority (lower number = higher priority)
        rule_priority = {
            RuleType.DATE_READING: 1,    # Highest priority
            RuleType.VOCABULARY: 2,
            RuleType.PATTERN: 3,
            RuleType.KATAKANA_FUZZY: 4,  # Lower priority
            RuleType.CUSTOM: 5
        }
        
        # Sort by priority, then by confidence
        def result_key(result: UnifiedRuleResult) -> Tuple[int, float]:
            priority = rule_priority.get(result.rule_type, 999)
            return (priority, -result.confidence)  # Negative for descending confidence
        
        best_result = min(results, key=result_key)
        
        logger.debug(f"Selected best result: {best_result.rule_type.value} "
                    f"(confidence: {best_result.confidence:.3f})")
        
        return best_result
    
    def _update_stats(self, result: Optional[UnifiedRuleResult], processing_time: float):
        """Update performance statistics"""
        # Update processing time average
        total_time = self.rule_stats["average_processing_time"] * self.rule_stats["total_queries"]
        total_time += processing_time
        self.rule_stats["average_processing_time"] = total_time / self.rule_stats["total_queries"]
        
        if result:
            self.rule_stats["rule_hits"] += 1
            
            rule_type_name = result.rule_type.value
            if rule_type_name not in self.rule_stats["rule_type_counts"]:
                self.rule_stats["rule_type_counts"][rule_type_name] = 0
            self.rule_stats["rule_type_counts"][rule_type_name] += 1
    
    def should_skip_llm(self, result: UnifiedRuleResult) -> bool:
        """
        Determine if LLM processing should be skipped based on rule result.
        
        Args:
            result: UnifiedRuleResult from rule processing
            
        Returns:
            True if LLM should be skipped, False otherwise
        """
        # Check explicit override flag
        if result.should_override_llm:
            logger.info(f"Rule {result.rule_name} set explicit LLM override")
            return True
        
        # High confidence date/reading rules should override
        if (result.rule_type == RuleType.DATE_READING and 
            result.confidence >= 0.9):
            logger.info(f"High confidence date/reading rule overriding LLM")
            return True
        
        # Very high confidence fuzzy matches can override
        if (result.rule_type == RuleType.KATAKANA_FUZZY and 
            result.confidence >= 0.95):
            logger.info(f"Very high confidence katakana match overriding LLM")
            return True
        
        return False
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get comprehensive rule engine statistics"""
        hit_rate = (self.rule_stats["rule_hits"] / self.rule_stats["total_queries"] 
                   if self.rule_stats["total_queries"] > 0 else 0.0)
        
        stats = {
            "total_queries": self.rule_stats["total_queries"],
            "rule_hits": self.rule_stats["rule_hits"],
            "hit_rate": hit_rate,
            "average_processing_time": self.rule_stats["average_processing_time"],
            "rule_type_breakdown": self.rule_stats["rule_type_counts"].copy()
        }
        
        # Add individual engine stats
        if self.date_engine:
            stats["date_engine_stats"] = self.date_engine.get_mapping_stats()
        
        return stats
    
    def add_custom_rule(self, 
                       rule_name: str,
                       pattern: str,
                       replacement: str,
                       confidence: float = 0.9,
                       rule_type: RuleType = RuleType.CUSTOM):
        """
        Add a custom rule to the system.
        
        Args:
            rule_name: Name of the rule
            pattern: Pattern to match (can be regex)
            replacement: Replacement text
            confidence: Confidence score for this rule
            rule_type: Type of rule (for prioritization)
        """
        # For now, add to date engine if it's a reading rule
        if rule_type == RuleType.DATE_READING and self.date_engine:
            self.date_engine.add_custom_mapping(pattern, replacement, "custom")
            logger.info(f"Added custom date/reading rule: {rule_name}")
        else:
            logger.warning(f"Custom rule type {rule_type} not yet implemented")
    
    def configure_thresholds(self,
                           date_threshold: Optional[float] = None,
                           katakana_threshold: Optional[float] = None):
        """
        Update confidence thresholds for rule engines.
        
        Args:
            date_threshold: New threshold for date/reading rules
            katakana_threshold: New threshold for katakana fuzzy matching
        """
        if date_threshold is not None:
            self.date_rule_threshold = date_threshold
            logger.info(f"Updated date rule threshold to {date_threshold}")
        
        if katakana_threshold is not None:
            self.katakana_fuzzy_threshold = katakana_threshold
            if self.katakana_matcher:
                self.katakana_matcher.similarity_threshold = katakana_threshold
            logger.info(f"Updated katakana fuzzy threshold to {katakana_threshold}")
    
    def reset_statistics(self):
        """Reset all performance statistics"""
        self.rule_stats = {
            "total_queries": 0,
            "rule_hits": 0,
            "rule_type_counts": {},
            "average_processing_time": 0.0
        }
        logger.info("Rule engine statistics reset")
    
    def test_all_rules(self, test_cases: List[Tuple[str, List[str]]]) -> Dict[str, Any]:
        """
        Test all rules against provided test cases.
        
        Args:
            test_cases: List of (text, options) tuples for testing
            
        Returns:
            Dictionary with test results and statistics
        """
        logger.info(f"Testing {len(test_cases)} cases against all rules")
        
        results = []
        correct_predictions = 0
        
        for i, (text, options) in enumerate(test_cases):
            result = self.process_text(text, options)
            
            test_result = {
                "test_id": i,
                "input_text": text,
                "options": options,
                "rule_result": result,
                "found_match": result is not None
            }
            
            results.append(test_result)
            
            if result:
                logger.info(f"Test {i}: Rule match - {result.rule_type.value}")
            else:
                logger.info(f"Test {i}: No rule match")
        
        # Calculate statistics
        matches_found = sum(1 for r in results if r["found_match"])
        match_rate = matches_found / len(test_cases) if test_cases else 0.0
        
        test_summary = {
            "total_tests": len(test_cases),
            "matches_found": matches_found,
            "match_rate": match_rate,
            "results": results,
            "rule_stats": self.get_rule_statistics()
        }
        
        logger.info(f"Rule testing completed: {matches_found}/{len(test_cases)} matches found "
                   f"({match_rate:.1%} match rate)")
        
        return test_summary


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize unified rule engine
    engine = UnifiedRuleEngine()
    
    # Test cases covering different rule types
    test_cases = [
        # Date/reading rules
        ("むいかにがっこうにいきます", ["六日", "七日", "八日", "九日"]),
        ("はつかはやすみです", ["二十日", "二十一日", "十九日", "十八日"]),
        ("ちちはかいしゃいんです", ["父", "母", "兄", "姉"]),
        
        # Katakana fuzzy matching
        ("ソニーのカメラです", ["ソニー", "キャノン", "ニコン", "オリンパス"]),
        ("ンニーのテレビです", ["ソニー", "パナソニック", "東芝", "シャープ"]),  # ソ->ン error
        ("コーヒーをのみます", ["コーヒー", "ジュース", "ミルク", "ウォーター"]),
        
        # Mixed cases
        ("3がつみっかはひなまつり", ["三月三日", "三月四日", "四月三日", "二月三日"]),
        ("あにはゲームがすきです", ["兄", "姉", "弟", "妹"]),
        
        # No match cases
        ("これはえいごのぶんです", ["This is English", "これは日本語", "混合文章", "その他"])
    ]
    
    print("Unified Rule Engine Test Results:")
    print("=" * 60)
    
    for i, (text, options) in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{text}'")
        print(f"Options: {options}")
        
        result = engine.process_text(text, options)
        
        if result:
            print(f"✅ Rule Match Found:")
            print(f"   Type: {result.rule_type.value}")
            print(f"   Matched: '{result.matched_text}'")
            print(f"   Answer: '{result.suggested_answer}'")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Should override LLM: {result.should_override_llm}")
        else:
            print("❌ No rule match found")
    
    # Print statistics
    stats = engine.get_rule_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Rule hits: {stats['rule_hits']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Average processing time: {stats['average_processing_time']:.3f}s")
    print(f"  Rule type breakdown: {stats['rule_type_breakdown']}")
    
    print("\nUnified Rule Engine test completed!")
