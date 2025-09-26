"""
Date and Reading Rule Engine

This module provides deterministic rule-based matching for Japanese dates,
readings, and common vocabulary patterns. Rules run BEFORE LLM processing
and provide exact matches with high confidence.

Key features:
- Date reading mappings (むいか → 六日, はつか → 二十日)
- Kanji reading verification (はは → 母, ちち → 父)
- Common vocabulary pattern matching
- Number/counter combinations
- Time expression mappings
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RuleMatch:
    """Result from rule matching"""
    matched_text: str
    correct_form: str
    rule_type: str
    confidence: float
    pattern_used: str


class DateReadingRuleEngine:
    """
    Engine for matching Japanese date readings and common patterns
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("./data")
        self.date_mappings = {}
        self.reading_mappings = {}
        self.number_mappings = {}
        self.time_mappings = {}
        self.vocabulary_mappings = {}
        
        self._load_rule_data()
        logger.info("Date/Reading Rule Engine initialized")
    
    def _load_rule_data(self):
        """Load rule mappings from data files"""
        try:
            # Initialize built-in date mappings
            self._init_date_mappings()
            
            # Load additional mappings from files if available
            mapping_file = self.data_dir / "date_mappings.json"
            if mapping_file.exists():
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    external_mappings = json.load(f)
                    self.date_mappings.update(external_mappings.get('dates', {}))
                    self.reading_mappings.update(external_mappings.get('readings', {}))
                    self.vocabulary_mappings.update(external_mappings.get('vocabulary', {}))
                logger.info(f"Loaded external mappings from {mapping_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load external mappings: {e}")
    
    def _init_date_mappings(self):
        """Initialize core Japanese date and reading mappings"""
        
        # Date readings (day of month)
        self.date_mappings.update({
            # Days 1-10
            "ついたち": "一日", "いちにち": "一日",
            "ふつか": "二日", "にかにち": "二日",
            "みっか": "三日", "みかにち": "三日",
            "よっか": "四日", "よかにち": "四日",
            "いつか": "五日", "いかにち": "五日", 
            "むいか": "六日", "むかにち": "六日",
            "なのか": "七日", "なかにち": "七日",
            "ようか": "八日", "やかにち": "八日",
            "ここのか": "九日", "きゅうにち": "九日",
            "とおか": "十日", "じゅうにち": "十日",
            
            # Days 11-20  
            "じゅういちにち": "十一日", "じゅういちんち": "十一日",
            "じゅうににち": "十二日", "じゅうにんち": "十二日", 
            "じゅうさんにち": "十三日", "じゅうさんんち": "十三日",
            "じゅうよっか": "十四日", "じゅうよんにち": "十四日",
            "じゅうごにち": "十五日", "じゅうごんち": "十五日",
            "じゅうろくにち": "十六日", "じゅうろくんち": "十六日",
            "じゅうしちにち": "十七日", "じゅうななにち": "十七日",
            "じゅうはちにち": "十八日", "じゅうはちんち": "十八日",
            "じゅうくにち": "十九日", "じゅうきゅうにち": "十九日",
            "はつか": "二十日", "にじゅうにち": "二十日",
            
            # Days 21-31
            "にじゅういちにち": "二十一日", "にじゅういっか": "二十一日",
            "にじゅうににち": "二十二日", "にじゅうふつか": "二十二日",
            "にじゅうさんにち": "二十三日", "にじゅうみっか": "二十三日",
            "にじゅうよっか": "二十四日", "にじゅうよんにち": "二十四日",
            "にじゅうごにち": "二十五日", "にじゅういつか": "二十五日",
            "にじゅうろくにち": "二十六日", "にじゅうむいか": "二十六日",
            "にじゅうしちにち": "二十七日", "にじゅうなのか": "二十七日",
            "にじゅうはちにち": "二十八日", "にじゅうようか": "二十八日", 
            "にじゅうくにち": "二十九日", "にじゅうここのか": "二十九日",
            "さんじゅうにち": "三十日", "みそか": "三十日",
            "さんじゅういちにち": "三十一日", "おおみそか": "三十一日"
        })
        
        # Common vocabulary readings
        self.reading_mappings.update({
            # Family
            "はは": "母", "ちち": "父", "あに": "兄", "あね": "姉",
            "おとうと": "弟", "いもうと": "妹", "そふ": "祖父", "そぼ": "祖母",
            
            # Body parts
            "あたま": "頭", "かお": "顔", "め": "目", "はな": "鼻", 
            "くち": "口", "みみ": "耳", "て": "手", "あし": "足",
            
            # Colors
            "あか": "赤", "あお": "青", "きいろ": "黄色", "みどり": "緑",
            "しろ": "白", "くろ": "黒", "ちゃいろ": "茶色",
            
            # Time expressions
            "いま": "今", "きょう": "今日", "あした": "明日", "きのう": "昨日",
            "あさ": "朝", "ひる": "昼", "ばん": "晩", "よる": "夜",
            "つき": "月", "ひ": "日", "とし": "年", "じかん": "時間",
            
            # Numbers
            "いち": "一", "に": "二", "さん": "三", "よん": "四", "し": "四",
            "ご": "五", "ろく": "六", "なな": "七", "しち": "七",
            "はち": "八", "きゅう": "九", "く": "九", "じゅう": "十",
            
            # Common adjectives
            "おおきい": "大きい", "ちいさい": "小さい", "たかい": "高い", 
            "やすい": "安い", "あたらしい": "新しい", "ふるい": "古い",
            "いい": "良い", "よい": "良い", "わるい": "悪い",
            
            # Directions
            "ひがし": "東", "にし": "西", "みなみ": "南", "きた": "北",
            "うえ": "上", "した": "下", "みぎ": "右", "ひだり": "左"
        })
        
        # Time expressions
        self.time_mappings.update({
            "いちじ": "一時", "にじ": "二時", "さんじ": "三時", "よじ": "四時",
            "ごじ": "五時", "ろくじ": "六時", "しちじ": "七時", "はちじ": "八時",
            "くじ": "九時", "じゅうじ": "十時", "じゅういちじ": "十一時", "じゅうにじ": "十二時",
            "はん": "半", "じゅっぷん": "十分", "にじゅっぷん": "二十分", 
            "さんじゅっぷん": "三十分", "よんじゅっぷん": "四十分", "ごじゅっぷん": "五十分"
        })
    
    def find_exact_matches(self, text: str, options: List[str]) -> List[RuleMatch]:
        """
        Find exact rule-based matches between OCR text and provided options.
        
        Args:
            text: OCR detected text
            options: List of multiple choice options
            
        Returns:
            List of RuleMatch objects for exact matches found
        """
        matches = []
        
        # Clean and normalize text
        clean_text = self._normalize_text(text)
        
        # Check each mapping type
        all_mappings = [
            (self.date_mappings, "date"),
            (self.reading_mappings, "reading"),
            (self.time_mappings, "time"),
            (self.vocabulary_mappings, "vocabulary")
        ]
        
        for mapping_dict, rule_type in all_mappings:
            for reading, kanji in mapping_dict.items():
                # Check if reading appears in OCR text
                if reading in clean_text:
                    # Check if corresponding kanji is in options
                    for option in options:
                        if kanji in option:
                            match = RuleMatch(
                                matched_text=reading,
                                correct_form=kanji,
                                rule_type=rule_type,
                                confidence=0.95,  # High confidence for exact matches
                                pattern_used=f"{reading} → {kanji}"
                            )
                            matches.append(match)
                            logger.info(f"Exact {rule_type} match: {reading} → {kanji}")
        
        return matches
    
    def find_pattern_matches(self, text: str, options: List[str]) -> List[RuleMatch]:
        """
        Find pattern-based matches using regex patterns.
        
        Args:
            text: OCR detected text  
            options: List of multiple choice options
            
        Returns:
            List of RuleMatch objects for pattern matches
        """
        matches = []
        clean_text = self._normalize_text(text)
        
        # Date patterns
        date_patterns = [
            (r'(\d+)にち', r'\1日', "date_number"),
            (r'(\d+)がつ', r'\1月', "month_number"), 
            (r'(\d+)ねん', r'\1年', "year_number"),
            (r'(\d+)じ', r'\1時', "time_number"),
            (r'(\d+)ふん', r'\1分', "minute_number")
        ]
        
        for pattern, replacement, pattern_type in date_patterns:
            matches_found = re.finditer(pattern, clean_text)
            for match in matches_found:
                converted = re.sub(pattern, replacement, match.group())
                
                # Check if converted form is in options
                for option in options:
                    if converted in option:
                        rule_match = RuleMatch(
                            matched_text=match.group(),
                            correct_form=converted,
                            rule_type="pattern",
                            confidence=0.85,
                            pattern_used=f"{pattern} → {replacement}"
                        )
                        matches.append(rule_match)
                        logger.info(f"Pattern match: {match.group()} → {converted}")
        
        return matches
    
    def find_best_match(self, text: str, options: List[str]) -> Optional[RuleMatch]:
        """
        Find the single best rule match.
        
        Args:
            text: OCR detected text
            options: List of multiple choice options
            
        Returns:
            Best RuleMatch or None if no matches found
        """
        # Get all matches
        exact_matches = self.find_exact_matches(text, options)
        pattern_matches = self.find_pattern_matches(text, options)
        
        all_matches = exact_matches + pattern_matches
        
        if not all_matches:
            return None
        
        # Sort by confidence and rule type priority
        def match_priority(match: RuleMatch) -> Tuple[int, float]:
            type_priority = {
                "date": 1,      # Highest priority
                "reading": 2,
                "time": 3,
                "vocabulary": 4,
                "pattern": 5    # Lowest priority
            }
            return (type_priority.get(match.rule_type, 99), -match.confidence)
        
        best_match = min(all_matches, key=match_priority)
        logger.info(f"Best rule match: {best_match.matched_text} → {best_match.correct_form} "
                   f"(confidence: {best_match.confidence:.2f})")
        
        return best_match
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent matching"""
        # Remove whitespace and convert to lowercase hiragana/katakana
        normalized = re.sub(r'\s+', '', text.lower())
        
        # Convert full-width characters to half-width if needed
        normalized = self._normalize_width(normalized)
        
        return normalized
    
    def _normalize_width(self, text: str) -> str:
        """Convert full-width characters to half-width for consistent matching"""
        # Full-width to half-width digit conversion
        full_width_digits = "０１２３４５６７８９"
        half_width_digits = "0123456789"
        
        translation_table = str.maketrans(full_width_digits, half_width_digits)
        return text.translate(translation_table)
    
    def add_custom_mapping(self, reading: str, kanji: str, rule_type: str = "custom"):
        """
        Add a custom reading-kanji mapping.
        
        Args:
            reading: Reading in hiragana
            kanji: Corresponding kanji
            rule_type: Type of rule (for categorization)
        """
        if rule_type == "date":
            self.date_mappings[reading] = kanji
        elif rule_type == "time":
            self.time_mappings[reading] = kanji
        elif rule_type == "vocabulary":
            self.vocabulary_mappings[reading] = kanji
        else:
            self.reading_mappings[reading] = kanji
        
        logger.info(f"Added custom {rule_type} mapping: {reading} → {kanji}")
    
    def save_mappings(self, filename: Optional[str] = None):
        """Save current mappings to JSON file"""
        if filename is None:
            filename = self.data_dir / "date_mappings.json"
        
        mappings_data = {
            "dates": self.date_mappings,
            "readings": self.reading_mappings,
            "times": self.time_mappings,
            "vocabulary": self.vocabulary_mappings
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(mappings_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved mappings to {filename}")
        except Exception as e:
            logger.error(f"Failed to save mappings: {e}")
    
    def get_mapping_stats(self) -> Dict[str, int]:
        """Get statistics about loaded mappings"""
        return {
            "date_mappings": len(self.date_mappings),
            "reading_mappings": len(self.reading_mappings),
            "time_mappings": len(self.time_mappings),
            "vocabulary_mappings": len(self.vocabulary_mappings),
            "total_mappings": (len(self.date_mappings) + len(self.reading_mappings) +
                             len(self.time_mappings) + len(self.vocabulary_mappings))
        }


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize rule engine
    engine = DateReadingRuleEngine()
    
    # Print mapping statistics
    stats = engine.get_mapping_stats()
    print("Rule Engine Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test exact matching
    test_cases = [
        ("むいかにバスでいきました", ["六日", "七日", "八日", "九日"]),
        ("はつかまでまってください", ["二十日", "二十一日", "十九日", "十八日"]),
        ("あにはがくせいです", ["兄", "姉", "弟", "妹"]),
        ("いまなんじですか", ["今", "昨日", "明日", "明後日"]),
        ("3じにあいましょう", ["三時", "四時", "五時", "六時"])
    ]
    
    print("\nTest Results:")
    for text, options in test_cases:
        best_match = engine.find_best_match(text, options)
        if best_match:
            print(f"✅ '{text}' → {best_match.correct_form} ({best_match.confidence:.2f})")
        else:
            print(f"❌ '{text}' → No match found")
    
    print("\nDate/Reading Rule Engine test completed!")
