"""
Rules Engine Module for Ultimate Japanese Quiz Solver

Deterministic rule-based processing for common Japanese quiz patterns:
- Date/reading mappings (むいか → 六日)  
- Katakana fuzzy matching
- JLPT vocabulary verification
- Pattern recognition and exact matching
"""

from .rules_date import DateReadingRuleEngine
from .fuzzy_kata import KatakanaFuzzyMatcher
from .rules_engine import UnifiedRuleEngine

__all__ = [
    'DateReadingRuleEngine',
    'KatakanaFuzzyMatcher', 
    'UnifiedRuleEngine'
]
