"""
OCR module for Ultimate Japanese Quiz Solver
Enhanced preprocessing and multi-PSM testing for optimal Japanese text recognition
"""

from .ocr_preprocess import preprocess_image_for_ocr, enhance_japanese_text
from .ocr_multi_psm import best_ocr_result, test_multiple_psm_modes

__all__ = [
    'preprocess_image_for_ocr',
    'enhance_japanese_text', 
    'best_ocr_result',
    'test_multiple_psm_modes'
]
