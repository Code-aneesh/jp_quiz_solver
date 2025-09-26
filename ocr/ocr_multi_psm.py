"""
Multi-PSM OCR Testing Module

This module tests multiple Page Segmentation Modes (PSMs) with Tesseract
and selects the result with the highest Japanese character ratio.

Key features:
- Tests multiple PSM modes optimized for different text layouts
- Calculates Japanese character ratio to determine quality
- Returns best result with confidence metrics
- Optimized for Japanese quiz content recognition
"""

import pytesseract
from PIL import Image
import re
import logging
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Container for OCR result with metadata"""
    text: str
    confidence: float
    psm_mode: int
    japanese_ratio: float
    processing_time: float
    char_count: int


class PSMTestResult(NamedTuple):
    """Result from PSM testing"""
    best_result: OCRResult
    all_results: List[OCRResult]
    total_time: float


def best_ocr_result(image: Image.Image,
                   language: str = "jpn+eng",
                   test_psms: List[int] = None,
                   min_japanese_ratio: float = 0.1) -> OCRResult:
    """
    Test multiple PSM modes and return the result with highest Japanese character ratio.
    
    Args:
        image: PIL Image to process
        language: Tesseract language parameter
        test_psms: List of PSM modes to test. If None, uses optimal defaults.
        min_japanese_ratio: Minimum Japanese character ratio to consider valid
    
    Returns:
        OCRResult with best Japanese text recognition
    """
    if test_psms is None:
        # Optimized PSM modes for Japanese quiz content
        test_psms = [6, 3, 4, 11, 12, 8, 13]
    
    logger.info(f"Testing {len(test_psms)} PSM modes for optimal Japanese OCR")
    
    results = []
    total_start_time = time.time()
    
    for psm in test_psms:
        try:
            start_time = time.time()
            
            # Configure Tesseract with current PSM
            config = f"--psm {psm} -c tessedit_char_whitelist= " \
                    f"-c preserve_interword_spaces=1"
            
            # Get OCR text
            text = pytesseract.image_to_string(image, lang=language, config=config)
            
            # Get confidence data
            data = pytesseract.image_to_data(image, lang=language, config=config, 
                                           output_type=pytesseract.Output.DICT)
            
            processing_time = time.time() - start_time
            
            # Calculate metrics
            japanese_ratio = calculate_japanese_ratio(text)
            avg_confidence = calculate_average_confidence(data)
            char_count = len(text.strip())
            
            result = OCRResult(
                text=text.strip(),
                confidence=avg_confidence,
                psm_mode=psm,
                japanese_ratio=japanese_ratio,
                processing_time=processing_time,
                char_count=char_count
            )
            
            results.append(result)
            
            logger.debug(f"PSM {psm}: {japanese_ratio:.3f} JP ratio, "
                        f"{avg_confidence:.1f}% conf, {char_count} chars")
            
        except Exception as e:
            logger.warning(f"PSM {psm} failed: {e}")
            continue
    
    total_time = time.time() - total_start_time
    
    if not results:
        logger.error("All PSM modes failed!")
        return OCRResult("", 0.0, 0, 0.0, 0.0, 0)
    
    # Find best result based on Japanese character ratio and confidence
    best = select_best_result(results, min_japanese_ratio)
    
    logger.info(f"Best result: PSM {best.psm_mode} with {best.japanese_ratio:.3f} "
                f"JP ratio and {best.confidence:.1f}% confidence")
    logger.info(f"Total PSM testing time: {total_time:.2f}s")
    
    return best


def test_multiple_psm_modes(image: Image.Image,
                           language: str = "jpn+eng",
                           test_psms: List[int] = None) -> PSMTestResult:
    """
    Comprehensive PSM testing with detailed results.
    
    Args:
        image: PIL Image to process
        language: Tesseract language parameter
        test_psms: List of PSM modes to test
    
    Returns:
        PSMTestResult with best result and all test results
    """
    if test_psms is None:
        test_psms = [6, 3, 4, 11, 12, 8, 13]
    
    total_start_time = time.time()
    results = []
    
    for psm in test_psms:
        try:
            start_time = time.time()
            
            config = f"--psm {psm}"
            text = pytesseract.image_to_string(image, lang=language, config=config)
            data = pytesseract.image_to_data(image, lang=language, config=config,
                                           output_type=pytesseract.Output.DICT)
            
            processing_time = time.time() - start_time
            
            result = OCRResult(
                text=text.strip(),
                confidence=calculate_average_confidence(data),
                psm_mode=psm,
                japanese_ratio=calculate_japanese_ratio(text),
                processing_time=processing_time,
                char_count=len(text.strip())
            )
            
            results.append(result)
            
        except Exception as e:
            logger.warning(f"PSM {psm} failed: {e}")
    
    total_time = time.time() - total_start_time
    
    best = select_best_result(results) if results else None
    
    return PSMTestResult(
        best_result=best,
        all_results=results,
        total_time=total_time
    )


def calculate_japanese_ratio(text: str) -> float:
    """
    Calculate the ratio of Japanese characters in the text.
    
    Args:
        text: Input text string
    
    Returns:
        Ratio of Japanese characters (0.0 to 1.0)
    """
    if not text or not text.strip():
        return 0.0
    
    # Clean text - remove whitespace and newlines for accurate counting
    clean_text = re.sub(r'\s+', '', text)
    
    if not clean_text:
        return 0.0
    
    total_chars = len(clean_text)
    japanese_chars = 0
    
    for char in clean_text:
        if is_japanese_character(char):
            japanese_chars += 1
    
    ratio = japanese_chars / total_chars if total_chars > 0 else 0.0
    return ratio


def is_japanese_character(char: str) -> bool:
    """
    Check if a character is Japanese (hiragana, katakana, or kanji).
    
    Args:
        char: Single character to check
    
    Returns:
        True if character is Japanese
    """
    code = ord(char)
    
    # Hiragana (U+3040-U+309F)
    if 0x3040 <= code <= 0x309F:
        return True
    
    # Katakana (U+30A0-U+30FF)
    if 0x30A0 <= code <= 0x30FF:
        return True
    
    # CJK Unified Ideographs (Kanji) - Main block (U+4E00-U+9FFF)
    if 0x4E00 <= code <= 0x9FFF:
        return True
    
    # CJK Unified Ideographs Extension A (U+3400-U+4DBF)
    if 0x3400 <= code <= 0x4DBF:
        return True
    
    # Additional Japanese punctuation and symbols
    # Japanese punctuation (U+3000-U+303F)
    if 0x3000 <= code <= 0x303F:
        return True
    
    # Halfwidth and Fullwidth Forms - Katakana (U+FF65-U+FF9F)
    if 0xFF65 <= code <= 0xFF9F:
        return True
    
    return False


def calculate_average_confidence(tesseract_data: Dict) -> float:
    """
    Calculate average confidence from Tesseract data output.
    
    Args:
        tesseract_data: Dictionary from pytesseract.image_to_data()
    
    Returns:
        Average confidence percentage
    """
    confidences = tesseract_data.get('conf', [])
    
    if not confidences:
        return 0.0
    
    # Filter out -1 values (invalid/no text detected)
    valid_confidences = [c for c in confidences if c != -1]
    
    if not valid_confidences:
        return 0.0
    
    return sum(valid_confidences) / len(valid_confidences)


def select_best_result(results: List[OCRResult], 
                      min_japanese_ratio: float = 0.1) -> OCRResult:
    """
    Select the best OCR result based on Japanese content and confidence.
    
    Args:
        results: List of OCRResult objects
        min_japanese_ratio: Minimum required Japanese character ratio
    
    Returns:
        Best OCRResult
    """
    if not results:
        raise ValueError("No results provided")
    
    # Filter results with sufficient Japanese content
    japanese_results = [r for r in results if r.japanese_ratio >= min_japanese_ratio]
    
    if not japanese_results:
        # If no results meet Japanese ratio threshold, use all results
        logger.warning(f"No results with Japanese ratio >= {min_japanese_ratio}, using all results")
        japanese_results = results
    
    # Filter out empty or very short results
    substantial_results = [r for r in japanese_results if r.char_count >= 3]
    
    if not substantial_results:
        substantial_results = japanese_results
    
    # Score results based on multiple factors
    def score_result(result: OCRResult) -> float:
        """Calculate composite score for result quality"""
        jp_score = result.japanese_ratio * 100  # Japanese ratio (0-100)
        conf_score = result.confidence  # Confidence (0-100)
        length_bonus = min(result.char_count / 50.0, 1.0) * 10  # Length bonus (0-10)
        
        # Weighted combination
        total_score = (jp_score * 0.5) + (conf_score * 0.3) + (length_bonus * 0.2)
        return total_score
    
    # Find result with highest score
    best_result = max(substantial_results, key=score_result)
    
    logger.debug(f"Selected best result with score: {score_result(best_result):.2f}")
    
    return best_result


def get_optimal_psm_modes_by_content_type(content_type: str = "quiz") -> List[int]:
    """
    Get optimal PSM modes based on expected content type.
    
    Args:
        content_type: Type of content ("quiz", "document", "menu", "mixed")
    
    Returns:
        List of PSM modes to test, ordered by expected effectiveness
    """
    psm_configs = {
        "quiz": [6, 4, 11, 3, 12],  # Single uniform block, single text line
        "document": [3, 6, 4, 11],   # Fully automatic page segmentation
        "menu": [11, 12, 6, 13],     # Sparse text, single text line
        "mixed": [6, 3, 4, 11, 12, 8, 13],  # Try all common modes
    }
    
    return psm_configs.get(content_type, psm_configs["mixed"])


def analyze_ocr_quality(result: OCRResult) -> Dict[str, any]:
    """
    Analyze OCR quality and provide detailed metrics.
    
    Args:
        result: OCRResult to analyze
    
    Returns:
        Dictionary with quality analysis
    """
    analysis = {
        "overall_quality": "unknown",
        "japanese_content": result.japanese_ratio > 0.3,
        "sufficient_confidence": result.confidence > 70,
        "reasonable_length": result.char_count > 5,
        "processing_speed": "fast" if result.processing_time < 1.0 else "slow",
        "recommendations": []
    }
    
    # Determine overall quality
    if (analysis["japanese_content"] and 
        analysis["sufficient_confidence"] and 
        analysis["reasonable_length"]):
        analysis["overall_quality"] = "excellent"
    elif (result.japanese_ratio > 0.1 and result.confidence > 50):
        analysis["overall_quality"] = "good"
    elif result.char_count > 0:
        analysis["overall_quality"] = "poor"
    else:
        analysis["overall_quality"] = "failed"
    
    # Generate recommendations
    if not analysis["japanese_content"]:
        analysis["recommendations"].append("Try image preprocessing to enhance Japanese characters")
    
    if not analysis["sufficient_confidence"]:
        analysis["recommendations"].append("Consider image upscaling or noise reduction")
    
    if not analysis["reasonable_length"]:
        analysis["recommendations"].append("Check if correct region is selected")
    
    return analysis


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    print("Multi-PSM OCR module loaded successfully")
    print("Available functions:")
    print("- best_ocr_result()")
    print("- test_multiple_psm_modes()")
    print("- calculate_japanese_ratio()")
    print("- get_optimal_psm_modes_by_content_type()")
    print("- analyze_ocr_quality()")
    
    # Test Japanese ratio calculation
    test_texts = [
        "これは日本語のテストです",  # Pure Japanese
        "This is English text",      # Pure English
        "これはmixed textです",      # Mixed
        "",                         # Empty
        "123456"                    # Numbers
    ]
    
    print("\nJapanese ratio test results:")
    for text in test_texts:
        ratio = calculate_japanese_ratio(text)
        print(f"'{text}': {ratio:.3f}")
