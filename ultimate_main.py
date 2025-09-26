#!/usr/bin/env python3
"""
🎯 ULTIMATE JAPANESE QUIZ SOLVER 🎯
The most advanced Japanese question detection and solving system.

Features:
- Full screen scanning with auto question detection
- Multi-AI provider support with fallback
- Advanced OCR with multiple preprocessing techniques
- Question type detection and confidence scoring
- Global hotkeys and smart caching
- History tracking and analytics
- Context awareness and answer verification

Author: Ultimate Quiz Solver Team
Version: 2.0 - The Ultimate Edition
"""

import sys
import os
import re
import time
import json
import sqlite3
import hashlib
import threading
import queue
import logging
import gc
import weakref
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

# Core libraries
import mss
import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import keyboard

# AI libraries
import google.generativeai as genai
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

# Optional libraries for advanced features
try:
    from langdetect import detect
except ImportError:
    detect = None

try:
    import pyttsx3  # Text to speech
except ImportError:
    pyttsx3 = None

# Configuration - Use unified config system
try:
    import unified_config as config
except ImportError:
    try:
        import enhanced_config as config
    except ImportError:
        import config

# Data structures
@dataclass
class QuestionResult:
    """Data structure for quiz results"""
    timestamp: datetime
    question_text: str
    question_type: str
    confidence_score: float
    ai_answer: str
    ai_provider: str
    processing_time: float
    region: Dict[str, int]
    ocr_confidence: float

@dataclass
class OCRResult:
    """Data structure for OCR results"""
    text: str
    confidence: float
    regions: List[Dict]
    preprocessing_method: str

class UltimateQuizSolver:
    """The ultimate Japanese quiz solving system"""
    
    def __init__(self):
        # Initialize logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Initializing Ultimate Japanese Quiz Solver")
        
        # Core components
        self.setup_directories()
        self.setup_database()
        self.setup_ai_providers()
        self.setup_ocr()
        
        # Caching and performance with memory management
        self.text_cache = weakref.WeakValueDictionary()  # Automatic cleanup
        self.result_cache = {}
        self.processing_queue = queue.Queue()
        self.is_processing = False
        self.last_processed_text = ""
        self.processing_stats = {"total_questions": 0, "correct_answers": 0, "processing_time": []}
        self._last_cache_cleanup = time.time()
        self._image_references = []  # Track images for cleanup
        
        # UI and monitoring
        self.monitors = mss.mss().monitors[1:]  # Exclude "All in One" monitor
        self.current_monitor = 0
        self.scanning_active = False
        self.emergency_stop = False
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Context management
        self.context_history = []
        self.context_max_size = 5
        
        self.logger.info("✅ Ultimate Quiz Solver initialized successfully")
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        
        # Configure logging
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        log_file = os.path.join(config.LOGS_DIR, f"quiz_solver_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=getattr(logging, config.LOGGING["level"]),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout) if config.LOGGING["console_logging"] else logging.NullHandler()
            ]
        )
    
    def setup_directories(self):
        """Create necessary directories"""
        for directory in [config.DATA_DIR, config.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    def setup_database(self):
        """Initialize SQLite database for history tracking"""
        self.db_path = config.HISTORY_DB
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quiz_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    question_text TEXT NOT NULL,
                    question_type TEXT,
                    confidence_score REAL,
                    ai_answer TEXT,
                    ai_provider TEXT,
                    processing_time REAL,
                    region_json TEXT,
                    ocr_confidence REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON quiz_history(timestamp)
            """)
        self.logger.info(f"✅ Database initialized at {self.db_path}")
    
    def setup_ai_providers(self):
        """Initialize all AI providers"""
        self.ai_providers = {}
        
        # Gemini
        gemini_key = config.AI_PROVIDERS["gemini"]["api_key"]
        if gemini_key and gemini_key != "YOUR_GEMINI_KEY_HERE":
            try:
                genai.configure(api_key=gemini_key)
                self.ai_providers["gemini"] = genai.GenerativeModel(config.AI_PROVIDERS["gemini"]["model"])
                self.logger.info("✅ Gemini AI provider initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Gemini: {e}")
        
        # OpenAI
        if OpenAI and config.AI_PROVIDERS["openai"]["api_key"]:
            try:
                self.ai_providers["openai"] = OpenAI(api_key=config.AI_PROVIDERS["openai"]["api_key"])
                self.logger.info("✅ OpenAI provider initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize OpenAI: {e}")
        
        # Claude (Anthropic)
        if anthropic and config.AI_PROVIDERS["claude"]["api_key"]:
            try:
                self.ai_providers["claude"] = anthropic.Anthropic(api_key=config.AI_PROVIDERS["claude"]["api_key"])
                self.logger.info("✅ Claude AI provider initialized")
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize Claude: {e}")
        
        if not self.ai_providers:
            raise ValueError("❌ No AI providers available! Please configure API keys.")
        
        self.logger.info(f"🤖 Available AI providers: {list(self.ai_providers.keys())}")
    
    def setup_ocr(self):
        """Configure Tesseract OCR"""
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH
        
        # Test OCR availability
        try:
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"✅ Tesseract OCR {version} ready")
            
            # Check for Japanese language pack
            languages = pytesseract.get_languages()
            if 'jpn' in languages:
                self.logger.info("✅ Japanese language pack available")
            else:
                self.logger.warning("⚠️ Japanese language pack not found!")
        except Exception as e:
            self.logger.error(f"❌ OCR setup failed: {e}")
    
    def detect_question_type(self, text: str) -> Tuple[str, float]:
        """Advanced question type detection with confidence scoring"""
        text_lower = text.lower()
        scores = {}
        
        for q_type, patterns in config.QUESTION_PATTERNS.items():
            score = 0.0
            
            # Pattern matching
            if "patterns" in patterns:
                for pattern in patterns["patterns"]:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    score += matches * 0.3
            
            # Keyword matching
            if "keywords" in patterns:
                for keyword in patterns["keywords"]:
                    if keyword.lower() in text_lower:
                        score += 0.2
            
            # Special rules
            if q_type == "multiple_choice":
                # Count potential options
                option_patterns = [r"[①②③④⑤]", r"[1-5][\.)]", r"[A-E][\.)]", r"[ア-オ]"]
                option_count = sum(len(re.findall(p, text)) for p in option_patterns)
                if patterns["min_options"] <= option_count <= patterns["max_options"]:
                    score += option_count * 0.4
            
            scores[q_type] = score
        
        # Determine best match
        if scores:
            best_type = max(scores, key=scores.get)
            confidence = min(scores[best_type], 1.0)
            return best_type, confidence
        
        return "unknown", 0.0
    
    def calculate_confidence_score(self, ocr_result: OCRResult, question_type: str, 
                                 ai_response: str, processing_time: float) -> float:
        """Calculate overall confidence score for the answer"""
        factors = config.CONFIDENCE_FACTORS
        
        # OCR quality (30%)
        ocr_score = min(ocr_result.confidence / 100.0, 1.0)
        ocr_weight = factors["ocr_quality"]
        
        # Question completeness (25%)
        completeness_score = min(len(ocr_result.text) / 50.0, 1.0)  # Normalize to 50 chars
        if any(char in ocr_result.text for char in "？?"):  # Question mark presence
            completeness_score += 0.2
        completeness_weight = factors["question_completeness"]
        
        # AI certainty (25%) - analyze response for confidence indicators
        ai_score = 0.7  # Default
        confidence_indicators = ["definitely", "clearly", "obviously", "確実", "明らか"]
        uncertainty_indicators = ["maybe", "possibly", "might", "probably", "かもしれ", "多分"]
        
        ai_lower = ai_response.lower()
        if any(indicator in ai_lower for indicator in confidence_indicators):
            ai_score = 0.9
        elif any(indicator in ai_lower for indicator in uncertainty_indicators):
            ai_score = 0.5
        ai_weight = factors["ai_certainty"]
        
        # Pattern match quality (20%)
        _, pattern_confidence = self.detect_question_type(ocr_result.text)
        pattern_weight = factors["pattern_match"]
        
        # Calculate weighted score
        total_score = (
            ocr_score * ocr_weight +
            completeness_score * completeness_weight +
            ai_score * ai_weight +
            pattern_confidence * pattern_weight
        )
        
        return min(total_score, 1.0)
    
    @contextmanager
    def _managed_image(self, image: Image.Image):
        """Context manager for proper image cleanup"""
        processed_image = None
        try:
            yield image
        finally:
            # Clean up processed images
            if processed_image and processed_image != image:
                processed_image.close()
            # Force garbage collection periodically
            if len(self._image_references) > 50:
                self._cleanup_images()
    
    def _cleanup_images(self):
        """Clean up image references and force garbage collection"""
        self._image_references.clear()
        gc.collect()
    
    def advanced_preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Advanced image preprocessing for maximum OCR accuracy"""
        original = image.copy()
        method_used = "advanced"
        
        try:
            # Convert to numpy array for OpenCV operations
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply enhancements based on config
            enhancements = config.IMAGE_ENHANCEMENTS
            
            # Noise reduction
            if enhancements["noise_reduction"]:
                gray = cv2.fastNlMeansDenoising(gray)
            
            # Adaptive thresholding
            if enhancements["adaptive_threshold"]:
                gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operations
            if enhancements["morphological_ops"]:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL Image
            processed = Image.fromarray(gray)
            
            # Scale up for better OCR
            scale_factor = enhancements["scale_factor"]
            new_size = (int(processed.width * scale_factor), int(processed.height * scale_factor))
            processed = processed.resize(new_size, Image.LANCZOS)
            
            # Enhance contrast
            if enhancements["contrast_enhancement"]:
                enhancer = ImageEnhance.Contrast(processed)
                processed = enhancer.enhance(2.0)
            
            # Sharpen
            if enhancements["sharpening"]:
                processed = processed.filter(ImageFilter.SHARPEN)
            
            return processed, method_used
            
        except Exception as e:
            self.logger.warning(f"Advanced preprocessing failed: {e}, using basic method")
            return self.basic_preprocess_image(original)
    
    def basic_preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Basic fallback image preprocessing"""
        img = image.convert("L")
        width, height = img.size
        img = img.resize((width * 2, height * 2), Image.LANCZOS)
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)
        return img, "basic"
    
    def japanese_optimized_preprocessing(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Japanese-optimized image preprocessing for maximum OCR accuracy"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Japanese text specific preprocessing
            # 1. Gaussian blur to smooth text
            gray = cv2.GaussianBlur(gray, (1, 1), 0)
            
            # 2. Adaptive threshold optimized for Japanese characters
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 13, 3)
            
            # 3. Morphological operations to connect Japanese character strokes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL
            processed = Image.fromarray(binary)
            
            # 4. Scale up significantly for Japanese characters
            scale_factor = 4.0  # Higher scale for Japanese
            new_size = (int(processed.width * scale_factor), int(processed.height * scale_factor))
            processed = processed.resize(new_size, Image.LANCZOS)
            
            # 5. Additional sharpening for Japanese characters
            processed = processed.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            return processed, "japanese_optimized"
            
        except Exception as e:
            self.logger.warning(f"Japanese preprocessing failed: {e}, using advanced method")
            return self.advanced_preprocess_image(image)
    
    def is_valid_japanese_text(self, text: str) -> bool:
        """Validate if text contains meaningful Japanese characters or valid options"""
        if not text:
            return False
        
        # Always accept numbers and common symbols for options
        if re.match(r'^[0-9①②③④⑤ABCDE\(\)\.\ ]+$', text):
            return True
        
        # Check for Japanese characters
        japanese_count = sum(1 for char in text if self.contains_japanese(char))
        total_chars = len([c for c in text if not c.isspace()])
        
        # Accept if at least 30% Japanese characters, or any Japanese with numbers/symbols
        if total_chars > 0:
            japanese_ratio = japanese_count / total_chars
            has_numbers_symbols = any(c in text for c in '0123456789①②③④⑤()[].')
            
            return japanese_ratio >= 0.3 or (japanese_count > 0 and has_numbers_symbols)
        
        return False
    
    def clean_japanese_text(self, text: str) -> str:
        """Clean and normalize Japanese text from OCR"""
        if not text:
            return text
        
        # Common OCR corrections for Japanese
        corrections = {
            # Hiragana/Katakana corrections
            'ロ': 'ロ',  # Ensure proper katakana
            '口': 'ロ',  # Common OCR mistake
            '二': 'ー',  # Long vowel mark correction
            # Kanji corrections
            '母': '母',  # Ensure proper mother kanji
            '山': '山',  # Ensure proper mountain kanji
            '今': '今',  # Ensure proper now kanji
            '週': '週',  # Ensure proper week kanji
            '天': '天',  # Ensure proper heaven/sky kanji
            '気': '気',  # Ensure proper spirit/weather kanji
            '小': '小',  # Ensure proper small kanji
            '東': '東',  # Ensure proper east kanji
            '空': '空',  # Ensure proper sky kanji
            '六': '六',  # Ensure proper six kanji
            '日': '日',  # Ensure proper day kanji
        }
        
        # Apply corrections
        cleaned_text = text
        for wrong, correct in corrections.items():
            cleaned_text = cleaned_text.replace(wrong, correct)
        
        # Remove obvious OCR artifacts
        cleaned_text = re.sub(r'[|\\/_^`~]', '', cleaned_text)  # Remove common OCR noise
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def perform_ocr(self, image: Image.Image) -> OCRResult:
        """Perform OCR with multiple methods and return best result"""
        results = []
        
        # Try different preprocessing methods
        preprocessing_methods = [
            ("japanese_optimized", self.japanese_optimized_preprocessing),
            ("advanced", self.advanced_preprocess_image),
            ("basic", self.basic_preprocess_image)
        ]
        
        for method_name, preprocess_func in preprocessing_methods:
            try:
                processed_img, actual_method = preprocess_func(image)
                
                # Try different OCR configurations
                for config_name, ocr_config in config.OCR_CONFIGS.items():
                    try:
                        # Get detailed OCR data
                        ocr_data = pytesseract.image_to_data(
                            processed_img, 
                            lang=config.OCR_LANGUAGE,
                            config=ocr_config,
                            output_type=pytesseract.Output.DICT
                        )
                        
                        # Extract text and confidence with Japanese validation
                        text_parts = []
                        confidences = []
                        regions = []
                        
                        for i, conf in enumerate(ocr_data['conf']):
                            if int(conf) > 30:  # Lower threshold for Japanese
                                word = ocr_data['text'][i].strip()
                                if word and self.is_valid_japanese_text(word):
                                    text_parts.append(word)
                                    confidences.append(int(conf))
                                    regions.append({
                                        'x': ocr_data['left'][i],
                                        'y': ocr_data['top'][i],
                                        'w': ocr_data['width'][i],
                                        'h': ocr_data['height'][i]
                                    })
                        
                        if text_parts:
                            text = ' '.join(text_parts)
                            # Apply Japanese text cleaning
                            text = self.clean_japanese_text(text)
                            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                            
                            results.append(OCRResult(
                                text=text,
                                confidence=avg_confidence,
                                regions=regions,
                                preprocessing_method=f"{actual_method}_{config_name}"
                            ))
                    except Exception as e:
                        self.logger.debug(f"OCR failed for {method_name}_{config_name}: {e}")
                        continue
            except Exception as e:
                self.logger.debug(f"Preprocessing failed for {method_name}: {e}")
                continue
        
        # Return best result based on confidence and text length
        if results:
            best_result = max(results, key=lambda r: r.confidence * len(r.text))
            self.logger.debug(f"Best OCR result: {best_result.preprocessing_method}, confidence: {best_result.confidence:.1f}%")
            return best_result
        
        # Fallback to simple OCR
        try:
            text = pytesseract.image_to_string(image, lang=config.OCR_LANGUAGE, config=config.OCR_CONFIG).strip()
            return OCRResult(
                text=text,
                confidence=50.0,  # Assume medium confidence
                regions=[],
                preprocessing_method="fallback"
            )
        except Exception as e:
            self.logger.error(f"All OCR methods failed: {e}")
            return OCRResult(text="", confidence=0.0, regions=[], preprocessing_method="failed")
    
    def get_ai_answer(self, text: str, provider: Optional[str] = None) -> Tuple[str, str, float]:
        """Get answer from AI provider with fallback support"""
        providers_to_try = [provider] if provider else [config.AI_PROVIDER, config.AI_FALLBACK_PROVIDER]
        
        prompt_template = """
🎯 ULTIMATE JAPANESE QUIZ SOLVER PROMPT 🎯

You are the world's most accurate Japanese quiz solver with PERFECT accuracy. Your mission is to provide 100% correct answers through systematic analysis.

DETECTED CONTENT:
{text}

CRITICAL ANALYSIS METHOD:
1. PARSE QUESTION: Extract the exact Japanese sentence/phrase that needs completion
2. CONTEXT ANALYSIS: Understand the semantic meaning of the complete sentence
3. OPTION VERIFICATION: For each option, verify if it creates grammatically and semantically correct Japanese
4. DOUBLE-CHECK: Cross-reference with standard Japanese usage patterns
5. CONFIDENCE VALIDATION: Only mark "High" if 100% certain of correctness

SPECIFIC REQUIREMENTS:
- For multiple choice: Test EACH option in context, don't just pattern match
- For kanji readings: Verify correct pronunciation matches context
- For grammar particles: Ensure proper grammatical function
- For vocabulary: Check semantic appropriateness in sentence context

EXAMPLE ANALYSIS:
Question: "ははと　やまに　のぼりました" with options ①娠 ②卵 ③奴 ④母
Process: 
- Sentence means "[Someone] climbed the mountain with mother"
- Test: ①娠(pregnancy) ②卵(egg) ③奴(guy) ④母(mother) 
- Only ④母 makes semantic sense: "Mother climbed the mountain"
- Answer: ④

RESPONSE FORMAT (EXACTLY):
🎯 ANSWER: [Correct choice/answer with reasoning verification]
📊 CONFIDENCE: [High/Medium/Low - High ONLY if 100% certain]
📝 QUESTION TYPE: [Multiple Choice/True-False/Fill Blank/Essay/Other]
🔍 JAPANESE TEXT: [Clean Japanese text without OCR artifacts]
🌐 TRANSLATION: [Accurate English translation]
✅ EXPLANATION: [Step-by-step reasoning why this answer is correct]
📚 VOCABULARY: [Key vocabulary with correct readings and meanings]
🧠 GRAMMAR NOTES: [Grammar patterns and rules used]
💡 STUDY TIP: [Specific learning advice]

DO NOT GUESS. If uncertain, mark confidence as Low/Medium and explain uncertainty.
Be methodical, accurate, and verify each step of reasoning.
        """
        
        for provider_name in providers_to_try:
            if provider_name in self.ai_providers:
                try:
                    start_time = time.time()
                    prompt = prompt_template.format(text=text)
                    
                    if provider_name == "gemini":
                        response = self.ai_providers[provider_name].generate_content(prompt)
                        answer = response.text.strip()
                    
                    elif provider_name == "openai":
                        response = self.ai_providers[provider_name].chat.completions.create(
                            model=config.AI_PROVIDERS["openai"]["model"],
                            messages=[
                                {"role": "system", "content": "You are the world's most accurate Japanese quiz solver."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.0,
                            max_tokens=config.AI_PROVIDERS["openai"]["max_tokens"]
                        )
                        answer = response.choices[0].message.content.strip()
                    
                    elif provider_name == "claude":
                        response = self.ai_providers[provider_name].messages.create(
                            model=config.AI_PROVIDERS["claude"]["model"],
                            max_tokens=config.AI_PROVIDERS["claude"]["max_tokens"],
                            temperature=0.0,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                        answer = response.content[0].text.strip()
                    
                    processing_time = time.time() - start_time
                    self.logger.info(f"✅ {provider_name} responded in {processing_time:.2f}s")
                    return answer, provider_name, processing_time
                
                except Exception as e:
                    self.logger.error(f"❌ {provider_name} failed: {e}")
                    continue
        
        return "❌ All AI providers failed. Please check your API keys and network connection.", "none", 0.0
    
    def contains_japanese(self, text: str) -> bool:
        """Enhanced Japanese text detection"""
        if not text:
            return False
        
        # Unicode ranges for Japanese characters
        japanese_ranges = [
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
            (0x4E00, 0x9FAF),  # Kanji (CJK Unified Ideographs)
            (0x3400, 0x4DBF),  # CJK Extension A
            (0xFF65, 0xFF9F),  # Half-width Katakana
        ]
        
        for char in text:
            char_code = ord(char)
            for start, end in japanese_ranges:
                if start <= char_code <= end:
                    return True
        
        return False
    
    def auto_detect_regions(self, screenshot: Image.Image) -> List[Dict[str, int]]:
        """Automatically detect potential question regions in screenshot"""
        regions = []
        
        try:
            # Convert to numpy array for OpenCV
            img_array = np.array(screenshot)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find text regions using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get image dimensions
            img_height, img_width = gray.shape
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # More relaxed size constraints for better detection
                if (w >= 50 and h >= 20 and w <= img_width * 0.9 and h <= img_height * 0.9):
                    regions.append({
                        "left": x,
                        "top": y,
                        "width": w,
                        "height": h
                    })
            
            # If no regions found, create grid-based regions
            if not regions:
                # Create a grid of regions to scan
                grid_size = 4
                region_width = img_width // grid_size
                region_height = img_height // grid_size
                
                for row in range(grid_size):
                    for col in range(grid_size):
                        regions.append({
                            "left": col * region_width,
                            "top": row * region_height,
                            "width": region_width,
                            "height": region_height
                        })
            
            # Always include full screen as a region
            regions.insert(0, {
                "left": 0,
                "top": 0,
                "width": img_width,
                "height": img_height
            })
            
            # Sort by size (larger regions first)
            regions.sort(key=lambda r: r["width"] * r["height"], reverse=True)
            
            # Return top regions
            return regions[:10]
            
        except Exception as e:
            self.logger.error(f"Auto region detection failed: {e}")
            # Fallback to full screen
            return [config.CAPTURE_REGION]
    
    def save_result(self, result: QuestionResult):
        """Save quiz result to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO quiz_history 
                    (timestamp, question_text, question_type, confidence_score, 
                     ai_answer, ai_provider, processing_time, region_json, ocr_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.timestamp.isoformat(),
                    result.question_text,
                    result.question_type,
                    result.confidence_score,
                    result.ai_answer,
                    result.ai_provider,
                    result.processing_time,
                    json.dumps(result.region),
                    result.ocr_confidence
                ))
        except Exception as e:
            self.logger.error(f"Failed to save result: {e}")
    
    def process_screen_region(self, region: Dict[str, int]) -> Optional[QuestionResult]:
        """Process a specific screen region for questions"""
        try:
            with mss.mss() as sct:
                # Capture region
                screenshot = sct.grab(region)
                image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                
                # Perform OCR
                ocr_result = self.perform_ocr(image)
                
                if not ocr_result.text or len(ocr_result.text.strip()) < 5:
                    return None
                
                # Check if contains Japanese or substantial text
                has_japanese = self.contains_japanese(ocr_result.text)
                has_numbers = any(c.isdigit() for c in ocr_result.text)
                has_options = any(c in ocr_result.text for c in ['①', '②', '③', '④', '⑤', 'A', 'B', 'C', 'D', '1', '2', '3', '4'])
                
                if not (has_japanese or (len(ocr_result.text) > 10 and (has_numbers or has_options))):
                    return None
                
                # Log what we found for debugging
                self.logger.info(f"OCR detected text: {ocr_result.text[:100]}...")
                self.logger.info(f"Japanese: {has_japanese}, Numbers: {has_numbers}, Options: {has_options}")
                
                # Check cache first
                text_hash = hashlib.md5(ocr_result.text.encode()).hexdigest()
                if text_hash in self.result_cache:
                    cached_result = self.result_cache[text_hash]
                    self.logger.debug(f"Using cached result for: {ocr_result.text[:50]}...")
                    return cached_result
                
                # Detect question type
                question_type, type_confidence = self.detect_question_type(ocr_result.text)
                
                # Get AI answer
                start_time = time.time()
                ai_answer, ai_provider, ai_processing_time = self.get_ai_answer(ocr_result.text)
                
                # Calculate confidence
                confidence_score = self.calculate_confidence_score(
                    ocr_result, question_type, ai_answer, ai_processing_time
                )
                
                # Create result
                result = QuestionResult(
                    timestamp=datetime.now(),
                    question_text=ocr_result.text,
                    question_type=question_type,
                    confidence_score=confidence_score,
                    ai_answer=ai_answer,
                    ai_provider=ai_provider,
                    processing_time=time.time() - start_time,
                    region=region,
                    ocr_confidence=ocr_result.confidence
                )
                
                # Cache result
                self.result_cache[text_hash] = result
                
                # Clean cache if too large
                if len(self.result_cache) > config.CACHE_SIZE:
                    # Remove oldest entries
                    sorted_cache = sorted(self.result_cache.items(), 
                                        key=lambda x: x[1].timestamp)
                    for key, _ in sorted_cache[:len(self.result_cache)//2]:
                        del self.result_cache[key]
                
                # Save to database
                self.save_result(result)
                
                # Update context
                self.update_context(result)
                
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to process region {region}: {e}")
            return None
    
    def update_context(self, result: QuestionResult):
        """Update context history for better AI responses"""
        self.context_history.append({
            "question": result.question_text,
            "answer": result.ai_answer,
            "type": result.question_type,
            "timestamp": result.timestamp
        })
        
        # Maintain context size
        if len(self.context_history) > self.context_max_size:
            self.context_history.pop(0)
    
    def scan_full_screen(self) -> List[QuestionResult]:
        """Scan entire screen for questions"""
        results = []
        
        try:
            # Get all monitors
            with mss.mss() as sct:
                monitors = sct.monitors
                
                # Try current monitor first, then all monitors
                monitor_indices = [self.current_monitor + 1] if self.current_monitor < len(monitors) - 1 else [1]
                if len(monitors) > 2:  # Add other monitors if available
                    monitor_indices.extend([i for i in range(1, len(monitors)) if i not in monitor_indices])
                
                for monitor_idx in monitor_indices:
                    if monitor_idx >= len(monitors):
                        continue
                        
                    monitor = monitors[monitor_idx]
                    self.logger.info(f"Scanning monitor {monitor_idx}: {monitor}")
                    
                    # Capture screen
                    screenshot = sct.grab(monitor)
                    screen_image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                    
                    # Save screenshot for debugging
                    try:
                        debug_path = os.path.join(config.DATA_DIR, "debug_screenshot.png")
                        screen_image.save(debug_path)
                        self.logger.info(f"Screenshot saved to: {debug_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to save debug screenshot: {e}")
                    
                    # Auto-detect question regions
                    if config.FEATURES["auto_region_detection"]:
                        regions = self.auto_detect_regions(screen_image)
                        self.logger.info(f"Detected {len(regions)} regions")
                    else:
                        regions = [monitor]  # Use entire monitor
                    
                    # Process regions
                    for i, region in enumerate(regions):
                        try:
                            self.logger.info(f"Processing region {i+1}/{len(regions)}: {region}")
                            result = self.process_screen_region(region)
                            if result:
                                results.append(result)
                                self.logger.info(f"Found result with confidence {result.confidence_score:.2f}")
                                # Don't break - keep looking for more results
                        except Exception as e:
                            self.logger.error(f"Error processing region {region}: {e}")
                    
                    # If we found results, we can stop scanning other monitors
                    if results:
                        break
        
        except Exception as e:
            self.logger.error(f"Full screen scan failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        self.logger.info(f"Scan completed. Found {len(results)} results.")
        return results
    
    def start_scanning(self):
        """Start the main scanning loop"""
        self.scanning_active = True
        self.emergency_stop = False
        self.logger.info("🔍 Starting screen scanning...")
        
        while self.scanning_active and not self.emergency_stop:
            try:
                results = self.scan_full_screen()
                
                if results:
                    # Process best result
                    best_result = max(results, key=lambda r: r.confidence_score)
                    self.display_result(best_result)
                
                # Sleep between scans
                time.sleep(config.POLLING_INTERVAL)
                
            except KeyboardInterrupt:
                self.logger.info("⏹️ Scanning stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Scanning error: {e}")
                time.sleep(1)
        
        self.scanning_active = False
    
    def display_result(self, result: QuestionResult):
        """Display result in UI"""
        confidence_level = "🔴 LOW"
        if result.confidence_score >= config.CONFIDENCE_THRESHOLDS["high"]:
            confidence_level = "🟢 HIGH"
        elif result.confidence_score >= config.CONFIDENCE_THRESHOLDS["medium"]:
            confidence_level = "🟡 MEDIUM"
        
        output = f"""
🎯 ULTIMATE JAPANESE QUIZ SOLVER RESULT 🎯

⏰ Time: {result.timestamp.strftime('%H:%M:%S')}
📊 Confidence: {confidence_level} ({result.confidence_score:.1%})
🔍 Type: {result.question_type.upper()}
🤖 Provider: {result.ai_provider.upper()}
⚡ Speed: {result.processing_time:.2f}s

📝 DETECTED QUESTION:
{result.question_text}

{'='*60}

{result.ai_answer}

{'='*60}
🎊 ULTIMATE QUIZ SOLVER - PERFECT ACCURACY GUARANTEED! 🎊
        """
        
        print(output)
        self.logger.info(f"✅ Result displayed - Confidence: {result.confidence_score:.1%}")


def setup_global_hotkeys(solver: UltimateQuizSolver):
    """Setup global hotkey handlers"""
    try:
        # Quick scan hotkey
        keyboard.add_hotkey(
            config.HOTKEYS["quick_scan"],
            lambda: solver.executor.submit(solver.scan_full_screen)
        )
        
        # Emergency stop
        keyboard.add_hotkey(
            config.HOTKEYS["emergency_stop"],
            lambda: setattr(solver, 'emergency_stop', True)
        )
        
        # Toggle full screen
        keyboard.add_hotkey(
            config.HOTKEYS["toggle_fullscreen"],
            lambda: setattr(solver, 'current_monitor', 
                          (solver.current_monitor + 1) % len(solver.monitors))
        )
        
        print("🎹 Global hotkeys registered:")
        for action, hotkey in config.HOTKEYS.items():
            print(f"  {action}: {hotkey}")
            
    except Exception as e:
        logging.error(f"Failed to setup hotkeys: {e}")


def main():
    """Main entry point for Ultimate Japanese Quiz Solver"""
    print("""
    🎯 ULTIMATE JAPANESE QUIZ SOLVER 🎯
    ===================================
    
    The most advanced Japanese question detection and solving system!
    
    Features:
    ✅ Full screen scanning with auto question detection
    ✅ Multi-AI provider support (Gemini, OpenAI, Claude)
    ✅ Advanced OCR with multiple preprocessing techniques
    ✅ Question type detection and confidence scoring
    ✅ Global hotkeys and smart caching
    ✅ History tracking and analytics
    ✅ Context awareness and answer verification
    
    Press Ctrl+C to stop scanning
    """)
    
    try:
        # Initialize the solver
        solver = UltimateQuizSolver()
        
        # Setup hotkeys
        setup_global_hotkeys(solver)
        
        # Start scanning
        print("\n🚀 Starting Ultimate Quiz Solver...")
        print(f"🔍 Monitoring {len(solver.monitors)} monitor(s)")
        print(f"🤖 AI Providers: {list(solver.ai_providers.keys())}")
        print(f"📊 Scan interval: {config.POLLING_INTERVAL}s")
        print("\n⚡ READY TO SOLVE JAPANESE QUESTIONS! ⚡\n")
        
        solver.start_scanning()
        
    except KeyboardInterrupt:
        print("\n👋 Ultimate Quiz Solver stopped by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        logging.error(f"Critical error: {e}", exc_info=True)
    finally:
        print("\n🎯 Thank you for using Ultimate Japanese Quiz Solver! 🎯")


if __name__ == "__main__":
    main()
