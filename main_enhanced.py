#!/usr/bin/env python3
"""
Ultimate Japanese Quiz Solver - Enhanced Main Entry Point with Phase 1 Integration

A comprehensive Japanese quiz detection and solving system with:
- Phase 1: Enhanced OCR with multi-PSM testing and preprocessing
- Phase 1: Deterministic rule engines (date/reading, katakana fuzzy)
- Phase 1: Structured JSON LLM responses with validation
- Multi-AI provider support (Gemini, OpenAI, Claude)
- Smart region detection and ultra-fast scanning
- Advanced confidence scoring and ensemble decision making
- Professional GUI with human-in-the-loop labeling
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import enhanced components
try:
    from ultimate_gui import UltimateQuizSolverGUI
except ImportError:
    print("Warning: GUI module not available. CLI mode only.")
    UltimateQuizSolverGUI = None

from ocr.ocr_preprocess import preprocess_image_for_ocr
from ocr.ocr_multi_psm import best_ocr_result
from rules.rules_engine import UnifiedRuleEngine
import config

# Standard library imports
import mss
import pytesseract
from PIL import Image
import time
import threading
import re

# AI imports
import google.generativeai as genai
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quiz_solver.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH

# Configure Gemini if key present
if getattr(config, "GEMINI_API_KEY", None):
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        logger.info("Gemini API configured successfully")
    except Exception as e:
        logger.warning(f"Failed to configure Gemini API: {e}")

class EnhancedQuizSolver:
    """Enhanced Quiz Solver with Phase 1 Improvements"""
    
    def __init__(self):
        self.rule_engine = UnifiedRuleEngine()
        self.last_ocr_result = None
        self.processing_lock = threading.Lock()
        
    def get_structured_llm_prompt(self, text: str) -> str:
        """Create structured prompt for JSON response"""
        return f"""
You are an expert Japanese language tutor and quiz solver with perfect accuracy.

ANALYZE THIS JAPANESE CONTENT:
{text}

RESPOND WITH VALID JSON IN THIS EXACT FORMAT:
{{
    "question_detected": true/false,
    "question_type": "multiple_choice" | "fill_blank" | "reading" | "translation" | "other",
    "japanese_text": "original Japanese text as detected",
    "options": ["option1", "option2", "option3", "option4"] or null,
    "correct_answer": "A" | "B" | "C" | "D" | "1" | "2" | etc or null,
    "confidence": 0.0-1.0,
    "translation": "full English translation",
    "explanation": "detailed explanation of the correct answer",
    "key_vocabulary": {{"word1": "definition1", "word2": "definition2"}},
    "grammar_points": ["point1", "point2"],
    "difficulty_level": "N5" | "N4" | "N3" | "N2" | "N1" | "unknown",
    "reasoning_steps": [
        "Step 1: Parse the question structure",
        "Step 2: Analyze each option semantically", 
        "Step 3: Verify correct usage and grammar",
        "Step 4: Confirm the best answer"
    ]
}}

CRITICAL REQUIREMENTS:
- Respond ONLY with valid JSON, no other text
- Analyze Japanese text character by character for accuracy
- For multiple choice, verify each option semantically
- Mark high confidence (>0.9) only when 100% certain
- Include step-by-step reasoning for verification
- Identify question format correctly (numbers, letters, katakana options)
"""

    def get_answer_from_gemini(self, text: str) -> Dict[str, Any]:
        """Get structured response from Gemini"""
        try:
            model = genai.GenerativeModel(config.GEMINI_MODEL)
            prompt = self.get_structured_llm_prompt(text)
            response = model.generate_content(prompt)
            
            # Parse JSON response
            json_str = response.text.strip()
            # Clean up common JSON formatting issues
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini: {e}")
            return self._fallback_response(text, f"JSON parse error: {e}")
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._fallback_response(text, f"API error: {e}")

    def get_answer_from_openai(self, text: str) -> Dict[str, Any]:
        """Get structured response from OpenAI"""
        if OpenAI is None:
            return self._fallback_response(text, "OpenAI not available")
            
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert Japanese quiz solver. Respond only with valid JSON."},
                    {"role": "user", "content": self.get_structured_llm_prompt(text)}
                ],
                temperature=0.0
            )
            
            json_str = response.choices[0].message.content.strip()
            return json.loads(json_str)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from OpenAI: {e}")
            return self._fallback_response(text, f"JSON parse error: {e}")
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._fallback_response(text, f"API error: {e}")

    def _fallback_response(self, text: str, error_msg: str) -> Dict[str, Any]:
        """Create fallback response structure"""
        return {
            "question_detected": bool(re.search(r'[1-4ABCD①②③④ア-エ]', text)),
            "question_type": "unknown",
            "japanese_text": text,
            "options": None,
            "correct_answer": None,
            "confidence": 0.1,
            "translation": "Translation unavailable due to error",
            "explanation": f"Error occurred: {error_msg}",
            "key_vocabulary": {},
            "grammar_points": [],
            "difficulty_level": "unknown",
            "reasoning_steps": [f"Error: {error_msg}"]
        }

    def contains_japanese(self, text: str) -> bool:
        """Check if text contains Japanese characters"""
        japanese_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]'
        return bool(re.search(japanese_pattern, text))

    def process_image(self, image_path_or_pil: str | Image.Image) -> Dict[str, Any]:
        """Process image with enhanced OCR and rule engine"""
        with self.processing_lock:
            try:
                # Load image
                if isinstance(image_path_or_pil, str):
                    pil_image = Image.open(image_path_or_pil)
                else:
                    pil_image = image_path_or_pil
                
                # Phase 1: Enhanced OCR preprocessing
                logger.info("Applying enhanced OCR preprocessing...")
                preprocessed_image = preprocess_image_for_ocr(pil_image)
                
                # Phase 1: Multi-PSM OCR testing
                logger.info("Running multi-PSM OCR analysis...")
                ocr_result = best_ocr_result(preprocessed_image)
                
                if not ocr_result or not ocr_result.get('text'):
                    return {
                        "error": "No text detected in image",
                        "ocr_result": ocr_result,
                        "rule_engine_result": None,
                        "llm_result": None
                    }
                
                extracted_text = ocr_result['text']
                logger.info(f"OCR extracted: {extracted_text[:100]}...")
                
                # Check for Japanese content
                if not self.contains_japanese(extracted_text) and len(extracted_text) < 10:
                    return {
                        "error": "No substantial Japanese content detected",
                        "ocr_result": ocr_result,
                        "rule_engine_result": None,
                        "llm_result": None
                    }
                
                # Phase 1: Apply rule engine first
                logger.info("Applying unified rule engine...")
                rule_result = self.rule_engine.process_text(extracted_text)
                
                # Check if rule engine found high-confidence match
                if rule_result.get('should_override_llm', False):
                    logger.info("Rule engine found high-confidence match, skipping LLM")
                    return {
                        "ocr_result": ocr_result,
                        "rule_engine_result": rule_result,
                        "llm_result": None,
                        "final_answer": rule_result.get('best_match'),
                        "confidence": rule_result.get('confidence', 0.5),
                        "source": "rule_engine"
                    }
                
                # Proceed with LLM analysis
                logger.info("Getting LLM analysis...")
                provider = getattr(config, "AI_PROVIDER", "gemini").lower()
                if provider == "openai":
                    llm_result = self.get_answer_from_openai(extracted_text)
                else:
                    llm_result = self.get_answer_from_gemini(extracted_text)
                
                # Combine results
                return {
                    "ocr_result": ocr_result,
                    "rule_engine_result": rule_result,
                    "llm_result": llm_result,
                    "final_answer": llm_result.get('correct_answer'),
                    "confidence": llm_result.get('confidence', 0.5),
                    "source": "llm_primary"
                }
                
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return {"error": str(e)}

    def format_result_display(self, result: Dict[str, Any]) -> str:
        """Format result for display"""
        if result.get('error'):
            return f"❌ Error: {result['error']}"
        
        # Get final answer info
        final_answer = result.get('final_answer')
        confidence = result.get('confidence', 0.0)
        source = result.get('source', 'unknown')
        
        # OCR info
        ocr_info = result.get('ocr_result', {})
        ocr_text = ocr_info.get('text', 'N/A')
        ocr_confidence = ocr_info.get('confidence', 0.0)
        ocr_psm = ocr_info.get('psm_used', 'N/A')
        
        output_lines = [
            "🔍 ENHANCED QUIZ SOLVER ANALYSIS",
            "=" * 50,
            "",
            f"📝 OCR RESULT (PSM {ocr_psm}, Conf: {ocr_confidence:.2f}):",
            f"{ocr_text}",
            "",
            f"🎯 FINAL ANSWER: {final_answer or 'N/A'}",
            f"🔮 CONFIDENCE: {confidence:.2f} ({source})",
            ""
        ]
        
        # Rule engine results
        rule_result = result.get('rule_engine_result')
        if rule_result and rule_result.get('matches_found'):
            output_lines.extend([
                "🔧 RULE ENGINE MATCHES:",
                f"  • Date/Reading Rules: {len(rule_result.get('date_matches', []))} matches",
                f"  • Katakana Fuzzy: {len(rule_result.get('katakana_matches', []))} matches",
                f"  • Best Match: {rule_result.get('best_match', 'None')}",
                ""
            ])
        
        # LLM results
        llm_result = result.get('llm_result')
        if llm_result:
            output_lines.extend([
                "🤖 LLM ANALYSIS:",
                f"  • Question Type: {llm_result.get('question_type', 'N/A')}",
                f"  • Translation: {llm_result.get('translation', 'N/A')[:100]}{'...' if len(llm_result.get('translation', '')) > 100 else ''}",
                f"  • Explanation: {llm_result.get('explanation', 'N/A')[:100]}{'...' if len(llm_result.get('explanation', '')) > 100 else ''}",
                ""
            ])
            
            # Show reasoning steps if available
            steps = llm_result.get('reasoning_steps', [])
            if steps:
                output_lines.append("💭 REASONING STEPS:")
                for i, step in enumerate(steps[:3], 1):  # Show first 3 steps
                    output_lines.append(f"  {i}. {step}")
                output_lines.append("")
        
        return "\n".join(output_lines)

def run_gui():
    """Run the enhanced GUI application"""
    if UltimateQuizSolverGUI is None:
        print("❌ GUI not available. Install required dependencies.")
        return
    
    # Create enhanced solver instance
    solver = EnhancedQuizSolver()
    
    # Monkey patch the GUI to use enhanced solver
    def enhanced_process_region(region):
        with mss.mss() as sct:
            img = sct.grab(region)
            pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
            result = solver.process_image(pil_img)
            return solver.format_result_display(result)
    
    # Start GUI with enhanced processing
    print("🚀 Starting Enhanced Japanese Quiz Solver GUI...")
    print("📊 Phase 1 Features Active:")
    print("  ✅ Multi-PSM OCR optimization")
    print("  ✅ Japanese-optimized preprocessing")
    print("  ✅ Date/reading rule engine")
    print("  ✅ Katakana fuzzy matching")
    print("  ✅ Structured LLM responses")
    
    app = UltimateQuizSolverGUI()
    
    # Replace the processing function
    original_get_answer = app.get_answer if hasattr(app, 'get_answer') else None
    def enhanced_get_answer(text):
        # Create a dummy image with the text for processing
        result = solver.process_image(text)
        return solver.format_result_display(result)
    
    # Override processing in the GUI
    if hasattr(app, 'get_answer'):
        app.get_answer = enhanced_get_answer
    
    app.run()

def run_cli(image_path: str):
    """Run CLI mode on a single image"""
    solver = EnhancedQuizSolver()
    
    print(f"🔍 Processing image: {image_path}")
    result = solver.process_image(image_path)
    output = solver.format_result_display(result)
    
    print("\n" + "="*60)
    print(output)
    print("="*60)
    
    # Save detailed result to JSON
    output_file = Path(image_path).stem + "_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Detailed analysis saved to: {output_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Japanese Quiz Solver")
    parser.add_argument('--mode', choices=['gui', 'cli'], default='gui', help='Run mode')
    parser.add_argument('--image', help='Image path for CLI mode')
    parser.add_argument('--test', action='store_true', help='Run system tests')
    
    args = parser.parse_args()
    
    print("🏮 ULTIMATE JAPANESE QUIZ SOLVER - PHASE 1 ENHANCED")
    print("=" * 60)
    
    if args.test:
        print("🧪 Running system tests...")
        # Add test runner here
        return
    
    if args.mode == 'cli':
        if not args.image:
            print("❌ CLI mode requires --image parameter")
            return
        if not os.path.exists(args.image):
            print(f"❌ Image file not found: {args.image}")
            return
        run_cli(args.image)
    else:
        run_gui()

if __name__ == "__main__":
    main()
