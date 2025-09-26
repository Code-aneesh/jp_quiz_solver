import os

# ============================================================
# ULTIMATE JAPANESE QUIZ SOLVER CONFIGURATION
# ============================================================

# PATH SETTINGS
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DATA_DIR = "quiz_data"
HISTORY_DB = os.path.join(DATA_DIR, "quiz_history.db")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

# SCREEN CAPTURE SETTINGS
CAPTURE_REGION = {"left": 0, "top": 0, "width": 1920, "height": 1080}  # Full screen by default
FULL_SCREEN_SCAN = True  # Scan entire screen for questions
AUTO_DETECT_REGIONS = True  # Automatically detect question areas
MIN_QUESTION_SIZE = {"width": 100, "height": 50}  # Minimum question box size
MAX_QUESTION_SIZE = {"width": 1200, "height": 800}  # Maximum question box size

# AI PROVIDER SETTINGS
AI_PROVIDER = "gemini"  # Primary provider
AI_FALLBACK_PROVIDER = "openai"  # Fallback if primary fails
AI_PROVIDERS = {
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE"),
        "model": "gemini-1.5-flash",  # Fast and accurate model
        "temperature": 0.0,
        "max_tokens": 2048
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": "gpt-4o",
        "temperature": 0.0,
        "max_tokens": 2048
    },
    "claude": {
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "model": "claude-3-sonnet-20240229",
        "temperature": 0.0,
        "max_tokens": 2048
    }
}

# Legacy settings for backward compatibility
GEMINI_API_KEY = AI_PROVIDERS["gemini"]["api_key"]
GEMINI_MODEL = AI_PROVIDERS["gemini"]["model"]
OPENAI_MODEL = AI_PROVIDERS["openai"]["model"]

# ADVANCED OCR SETTINGS
OCR_LANGUAGE = "jpn+eng"  # Japanese + English
OCR_CONFIGS = {
    "default": "--psm 6 --oem 3",
    "single_block": "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZあいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンー々〜？！（）「」『』。、①②③④⑤",
    "multiline": "--psm 4 --oem 3",
    "single_word": "--psm 8 --oem 3",
    "sparse": "--psm 11 --oem 3"
}
OCR_CONFIG = OCR_CONFIGS["single_block"]  # Default config

# IMAGE PREPROCESSING SETTINGS
IMAGE_ENHANCEMENTS = {
    "scale_factor": 3.0,  # Scale up for better OCR
    "noise_reduction": True,
    "contrast_enhancement": True,
    "sharpening": True,
    "edge_detection": True,
    "morphological_ops": True,
    "adaptive_threshold": True
}

# QUESTION TYPE DETECTION
QUESTION_PATTERNS = {
    "multiple_choice": {
        "patterns": [r"[①②③④⑤]", r"[1234５]", r"[ABCDE]", r"[ア-オ]", r"\([1-5]\)", r"[1-5]\."],
        "min_options": 2,
        "max_options": 6
    },
    "true_false": {
        "patterns": [r"正しい|間違い|True|False|○|×|はい|いいえ"],
        "keywords": ["正", "誤", "True", "False", "○", "×"]
    },
    "fill_blank": {
        "patterns": [r"_{2,}", r"（\s*）", r"［\s*］", r"\[\s*\]"],
        "keywords": ["空欄", "記入", "書き"]
    },
    "essay": {
        "patterns": [r"説明|理由|なぜ|どのよう|書きなさい|述べ"],
        "keywords": ["説明", "理由", "述べ", "書きなさい"]
    }
}

# CONFIDENCE SCORING SETTINGS
CONFIDENCE_THRESHOLDS = {
    "high": 0.85,
    "medium": 0.70,
    "low": 0.50
}

CONFIDENCE_FACTORS = {
    "ocr_quality": 0.30,     # Text clarity from OCR
    "question_completeness": 0.25,  # How complete the question appears
    "ai_certainty": 0.25,    # AI model confidence
    "pattern_match": 0.20     # How well question matches known patterns
}

# PERFORMANCE SETTINGS
POLLING_INTERVAL = 0.5  # Faster scanning - 2 FPS
STABLE_TEXT_THRESHOLD = 2  # Text must be stable for 2 captures
CACHE_SIZE = 1000  # Number of cached results
CACHE_EXPIRY = 3600  # Cache expiry in seconds (1 hour)

# HOTKEY SETTINGS
HOTKEYS = {
    "quick_scan": "ctrl+shift+q",
    "select_region": "ctrl+shift+r", 
    "show_history": "ctrl+shift+h",
    "toggle_fullscreen": "ctrl+shift+f",
    "emergency_stop": "ctrl+shift+x",
    "cycle_provider": "ctrl+shift+p",
    "screenshot": "ctrl+shift+s"
}

# UI SETTINGS
UI_THEME = "dark"  # "dark" or "light"
UI_COLORS = {
    "dark": {
        "bg": "#1a1a1a",
        "fg": "#ffffff",
        "accent": "#00ff88",
        "error": "#ff4444",
        "warning": "#ffaa00",
        "success": "#00ff88",
        "info": "#44aaff"
    },
    "light": {
        "bg": "#ffffff",
        "fg": "#000000", 
        "accent": "#007744",
        "error": "#cc0000",
        "warning": "#cc6600",
        "success": "#007744",
        "info": "#0066cc"
    }
}

WINDOW_SIZE = {"width": 800, "height": 600}
ALWAYS_ON_TOP = True
TRANSPARENCY = 0.95

# LOGGING SETTINGS
LOGGING = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "file_logging": True,
    "console_logging": True,
    "max_log_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# ANALYTICS SETTINGS
ANALYTICS = {
    "track_performance": True,
    "track_accuracy": True,
    "save_screenshots": False,  # Privacy setting
    "save_questions": True,
    "export_formats": ["json", "csv", "pdf"]
}

# LANGUAGE DETECTION
LANGUAGE_DETECTION = {
    "auto_detect": True,
    "supported_langs": ["ja", "en", "ko", "zh"],
    "confidence_threshold": 0.8
}

# EMERGENCY SETTINGS
EMERGENCY_MODE = {
    "enabled": True,
    "hotkey": "ctrl+alt+shift+x",
    "actions": ["stop_scanning", "hide_window", "clear_history"]
}

# ADVANCED FEATURES
FEATURES = {
    "auto_region_detection": True,
    "multi_monitor_support": True,
    "context_awareness": True,
    "answer_verification": True,
    "smart_caching": True,
    "batch_processing": True,
    "real_time_translation": True,
    "voice_output": False,  # Experimental
    "api_rate_limiting": True
}
