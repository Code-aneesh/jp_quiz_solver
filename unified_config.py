#!/usr/bin/env python3
"""
🎯 ULTIMATE JAPANESE QUIZ SOLVER - UNIFIED CONFIGURATION
=======================================================

This unified configuration system consolidates all settings from config.py 
and enhanced_config.py into a single, well-organized, validated configuration.

Features:
- Environment variable support with fallbacks
- Configuration validation
- Dynamic configuration updates
- Backward compatibility
- Type checking and default values
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """Path-related configuration"""
    tesseract_path: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    data_dir: str = "quiz_data" 
    logs_dir: str = field(default_factory=lambda: os.path.join("quiz_data", "logs"))
    history_db: str = field(default_factory=lambda: os.path.join("quiz_data", "quiz_history.db"))
    cache_dir: str = field(default_factory=lambda: os.path.join("quiz_data", "cache"))
    exports_dir: str = field(default_factory=lambda: os.path.join("quiz_data", "exports"))
    
    def __post_init__(self):
        """Ensure all paths are absolute and directories exist"""
        # Convert relative paths to absolute
        for attr in ["data_dir", "logs_dir", "cache_dir", "exports_dir"]:
            path = getattr(self, attr)
            if not os.path.isabs(path):
                setattr(self, attr, os.path.abspath(path))
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.logs_dir, self.cache_dir, self.exports_dir]:
            os.makedirs(directory, exist_ok=True)


@dataclass  
class ScreenCaptureConfig:
    """Screen capture and region detection settings"""
    capture_region: Dict[str, int] = field(default_factory=lambda: {
        "left": 0, "top": 0, "width": 1920, "height": 1080
    })
    full_screen_scan: bool = True
    auto_detect_regions: bool = True
    min_question_size: Dict[str, int] = field(default_factory=lambda: {
        "width": 100, "height": 50
    })
    max_question_size: Dict[str, int] = field(default_factory=lambda: {
        "width": 1200, "height": 800
    })
    

@dataclass
class AIProviderConfig:
    """AI provider settings with fallback support"""
    primary_provider: str = "gemini"
    fallback_provider: str = "openai"
    
    # Provider configurations
    providers: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "gemini": {
            "api_key": os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_KEY_HERE"),
            "model": "gemini-1.5-flash",
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
    })
    
    def get_active_providers(self) -> List[str]:
        """Return list of providers with valid API keys"""
        active = []
        for name, config in self.providers.items():
            api_key = config.get("api_key", "")
            if api_key and api_key not in ["", "YOUR_GEMINI_KEY_HERE"]:
                active.append(name)
        return active
    
    def is_provider_available(self, provider: str) -> bool:
        """Check if a specific provider is available"""
        if provider not in self.providers:
            return False
        api_key = self.providers[provider].get("api_key", "")
        return bool(api_key and api_key not in ["", "YOUR_GEMINI_KEY_HERE"])


@dataclass
class OCRConfig:
    """OCR and image processing configuration"""
    language: str = "jpn+eng"
    
    # OCR configuration presets
    configs: Dict[str, str] = field(default_factory=lambda: {
        "default": "--psm 6 --oem 3",
        "single_block": "--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZあいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンー々〜？！（）「」『』。、①②③④⑤",
        "multiline": "--psm 4 --oem 3",
        "single_word": "--psm 8 --oem 3", 
        "sparse": "--psm 11 --oem 3"
    })
    
    default_config: str = "single_block"
    
    # Image enhancement settings
    image_enhancements: Dict[str, Any] = field(default_factory=lambda: {
        "scale_factor": 3.0,
        "noise_reduction": True,
        "contrast_enhancement": True,
        "sharpening": True,
        "edge_detection": True,
        "morphological_ops": True,
        "adaptive_threshold": True
    })


@dataclass
class QuestionDetectionConfig:
    """Question type detection patterns and settings"""
    patterns: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
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
    })


@dataclass
class ConfidenceConfig:
    """Confidence scoring configuration"""
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high": 0.85,
        "medium": 0.70, 
        "low": 0.50
    })
    
    factors: Dict[str, float] = field(default_factory=lambda: {
        "ocr_quality": 0.30,
        "question_completeness": 0.25,
        "ai_certainty": 0.25,
        "pattern_match": 0.20
    })


@dataclass
class PerformanceConfig:
    """Performance and caching settings"""
    polling_interval: float = 0.5
    stable_text_threshold: int = 2
    cache_size: int = 1000
    cache_expiry: int = 3600
    max_workers: int = 4
    processing_timeout: int = 30


@dataclass
class UIConfig:
    """User interface configuration"""
    theme: str = "dark"
    window_size: Dict[str, int] = field(default_factory=lambda: {
        "width": 800, "height": 600
    })
    always_on_top: bool = True
    transparency: float = 0.95
    
    colors: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "dark": {
            "bg": "#1a1a1a",
            "fg": "#ffffff", 
            "accent": "#00ff88",
            "success": "#00ff88",
            "error": "#ff4444",
            "warning": "#ffaa00",
            "info": "#44aaff"
        },
        "light": {
            "bg": "#ffffff",
            "fg": "#000000",
            "accent": "#2196F3",
            "success": "#4CAF50", 
            "error": "#f44336",
            "warning": "#ff9800",
            "info": "#2196f3"
        }
    })


@dataclass
class HotkeyConfig:
    """Global hotkey configuration"""
    hotkeys: Dict[str, str] = field(default_factory=lambda: {
        "quick_scan": "ctrl+shift+q",
        "select_region": "ctrl+shift+r",
        "show_history": "ctrl+shift+h", 
        "toggle_fullscreen": "ctrl+shift+f",
        "emergency_stop": "ctrl+shift+x",
        "cycle_provider": "ctrl+shift+p",
        "screenshot": "ctrl+shift+s"
    })


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    file_logging: bool = True
    console_logging: bool = True
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class AdvancedFeatureConfig:
    """Advanced feature toggles"""
    auto_region_detection: bool = True
    multi_monitor_support: bool = True
    context_awareness: bool = True
    answer_verification: bool = True
    smart_caching: bool = True
    batch_processing: bool = True
    real_time_translation: bool = True
    voice_output: bool = False  # Experimental
    api_rate_limiting: bool = True


class UnifiedConfig:
    """Unified configuration manager for Ultimate Japanese Quiz Solver"""
    
    def __init__(self):
        """Initialize configuration with all subsystems"""
        self.paths = PathConfig()
        self.screen_capture = ScreenCaptureConfig()
        self.ai_providers = AIProviderConfig()
        self.ocr = OCRConfig()
        self.question_detection = QuestionDetectionConfig()
        self.confidence = ConfidenceConfig()
        self.performance = PerformanceConfig()
        self.ui = UIConfig()
        self.hotkeys = HotkeyConfig()
        self.logging = LoggingConfig()
        self.features = AdvancedFeatureConfig()
        
        # Backward compatibility attributes
        self._setup_backward_compatibility()
        
        # Validate configuration
        self.validate()
    
    def _setup_backward_compatibility(self):
        """Set up backward compatibility with old config attributes"""
        # Map old attribute names to new ones
        self.TESSERACT_PATH = self.paths.tesseract_path
        self.DATA_DIR = self.paths.data_dir
        self.LOGS_DIR = self.paths.logs_dir
        self.HISTORY_DB = self.paths.history_db
        
        self.CAPTURE_REGION = self.screen_capture.capture_region
        self.FULL_SCREEN_SCAN = self.screen_capture.full_screen_scan
        self.MIN_QUESTION_SIZE = self.screen_capture.min_question_size
        self.MAX_QUESTION_SIZE = self.screen_capture.max_question_size
        
        self.AI_PROVIDER = self.ai_providers.primary_provider
        self.AI_FALLBACK_PROVIDER = self.ai_providers.fallback_provider
        self.AI_PROVIDERS = self.ai_providers.providers
        
        # Legacy single provider configs
        self.GEMINI_API_KEY = self.ai_providers.providers["gemini"]["api_key"]
        self.GEMINI_MODEL = self.ai_providers.providers["gemini"]["model"]
        self.OPENAI_MODEL = self.ai_providers.providers["openai"]["model"]
        self.AI_PROVIDERS = self.ai_providers.providers
        
        self.OCR_LANGUAGE = self.ocr.language
        self.OCR_CONFIG = self.ocr.configs[self.ocr.default_config]
        self.OCR_CONFIGS = self.ocr.configs
        self.IMAGE_ENHANCEMENTS = self.ocr.image_enhancements
        
        self.QUESTION_PATTERNS = self.question_detection.patterns
        self.CONFIDENCE_THRESHOLDS = self.confidence.thresholds
        self.CONFIDENCE_FACTORS = self.confidence.factors
        
        self.POLLING_INTERVAL = self.performance.polling_interval
        self.STABLE_TEXT_THRESHOLD = self.performance.stable_text_threshold
        self.CACHE_SIZE = self.performance.cache_size
        self.CACHE_EXPIRY = self.performance.cache_expiry
        
        self.UI_THEME = self.ui.theme
        self.UI_COLORS = self.ui.colors
        self.WINDOW_SIZE = self.ui.window_size
        self.ALWAYS_ON_TOP = self.ui.always_on_top
        
        self.HOTKEYS = self.hotkeys.hotkeys
        self.LOGGING = self.logging.__dict__
        self.FEATURES = self.features.__dict__
        
        # Additional backward compatibility for direct access
        for attr_name in dir(self.paths):
            if not attr_name.startswith('_'):
                setattr(self, attr_name.upper(), getattr(self.paths, attr_name))
        
        for attr_name in dir(self.performance):
            if not attr_name.startswith('_') and not hasattr(self, attr_name.upper()):
                setattr(self, attr_name.upper(), getattr(self.performance, attr_name))
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        issues = []
        
        # Validate paths
        if not os.path.exists(self.paths.tesseract_path):
            issues.append(f"Tesseract not found at: {self.paths.tesseract_path}")
        
        # Validate AI providers
        active_providers = self.ai_providers.get_active_providers()
        if not active_providers:
            issues.append("No AI providers have valid API keys configured")
        
        if self.ai_providers.primary_provider not in active_providers:
            if active_providers:
                issues.append(f"Primary provider '{self.ai_providers.primary_provider}' not available, using '{active_providers[0]}'")
                self.ai_providers.primary_provider = active_providers[0]
            else:
                issues.append("No valid AI providers available")
        
        # Validate OCR settings
        if self.ocr.default_config not in self.ocr.configs:
            issues.append(f"Invalid OCR config: {self.ocr.default_config}")
        
        # Validate performance settings  
        if self.performance.polling_interval <= 0:
            issues.append("Polling interval must be positive")
        
        if self.performance.cache_size <= 0:
            issues.append("Cache size must be positive")
        
        # Validate UI settings
        if self.ui.theme not in self.ui.colors:
            issues.append(f"Invalid UI theme: {self.ui.theme}")
        
        # Log issues
        if issues:
            print("⚠️  Configuration Issues Found:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print("✅ Configuration validation successful")
        return True
    
    def get_active_ai_providers(self) -> List[str]:
        """Get list of available AI providers"""
        return self.ai_providers.get_active_providers()
    
    def is_provider_available(self, provider: str) -> bool:
        """Check if specific AI provider is available"""
        return self.ai_providers.is_provider_available(provider)
    
    def get_ocr_config(self, config_name: Optional[str] = None) -> str:
        """Get OCR configuration string"""
        config_name = config_name or self.ocr.default_config
        return self.ocr.configs.get(config_name, self.ocr.configs["default"])
    
    def get_ui_colors(self, theme: Optional[str] = None) -> Dict[str, str]:
        """Get UI colors for specified theme"""
        theme = theme or self.ui.theme
        return self.ui.colors.get(theme, self.ui.colors["dark"])
    
    def update_setting(self, section: str, key: str, value: Any) -> bool:
        """Update a configuration setting dynamically"""
        try:
            section_obj = getattr(self, section)
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)
                # Update backward compatibility attributes
                self._setup_backward_compatibility()
                return True
            return False
        except AttributeError:
            return False
    
    def export_config(self, filepath: str) -> bool:
        """Export current configuration to JSON file"""
        try:
            import json
            config_dict = {
                "paths": self.paths.__dict__,
                "screen_capture": self.screen_capture.__dict__,
                "ai_providers": self.ai_providers.__dict__,
                "ocr": self.ocr.__dict__, 
                "question_detection": self.question_detection.__dict__,
                "confidence": self.confidence.__dict__,
                "performance": self.performance.__dict__,
                "ui": self.ui.__dict__,
                "hotkeys": self.hotkeys.__dict__,
                "logging": self.logging.__dict__,
                "features": self.features.__dict__
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Failed to export config: {e}")
            return False
    
    def import_config(self, filepath: str) -> bool:
        """Import configuration from JSON file"""
        try:
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # Update configuration sections
            for section_name, section_data in config_dict.items():
                if hasattr(self, section_name):
                    section_obj = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
            
            # Update backward compatibility
            self._setup_backward_compatibility()
            self.validate()
            return True
        except Exception as e:
            print(f"Failed to import config: {e}")
            return False


# Create global configuration instance
config = UnifiedConfig()

# Backward compatibility - expose old module-level attributes
TESSERACT_PATH = config.TESSERACT_PATH
DATA_DIR = config.DATA_DIR
LOGS_DIR = config.LOGS_DIR
HISTORY_DB = config.HISTORY_DB
CAPTURE_REGION = config.CAPTURE_REGION
FULL_SCREEN_SCAN = config.FULL_SCREEN_SCAN
MIN_QUESTION_SIZE = config.MIN_QUESTION_SIZE
MAX_QUESTION_SIZE = config.MAX_QUESTION_SIZE
AI_PROVIDER = config.AI_PROVIDER
AI_FALLBACK_PROVIDER = config.AI_FALLBACK_PROVIDER
AI_PROVIDERS = config.AI_PROVIDERS
GEMINI_API_KEY = config.GEMINI_API_KEY
GEMINI_MODEL = config.GEMINI_MODEL
OPENAI_MODEL = config.OPENAI_MODEL
OCR_LANGUAGE = config.OCR_LANGUAGE
OCR_CONFIG = config.OCR_CONFIG
OCR_CONFIGS = config.OCR_CONFIGS
IMAGE_ENHANCEMENTS = config.IMAGE_ENHANCEMENTS
QUESTION_PATTERNS = config.QUESTION_PATTERNS
CONFIDENCE_THRESHOLDS = config.CONFIDENCE_THRESHOLDS
CONFIDENCE_FACTORS = config.CONFIDENCE_FACTORS
POLLING_INTERVAL = config.POLLING_INTERVAL
STABLE_TEXT_THRESHOLD = config.STABLE_TEXT_THRESHOLD
CACHE_SIZE = config.CACHE_SIZE
CACHE_EXPIRY = config.CACHE_EXPIRY
UI_THEME = config.UI_THEME
UI_COLORS = config.UI_COLORS
WINDOW_SIZE = config.WINDOW_SIZE
ALWAYS_ON_TOP = config.ALWAYS_ON_TOP
HOTKEYS = config.HOTKEYS
LOGGING = config.LOGGING
FEATURES = config.FEATURES

# New unified access
CONFIG = config

if __name__ == "__main__":
    # Configuration test
    print("🎯 UNIFIED CONFIGURATION TEST")
    print("=" * 40)
    
    print(f"✅ Tesseract Path: {config.TESSERACT_PATH}")
    print(f"✅ Data Directory: {config.DATA_DIR}")
    print(f"✅ Active AI Providers: {config.get_active_ai_providers()}")
    print(f"✅ UI Theme: {config.UI_THEME}")
    print(f"✅ OCR Language: {config.OCR_LANGUAGE}")
    
    # Test configuration export
    if config.export_config("test_config_export.json"):
        print("✅ Configuration export test passed")
        os.remove("test_config_export.json")
    
    print("\n🎉 Unified configuration system ready!")
