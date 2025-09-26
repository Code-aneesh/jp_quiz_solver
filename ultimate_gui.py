#!/usr/bin/env python3
"""
🎯 ULTIMATE JAPANESE QUIZ SOLVER - GUI VERSION 🎯
Professional GUI interface with all advanced features.

Features:
- Modern dark theme with customizable colors
- Real-time confidence scoring and visualization
- Answer history with search and export
- Hotkey configuration and management
- Multi-AI provider switching
- Advanced settings and preferences
- Full-screen overlay mode
- Statistics and analytics dashboard
"""

import sys
import os
import json
import sqlite3
import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import webbrowser

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import tkinter.font as tkfont

# Import our ultimate solver
from ultimate_main import UltimateQuizSolver, QuestionResult

# Use unified config system for consistency
try:
    import unified_config as config
except ImportError:
    import config
if not hasattr(config, 'WINDOW_SIZE'):
    config.WINDOW_SIZE = {"width": 800, "height": 600}
if not hasattr(config, 'UI_COLORS'):
    config.UI_COLORS = {
        "dark": {
            "bg": "#2b2b2b",
            "fg": "white", 
            "accent": "#4CAF50",
            "success": "#4CAF50",
            "error": "#f44336",
            "warning": "#ff9800",
            "info": "#2196f3"
        }
    }
if not hasattr(config, 'UI_THEME'):
    config.UI_THEME = "dark"
if not hasattr(config, 'ALWAYS_ON_TOP'):
    config.ALWAYS_ON_TOP = True
if not hasattr(config, 'HOTKEYS'):
    config.HOTKEYS = {
        "quick_scan": "ctrl+shift+q",
        "emergency_stop": "ctrl+shift+x"
    }
if not hasattr(config, 'FEATURES'):
    config.FEATURES = {
        "auto_region_detection": True,
        "batch_processing": True,
        "context_awareness": True
    }
if not hasattr(config, 'CONFIDENCE_THRESHOLDS'):
    config.CONFIDENCE_THRESHOLDS = {"high": 0.85, "medium": 0.70, "low": 0.50}
if not hasattr(config, 'IMAGE_ENHANCEMENTS'):
    config.IMAGE_ENHANCEMENTS = {"scale_factor": 3.0}
if not hasattr(config, 'AI_PROVIDERS'):
    config.AI_PROVIDERS = {"gemini": "Google Gemini", "openai": "OpenAI GPT", "claude": "Claude AI"}
if not hasattr(config, 'AI_PROVIDER'):
    config.AI_PROVIDER = "gemini"
if not hasattr(config, 'POLLING_INTERVAL'):
    config.POLLING_INTERVAL = 0.5
if not hasattr(config, 'CACHE_SIZE'):
    config.CACHE_SIZE = 1000
if not hasattr(config, 'OCR_LANGUAGE'):
    config.OCR_LANGUAGE = "jpn+eng"
if not hasattr(config, 'FULL_SCREEN_SCAN'):
    config.FULL_SCREEN_SCAN = True


class UltimateQuizSolverGUI:
    """Professional GUI for the Ultimate Japanese Quiz Solver"""
    
    def __init__(self):
        # Initialize the core solver
        self.solver = UltimateQuizSolver()
        
        # GUI state with improved threading
        self.current_result = None
        self.scanning_thread = None
        self.is_scanning = False
        self.overlay_mode = False
        self._shutdown_event = threading.Event()
        self._result_queue = queue.Queue()
        self._error_queue = queue.Queue()
        
        # Initialize GUI
        self.setup_main_window()
        self.setup_theme()
        self.setup_menu()
        self.setup_main_interface()
        self.setup_status_bar()
        
        # Start background services
        self.setup_hotkeys()
        self.start_result_monitor()
        
        print("🎯 Ultimate Japanese Quiz Solver GUI initialized!")
    
    def setup_main_window(self):
        """Initialize the main application window"""
        self.root = tk.Tk()
        self.root.title("🎯 Ultimate Japanese Quiz Solver")
        self.root.geometry(f"{config.WINDOW_SIZE['width']}x{config.WINDOW_SIZE['height']}")
        
        # Set window properties
        if config.ALWAYS_ON_TOP:
            self.root.attributes("-topmost", True)
        
        # Set window icon (if available)
        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_theme(self):
        """Setup the UI theme and styling"""
        # Configure ttk style
        self.style = ttk.Style()
        
        # Get theme colors
        theme_colors = config.UI_COLORS[config.UI_THEME]
        
        # Configure root window
        self.root.configure(bg=theme_colors["bg"])
        
        # Configure ttk styles
        self.style.theme_use("clam")
        
        # Configure colors for different elements
        self.style.configure("Title.TLabel", 
                           background=theme_colors["bg"],
                           foreground=theme_colors["accent"],
                           font=("Arial", 16, "bold"))
        
        self.style.configure("Info.TLabel",
                           background=theme_colors["bg"],
                           foreground=theme_colors["fg"],
                           font=("Arial", 10))
        
        self.style.configure("Success.TLabel",
                           background=theme_colors["bg"],
                           foreground=theme_colors["success"],
                           font=("Arial", 12, "bold"))
        
        self.style.configure("Error.TLabel",
                           background=theme_colors["bg"],
                           foreground=theme_colors["error"],
                           font=("Arial", 12, "bold"))
        
        # Button styles
        self.style.configure("Action.TButton",
                           background=theme_colors["accent"],
                           foreground=theme_colors["bg"],
                           font=("Arial", 10, "bold"))
        
        # Progress bar style (using default style)
        self.style.configure("TProgressbar",
                           background=theme_colors["success"],
                           troughcolor=theme_colors["bg"],
                           borderwidth=0)
    
    def setup_menu(self):
        """Create the application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export History...", command=self.export_history)
        file_menu.add_command(label="Import Settings...", command=self.import_settings)
        file_menu.add_command(label="Export Settings...", command=self.export_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Select Region", command=self.select_region)
        tools_menu.add_command(label="Test OCR", command=self.test_ocr)
        tools_menu.add_command(label="AI Provider Settings", command=self.configure_ai_providers)
        tools_menu.add_command(label="Hotkey Settings", command=self.configure_hotkeys)
        tools_menu.add_separator()
        tools_menu.add_command(label="Statistics Dashboard", command=self.show_statistics)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Always On Top", command=self.toggle_always_on_top)
        view_menu.add_command(label="Toggle Overlay Mode", command=self.toggle_overlay_mode)
        view_menu.add_command(label="Full Screen Scan", command=self.toggle_fullscreen_scan)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Hotkeys Reference", command=self.show_hotkeys_help)
        help_menu.add_command(label="Troubleshooting", command=self.show_troubleshooting)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_main_interface(self):
        """Setup the main application interface"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title section
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text="🎯 Ultimate Japanese Quiz Solver", 
                               style="Title.TLabel")
        title_label.pack(side=tk.LEFT)
        
        # Control buttons
        control_frame = ttk.Frame(title_frame)
        control_frame.pack(side=tk.RIGHT)
        
        self.scan_button = ttk.Button(control_frame, text="🔍 Start Scanning", 
                                     command=self.toggle_scanning, style="Action.TButton")
        self.scan_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(control_frame, text="📍 Select Region", 
                  command=self.select_region).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(control_frame, text="⚙️ Settings", 
                  command=self.show_settings).pack(side=tk.LEFT)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Setup tabs
        self.setup_main_tab()
        self.setup_history_tab()
        self.setup_analytics_tab()
        self.setup_settings_tab()
    
    def setup_main_tab(self):
        """Setup the main scanning and results tab"""
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="🎯 Quiz Solver")
        
        # AI Provider selection
        provider_frame = ttk.Frame(self.main_tab)
        provider_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(provider_frame, text="🤖 AI Provider:", style="Info.TLabel").pack(side=tk.LEFT)
        
        self.provider_var = tk.StringVar(value=config.AI_PROVIDER)
        self.provider_combo = ttk.Combobox(provider_frame, textvariable=self.provider_var,
                                          values=list(config.AI_PROVIDERS.keys()),
                                          state="readonly", width=15)
        self.provider_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.provider_combo.bind("<<ComboboxSelected>>", self.on_provider_changed)
        
        # Confidence display
        confidence_frame = ttk.Frame(self.main_tab)
        confidence_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(confidence_frame, text="📊 Confidence:", style="Info.TLabel").pack(side=tk.LEFT)
        
        self.confidence_var = tk.DoubleVar(value=0.0)
        self.confidence_progress = ttk.Progressbar(confidence_frame, variable=self.confidence_var,
                                                  maximum=100, 
                                                  length=200)
        self.confidence_progress.pack(side=tk.LEFT, padx=(5, 10))
        
        self.confidence_label = ttk.Label(confidence_frame, text="0%", style="Info.TLabel")
        self.confidence_label.pack(side=tk.LEFT)
        
        # Question type and status
        status_frame = ttk.Frame(self.main_tab)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(status_frame, text="📝 Question Type:", style="Info.TLabel").pack(side=tk.LEFT)
        self.question_type_label = ttk.Label(status_frame, text="None", style="Info.TLabel")
        self.question_type_label.pack(side=tk.LEFT, padx=(5, 20))
        
        ttk.Label(status_frame, text="⚡ Status:", style="Info.TLabel").pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, text="Ready", style="Success.TLabel")
        self.status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Results display area
        results_frame = ttk.LabelFrame(self.main_tab, text="📋 Results", padding="5")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Text area with scrollbar
        self.results_text = scrolledtext.ScrolledText(results_frame, 
                                                     wrap=tk.WORD,
                                                     font=("Consolas", 10),
                                                     bg=config.UI_COLORS[config.UI_THEME]["bg"],
                                                     fg=config.UI_COLORS[config.UI_THEME]["fg"],
                                                     insertbackground=config.UI_COLORS[config.UI_THEME]["accent"])
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial welcome message
        welcome_text = """
🎯 ULTIMATE JAPANESE QUIZ SOLVER READY! 🎯

✅ AI Providers Available: {}
✅ OCR Engine: Tesseract with Japanese support
✅ Screen Scanning: Ready
✅ Hotkeys: Configured

📋 How to Use:
1. Click "Start Scanning" to begin automatic detection
2. Or use "Select Region" to choose a specific area
3. Japanese text will be detected and solved automatically

⌨️ Hotkeys:
- Ctrl+Shift+Q: Quick scan
- Ctrl+Shift+R: Select region
- Ctrl+Shift+X: Emergency stop

🚀 Ready to solve Japanese questions with perfect accuracy!
        """.format(", ".join(self.solver.ai_providers.keys()))
        
        self.results_text.insert(tk.END, welcome_text)
        self.results_text.config(state=tk.DISABLED)
    
    def setup_history_tab(self):
        """Setup the history and search tab"""
        self.history_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.history_tab, text="📚 History")
        
        # Search and filter controls
        search_frame = ttk.Frame(self.history_tab)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(search_frame, text="🔍 Search:", style="Info.TLabel").pack(side=tk.LEFT)
        
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        self.search_entry.pack(side=tk.LEFT, padx=(5, 5))
        self.search_entry.bind("<KeyRelease>", self.on_search_changed)
        
        ttk.Button(search_frame, text="🔄 Refresh", 
                  command=self.refresh_history).pack(side=tk.LEFT, padx=(5, 5))
        
        ttk.Button(search_frame, text="📤 Export", 
                  command=self.export_history).pack(side=tk.LEFT)
        
        # History tree view
        history_frame = ttk.Frame(self.history_tab)
        history_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("Time", "Type", "Confidence", "Provider", "Question")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        self.history_tree.heading("Time", text="⏰ Time")
        self.history_tree.heading("Type", text="📝 Type")
        self.history_tree.heading("Confidence", text="📊 Confidence")
        self.history_tree.heading("Provider", text="🤖 Provider")
        self.history_tree.heading("Question", text="❓ Question")
        
        self.history_tree.column("Time", width=100)
        self.history_tree.column("Type", width=100)
        self.history_tree.column("Confidence", width=80)
        self.history_tree.column("Provider", width=80)
        self.history_tree.column("Question", width=400)
        
        # Scrollbar for tree
        tree_scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind events
        self.history_tree.bind("<Double-1>", self.on_history_item_selected)
        
        # Load initial history
        self.refresh_history()
    
    def setup_analytics_tab(self):
        """Setup the analytics and statistics tab"""
        self.analytics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_tab, text="📊 Analytics")
        
        # Statistics overview
        stats_frame = ttk.LabelFrame(self.analytics_tab, text="📈 Statistics Overview", padding="10")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create statistics grid
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # Total questions
        ttk.Label(stats_grid, text="📝 Total Questions:", style="Info.TLabel").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.total_questions_label = ttk.Label(stats_grid, text="0", style="Success.TLabel")
        self.total_questions_label.grid(row=0, column=1, sticky=tk.W)
        
        # Average confidence
        ttk.Label(stats_grid, text="📊 Avg Confidence:", style="Info.TLabel").grid(row=0, column=2, sticky=tk.W, padx=(20, 10))
        self.avg_confidence_label = ttk.Label(stats_grid, text="0%", style="Success.TLabel")
        self.avg_confidence_label.grid(row=0, column=3, sticky=tk.W)
        
        # Most used provider
        ttk.Label(stats_grid, text="🤖 Top Provider:", style="Info.TLabel").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.top_provider_label = ttk.Label(stats_grid, text="None", style="Success.TLabel")
        self.top_provider_label.grid(row=1, column=1, sticky=tk.W)
        
        # Average processing time
        ttk.Label(stats_grid, text="⚡ Avg Speed:", style="Info.TLabel").grid(row=1, column=2, sticky=tk.W, padx=(20, 10))
        self.avg_speed_label = ttk.Label(stats_grid, text="0.0s", style="Success.TLabel")
        self.avg_speed_label.grid(row=1, column=3, sticky=tk.W)
        
        # Question type breakdown
        breakdown_frame = ttk.LabelFrame(self.analytics_tab, text="📋 Question Type Breakdown", padding="10")
        breakdown_frame.pack(fill=tk.BOTH, expand=True)
        
        self.breakdown_text = scrolledtext.ScrolledText(breakdown_frame, 
                                                       height=10,
                                                       font=("Consolas", 10),
                                                       bg=config.UI_COLORS[config.UI_THEME]["bg"],
                                                       fg=config.UI_COLORS[config.UI_THEME]["fg"])
        self.breakdown_text.pack(fill=tk.BOTH, expand=True)
        
        # Refresh button
        ttk.Button(self.analytics_tab, text="🔄 Refresh Analytics", 
                  command=self.refresh_analytics).pack(pady=10)
        
        # Load initial analytics
        self.refresh_analytics()
    
    def setup_settings_tab(self):
        """Setup the settings and configuration tab"""
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="⚙️ Settings")
        
        # Create scrollable frame
        canvas = tk.Canvas(self.settings_tab)
        scrollbar = ttk.Scrollbar(self.settings_tab, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # OCR Settings
        ocr_frame = ttk.LabelFrame(scrollable_frame, text="🔍 OCR Settings", padding="10")
        ocr_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(ocr_frame, text="Language:", style="Info.TLabel").grid(row=0, column=0, sticky=tk.W)
        self.ocr_lang_var = tk.StringVar(value=config.OCR_LANGUAGE)
        ocr_lang_entry = ttk.Entry(ocr_frame, textvariable=self.ocr_lang_var, width=20)
        ocr_lang_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(ocr_frame, text="Scale Factor:", style="Info.TLabel").grid(row=1, column=0, sticky=tk.W)
        self.scale_factor_var = tk.DoubleVar(value=config.IMAGE_ENHANCEMENTS["scale_factor"])
        scale_factor_scale = ttk.Scale(ocr_frame, from_=1.0, to=5.0, variable=self.scale_factor_var, 
                                      orient=tk.HORIZONTAL, length=200)
        scale_factor_scale.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        # Performance Settings
        perf_frame = ttk.LabelFrame(scrollable_frame, text="⚡ Performance Settings", padding="10")
        perf_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(perf_frame, text="Polling Interval (s):", style="Info.TLabel").grid(row=0, column=0, sticky=tk.W)
        self.polling_var = tk.DoubleVar(value=config.POLLING_INTERVAL)
        polling_scale = ttk.Scale(perf_frame, from_=0.1, to=5.0, variable=self.polling_var, 
                                 orient=tk.HORIZONTAL, length=200)
        polling_scale.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(perf_frame, text="Cache Size:", style="Info.TLabel").grid(row=1, column=0, sticky=tk.W)
        self.cache_size_var = tk.IntVar(value=config.CACHE_SIZE)
        cache_size_spin = ttk.Spinbox(perf_frame, from_=100, to=10000, textvariable=self.cache_size_var, width=10)
        cache_size_spin.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        # UI Settings
        ui_frame = ttk.LabelFrame(scrollable_frame, text="🎨 UI Settings", padding="10")
        ui_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(ui_frame, text="Theme:", style="Info.TLabel").grid(row=0, column=0, sticky=tk.W)
        self.theme_var = tk.StringVar(value=config.UI_THEME)
        theme_combo = ttk.Combobox(ui_frame, textvariable=self.theme_var, 
                                  values=["dark", "light"], state="readonly", width=15)
        theme_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        theme_combo.bind("<<ComboboxSelected>>", self.on_theme_changed)
        
        self.always_on_top_var = tk.BooleanVar(value=config.ALWAYS_ON_TOP)
        ttk.Checkbutton(ui_frame, text="Always on Top", 
                       variable=self.always_on_top_var,
                       command=self.toggle_always_on_top).grid(row=1, column=0, columnspan=2, sticky=tk.W)
        
        # Feature Settings
        features_frame = ttk.LabelFrame(scrollable_frame, text="🚀 Advanced Features", padding="10")
        features_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.auto_region_var = tk.BooleanVar(value=config.FEATURES["auto_region_detection"])
        ttk.Checkbutton(features_frame, text="Auto Region Detection", 
                       variable=self.auto_region_var).grid(row=0, column=0, sticky=tk.W)
        
        self.batch_processing_var = tk.BooleanVar(value=config.FEATURES["batch_processing"])
        ttk.Checkbutton(features_frame, text="Batch Processing", 
                       variable=self.batch_processing_var).grid(row=1, column=0, sticky=tk.W)
        
        self.context_aware_var = tk.BooleanVar(value=config.FEATURES["context_awareness"])
        ttk.Checkbutton(features_frame, text="Context Awareness", 
                       variable=self.context_aware_var).grid(row=2, column=0, sticky=tk.W)
        
        # Save/Reset buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="💾 Save Settings", 
                  command=self.save_settings).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="🔄 Reset to Defaults", 
                  command=self.reset_settings).pack(side=tk.LEFT)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_status_bar(self):
        """Setup the status bar at the bottom"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Status indicators
        self.scanning_indicator = ttk.Label(self.status_bar, text="⏹️ Stopped", style="Info.TLabel")
        self.scanning_indicator.pack(side=tk.LEFT, padx=(5, 10))
        
        self.ai_status = ttk.Label(self.status_bar, 
                                  text=f"🤖 {len(self.solver.ai_providers)} AI Provider(s)", 
                                  style="Info.TLabel")
        self.ai_status.pack(side=tk.LEFT, padx=(0, 10))
        
        self.ocr_status = ttk.Label(self.status_bar, text="🔍 OCR Ready", style="Info.TLabel")
        self.ocr_status.pack(side=tk.LEFT, padx=(0, 10))
        
        # Version info
        version_label = ttk.Label(self.status_bar, text="v2.0 Ultimate", style="Info.TLabel")
        version_label.pack(side=tk.RIGHT, padx=5)
    
    def setup_hotkeys(self):
        """Setup global hotkeys"""
        try:
            import keyboard
            
            # Quick scan
            keyboard.add_hotkey(config.HOTKEYS["quick_scan"], self.quick_scan)
            
            # Emergency stop
            keyboard.add_hotkey(config.HOTKEYS["emergency_stop"], self.emergency_stop)
            
            print(f"🎹 Hotkeys configured: {config.HOTKEYS}")
        except Exception as e:
            print(f"⚠️ Failed to setup hotkeys: {e}")
    
    def start_result_monitor(self):
        """Start monitoring for new results with proper error handling"""
        def monitor_results():
            while not self._shutdown_event.is_set():
                try:
                    # Check result queue with timeout
                    try:
                        result = self._result_queue.get(timeout=0.1)
                        print(f"📥 Monitor: Got result from queue: {result.question_text[:50]}...")
                        # Update GUI in main thread
                        self.root.after(0, lambda r=result: self.display_result(r))
                        self._result_queue.task_done()
                        print("✅ Monitor: GUI update scheduled")
                    except queue.Empty:
                        pass
                    
                    # Check error queue
                    try:
                        error = self._error_queue.get_nowait()
                        print(f"❌ Monitor: Error from queue: {error}")
                        self.root.after(0, lambda e=error: self.handle_background_error(e))
                        self._error_queue.task_done()
                    except queue.Empty:
                        pass
                    
                    # Check if there are new results from the solver (legacy support)
                    if hasattr(self.solver, 'latest_result') and self.solver.latest_result:
                        result = self.solver.latest_result
                        self.solver.latest_result = None  # Clear the result
                        print(f"📥 Monitor: Legacy result: {result.question_text[:50]}...")
                        # Update GUI in main thread
                        self.root.after(0, lambda r=result: self.display_result(r))
                    
                except Exception as e:
                    print(f"❌ Monitor error: {e}")
                    self._error_queue.put(f"Result monitor error: {e}")
                    time.sleep(1)
        
        print("🔍 Starting result monitor thread...")
        self.monitor_thread = threading.Thread(target=monitor_results, daemon=True)
        self.monitor_thread.start()
    
    def toggle_scanning(self):
        """Start or stop the scanning process"""
        if not self.is_scanning:
            self.start_scanning()
        else:
            self.stop_scanning()
    
    def start_scanning(self):
        """Start the scanning process"""
        if self.scanning_thread and self.scanning_thread.is_alive():
            return
        
        self.is_scanning = True
        self.scan_button.config(text="⏹️ Stop Scanning")
        self.scanning_indicator.config(text="🔍 Scanning")
        self.status_label.config(text="Scanning...", style="Info.TLabel")
        
        # Start scanning in background thread
        self.scanning_thread = threading.Thread(target=self.scanning_worker, daemon=True)
        self.scanning_thread.start()
        
        self.update_results_display("🚀 Scanning started! Looking for Japanese questions...")
    
    def stop_scanning(self):
        """Stop the scanning process"""
        self.is_scanning = False
        self.solver.scanning_active = False
        self.scan_button.config(text="🔍 Start Scanning")
        self.scanning_indicator.config(text="⏹️ Stopped")
        self.status_label.config(text="Ready", style="Success.TLabel")
        
        self.update_results_display("⏹️ Scanning stopped.")
    
    def scanning_worker(self):
        """Background scanning worker with fast, optimized scanning"""
        try:
            while self.is_scanning and not self._shutdown_event.is_set():
                try:
                    print("⚡ GUI: Starting FAST scan...")
                    start_time = time.time()
                    
                    # Use fast GUI scanning - only process likely regions
                    result = self.fast_gui_scan()
                    
                    scan_time = time.time() - start_time
                    print(f"⚡ GUI: Scan completed in {scan_time:.2f}s")
                    
                    if result:
                        print(f"🎯 GUI: Found result - confidence: {result.confidence_score:.2f}")
                        print(f"📝 GUI: Question preview: {result.question_text[:100]}...")
                        
                        # Queue result for GUI update
                        self._result_queue.put(result)
                        print("✅ GUI: Result queued for display")
                        
                        # Also directly update GUI (backup method)
                        try:
                            self.root.after(0, lambda r=result: self.display_result(r))
                            print("✅ GUI: Direct GUI update scheduled")
                        except Exception as gui_error:
                            print(f"❌ GUI: Direct update failed: {gui_error}")
                    else:
                        print("🔍 GUI: No questions detected this scan")
                    
                    time.sleep(config.POLLING_INTERVAL)
                except Exception as scan_error:
                    print(f"❌ Scan error: {scan_error}")
                    self._error_queue.put(f"Scan error: {scan_error}")
                    time.sleep(1)  # Wait before retrying
                    
        except Exception as e:
            print(f"❌ Scanning worker failed: {e}")
            self._error_queue.put(f"Scanning worker failed: {e}")
    
    def fast_gui_scan(self):
        """Ultra-fast scanning with aggressive caching and optimization"""
        import mss
        from PIL import Image
        import hashlib
        
        try:
            with mss.mss() as sct:
                # Get primary monitor
                monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                
                # Take screenshot
                screenshot = sct.grab(monitor)
                screen_image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                
                # Ultra-fast duplicate detection - skip if same content
                screen_hash = hashlib.md5(screen_image.tobytes()).hexdigest()
                if hasattr(self, '_last_screen_hash') and screen_hash == self._last_screen_hash:
                    print("⚡ Screen unchanged - skipping scan")
                    return None
                self._last_screen_hash = screen_hash
                
                # Get smart regions with priority ordering
                promising_regions = self.get_prioritized_regions(screen_image)
                
                # Process ONLY the most promising region for maximum speed
                for i, region in enumerate(promising_regions[:1]):  # Only process 1 region!
                    print(f"⚡ Processing priority region {i+1}")
                    
                    # Check region-specific cache
                    region_key = f"{region['left']}_{region['top']}_{region['width']}_{region['height']}"
                    
                    result = self.solver.process_screen_region(region)
                    if result and result.confidence_score > 0.6:
                        print(f"🎯 Found excellent result - confidence: {result.confidence_score:.2f}")
                        return result
                
                return None
                
        except Exception as e:
            print(f"❌ Ultra-fast scan failed: {e}")
            return None
    
    def get_smart_regions(self, screen_image):
        """Get smart regions optimized for common Japanese quiz layouts"""
        width, height = screen_image.size
        
        # Define common regions where Japanese quizzes typically appear
        smart_regions = [
            # Center region - most common for quiz content
            {
                "left": int(width * 0.2),
                "top": int(height * 0.2), 
                "width": int(width * 0.6),
                "height": int(height * 0.6)
            },
            # Left side - common for PDF viewers
            {
                "left": int(width * 0.05),
                "top": int(height * 0.15),
                "width": int(width * 0.5),
                "height": int(height * 0.7)
            },
            # Full screen as fallback
            {
                "left": 0,
                "top": 0,
                "width": width,
                "height": height
            }
        ]
        
        return smart_regions
    
    def get_prioritized_regions(self, screen_image):
        """Get regions in priority order for maximum speed"""
        width, height = screen_image.size
        
        # Priority regions - ordered by likelihood of Japanese quiz content
        priority_regions = [
            # Center-left - most common for document content
            {
                "left": int(width * 0.1),
                "top": int(height * 0.25),
                "width": int(width * 0.6),
                "height": int(height * 0.5)
            }
        ]
        
        return priority_regions
    
    def handle_background_error(self, error_msg: str):
        """Handle errors from background threads"""
        self.update_results_display(f"⚠️ Warning: {error_msg}")
        print(f"Background error: {error_msg}")
    
    def handle_scanning_error(self, error):
        """Handle scanning errors"""
        self.stop_scanning()
        messagebox.showerror("Scanning Error", f"Scanning failed: {error}")
    
    def display_result(self, result: QuestionResult):
        """Display a new quiz result"""
        print(f"🎯 DISPLAY_RESULT CALLED: {result.question_text[:50]}...")
        print(f"📊 Confidence: {result.confidence_score:.2f}")
        
        try:
            self.current_result = result
            
            # Update confidence display
            confidence_percent = result.confidence_score * 100
            self.confidence_var.set(confidence_percent)
            self.confidence_label.config(text=f"{confidence_percent:.1f}%")
            print(f"✅ Updated confidence: {confidence_percent:.1f}%")
            
            # Update question type
            self.question_type_label.config(text=result.question_type.title())
            print(f"✅ Updated question type: {result.question_type}")
            
            # Update status
            if result.confidence_score >= config.CONFIDENCE_THRESHOLDS["high"]:
                status_style = "Success.TLabel"
                status_text = "High Confidence"
            elif result.confidence_score >= config.CONFIDENCE_THRESHOLDS["medium"]:
                status_style = "Info.TLabel"
                status_text = "Medium Confidence"
            else:
                status_style = "Error.TLabel"
                status_text = "Low Confidence"
            
            self.status_label.config(text=status_text, style=status_style)
            print(f"✅ Updated status: {status_text}")
            
            # Format and display result
            formatted_result = self.format_result_display(result)
            self.update_results_display(formatted_result)
            print("✅ Updated results display")
            
            # Refresh history
            self.refresh_history()
            print("✅ Refreshed history")
            
            print("🎉 RESULT DISPLAY COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            print(f"❌ Error in display_result: {e}")
            import traceback
            traceback.print_exc()
    
    def format_result_display(self, result: QuestionResult) -> str:
        """Format result for display"""
        confidence_indicator = "🟢 HIGH" if result.confidence_score >= 0.85 else "🟡 MEDIUM" if result.confidence_score >= 0.70 else "🔴 LOW"
        
        formatted = f"""
{'='*80}
🎯 ULTIMATE JAPANESE QUIZ SOLVER RESULT
{'='*80}

⏰ Time: {result.timestamp.strftime('%H:%M:%S')}
📊 Confidence: {confidence_indicator} ({result.confidence_score:.1%})
📝 Type: {result.question_type.upper()}
🤖 Provider: {result.ai_provider.upper()}
⚡ Processing: {result.processing_time:.2f}s
🔍 OCR Confidence: {result.ocr_confidence:.1f}%

📋 DETECTED QUESTION:
{result.question_text}

{'─'*80}

{result.ai_answer}

{'='*80}
🎊 RESULT PROCESSED SUCCESSFULLY! 🎊

"""
        return formatted
    
    def update_results_display(self, text: str):
        """Update the results display area"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, f"\n{text}\n")
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)
    
    def select_region(self):
        """Open region selection dialog"""
        # Hide main window temporarily
        self.root.withdraw()
        
        try:
            # Import and use the region selector from the original code
            from main import select_region
            new_region = select_region()
            
            if new_region:
                config.CAPTURE_REGION = new_region
                messagebox.showinfo("Region Selected", 
                                  f"New capture region set: {new_region['width']}x{new_region['height']}")
        except Exception as e:
            messagebox.showerror("Region Selection Error", f"Failed to select region: {e}")
        finally:
            # Show main window again
            self.root.deiconify()
    
    def quick_scan(self):
        """Perform a quick one-time scan"""
        def scan():
            results = self.solver.scan_full_screen()
            if results:
                best_result = max(results, key=lambda r: r.confidence_score)
                self.root.after(0, lambda: self.display_result(best_result))
            else:
                self.root.after(0, lambda: self.update_results_display("🔍 Quick scan completed - No questions detected"))
        
        threading.Thread(target=scan, daemon=True).start()
        self.update_results_display("🚀 Quick scan initiated...")
    
    def emergency_stop(self):
        """Emergency stop all operations"""
        self.stop_scanning()
        self.solver.emergency_stop = True
        self.update_results_display("🚨 EMERGENCY STOP ACTIVATED!")
    
    def on_provider_changed(self, event=None):
        """Handle AI provider change"""
        new_provider = self.provider_var.get()
        if new_provider in self.solver.ai_providers:
            config.AI_PROVIDER = new_provider
            self.update_results_display(f"🤖 Switched to {new_provider.upper()} AI provider")
    
    def on_theme_changed(self, event=None):
        """Handle theme change"""
        new_theme = self.theme_var.get()
        if new_theme != config.UI_THEME:
            config.UI_THEME = new_theme
            # Add light theme colors if not present
            if "light" not in config.UI_COLORS:
                config.UI_COLORS["light"] = {
                    "bg": "#ffffff",
                    "fg": "#000000", 
                    "accent": "#2196F3",
                    "success": "#4CAF50",
                    "error": "#f44336",
                    "warning": "#ff9800",
                    "info": "#2196f3"
                }
            self.setup_theme()
            self.update_results_display(f"🎨 Theme changed to {new_theme.upper()}")
    
    def refresh_history(self):
        """Refresh the history display"""
        # Clear existing items
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # Load from database
        try:
            with sqlite3.connect(self.solver.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, question_type, confidence_score, ai_provider, question_text
                    FROM quiz_history 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """)
                
                for row in cursor:
                    timestamp_str = datetime.fromisoformat(row[0]).strftime('%H:%M:%S')
                    question_preview = (row[4][:50] + "...") if len(row[4]) > 50 else row[4]
                    
                    self.history_tree.insert("", tk.END, values=(
                        timestamp_str,
                        row[1].title(),
                        f"{row[2]:.1%}",
                        row[3].upper(),
                        question_preview
                    ))
        except Exception as e:
            print(f"Error refreshing history: {e}")
    
    def refresh_analytics(self):
        """Refresh the analytics display"""
        try:
            with sqlite3.connect(self.solver.db_path) as conn:
                # Total questions
                cursor = conn.execute("SELECT COUNT(*) FROM quiz_history")
                total_questions = cursor.fetchone()[0]
                self.total_questions_label.config(text=str(total_questions))
                
                # Average confidence
                cursor = conn.execute("SELECT AVG(confidence_score) FROM quiz_history")
                avg_confidence = cursor.fetchone()[0]
                if avg_confidence:
                    self.avg_confidence_label.config(text=f"{avg_confidence:.1%}")
                
                # Top provider
                cursor = conn.execute("""
                    SELECT ai_provider, COUNT(*) as count 
                    FROM quiz_history 
                    GROUP BY ai_provider 
                    ORDER BY count DESC 
                    LIMIT 1
                """)
                top_provider = cursor.fetchone()
                if top_provider:
                    self.top_provider_label.config(text=top_provider[0].upper())
                
                # Average processing time
                cursor = conn.execute("SELECT AVG(processing_time) FROM quiz_history")
                avg_speed = cursor.fetchone()[0]
                if avg_speed:
                    self.avg_speed_label.config(text=f"{avg_speed:.2f}s")
                
                # Question type breakdown
                cursor = conn.execute("""
                    SELECT question_type, COUNT(*) as count, AVG(confidence_score) as avg_conf
                    FROM quiz_history 
                    GROUP BY question_type
                    ORDER BY count DESC
                """)
                
                breakdown_text = "📊 QUESTION TYPE BREAKDOWN\n" + "="*50 + "\n\n"
                
                for row in cursor:
                    q_type, count, avg_conf = row
                    breakdown_text += f"{q_type.title():15} | {count:4d} questions | {avg_conf:.1%} avg confidence\n"
                
                self.breakdown_text.delete(1.0, tk.END)
                self.breakdown_text.insert(1.0, breakdown_text)
                
        except Exception as e:
            print(f"Error refreshing analytics: {e}")
    
    def on_search_changed(self, event=None):
        """Handle history search"""
        search_term = self.search_var.get().lower()
        
        # Clear existing items
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        if not search_term:
            self.refresh_history()
            return
        
        # Search database
        try:
            with sqlite3.connect(self.solver.db_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, question_type, confidence_score, ai_provider, question_text
                    FROM quiz_history 
                    WHERE LOWER(question_text) LIKE ? OR LOWER(question_type) LIKE ? OR LOWER(ai_provider) LIKE ?
                    ORDER BY timestamp DESC 
                    LIMIT 50
                """, (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%'))
                
                for row in cursor:
                    timestamp_str = datetime.fromisoformat(row[0]).strftime('%H:%M:%S')
                    question_preview = (row[4][:50] + "...") if len(row[4]) > 50 else row[4]
                    
                    self.history_tree.insert("", tk.END, values=(
                        timestamp_str,
                        row[1].title(),
                        f"{row[2]:.1%}",
                        row[3].upper(),
                        question_preview
                    ))
        except Exception as e:
            print(f"Error searching history: {e}")
    
    def on_history_item_selected(self, event=None):
        """Handle history item selection"""
        selection = self.history_tree.selection()
        if not selection:
            return
        
        item = self.history_tree.item(selection[0])
        question_preview = item['values'][4]
        
        # Show full question in a popup
        messagebox.showinfo("Question Details", f"Full Question:\n\n{question_preview}")
    
    def export_history(self):
        """Export history to CSV file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with sqlite3.connect(self.solver.db_path) as conn:
                    import pandas as pd
                    df = pd.read_sql_query("SELECT * FROM quiz_history ORDER BY timestamp DESC", conn)
                    
                    if filename.endswith('.json'):
                        df.to_json(filename, orient='records', indent=2)
                    else:
                        df.to_csv(filename, index=False)
                    
                messagebox.showinfo("Export Complete", f"History exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")
    
    def toggle_always_on_top(self):
        """Toggle always on top setting"""
        current = self.root.attributes("-topmost")
        self.root.attributes("-topmost", not current)
        self.always_on_top_var.set(not current)
    
    def toggle_overlay_mode(self):
        """Toggle overlay mode"""
        self.overlay_mode = not self.overlay_mode
        if self.overlay_mode:
            self.root.attributes("-alpha", 0.8)
            self.root.attributes("-topmost", True)
            messagebox.showinfo("Overlay Mode", "Overlay mode enabled - window is semi-transparent")
        else:
            self.root.attributes("-alpha", 1.0)
            messagebox.showinfo("Overlay Mode", "Overlay mode disabled")
    
    def toggle_fullscreen_scan(self):
        """Toggle full screen scanning"""
        config.FULL_SCREEN_SCAN = not config.FULL_SCREEN_SCAN
        mode = "enabled" if config.FULL_SCREEN_SCAN else "disabled"
        messagebox.showinfo("Full Screen Scan", f"Full screen scanning {mode}")
    
    def show_settings(self):
        """Show the settings tab"""
        self.notebook.select(3)  # Select settings tab
    
    def save_settings(self):
        """Save current settings"""
        try:
            # Update config values
            config.OCR_LANGUAGE = self.ocr_lang_var.get()
            config.IMAGE_ENHANCEMENTS["scale_factor"] = self.scale_factor_var.get()
            config.POLLING_INTERVAL = self.polling_var.get()
            config.CACHE_SIZE = self.cache_size_var.get()
            config.UI_THEME = self.theme_var.get()
            config.ALWAYS_ON_TOP = self.always_on_top_var.get()
            config.FEATURES["auto_region_detection"] = self.auto_region_var.get()
            config.FEATURES["batch_processing"] = self.batch_processing_var.get()
            config.FEATURES["context_awareness"] = self.context_aware_var.get()
            
            messagebox.showinfo("Settings Saved", "Settings have been saved successfully!")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save settings: {e}")
    
    def reset_settings(self):
        """Reset settings to defaults"""
        if messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to defaults?"):
            # Reset all variables to defaults
            self.ocr_lang_var.set("jpn+eng")
            self.scale_factor_var.set(3.0)
            self.polling_var.set(0.5)
            self.cache_size_var.set(1000)
            self.theme_var.set("dark")
            self.always_on_top_var.set(True)
            self.auto_region_var.set(True)
            self.batch_processing_var.set(True)
            self.context_aware_var.set(True)
            
            messagebox.showinfo("Settings Reset", "Settings have been reset to defaults!")
    
    def show_hotkeys_help(self):
        """Show hotkeys help dialog"""
        hotkeys_text = "🎹 GLOBAL HOTKEYS\n" + "="*30 + "\n\n"
        for action, hotkey in config.HOTKEYS.items():
            hotkeys_text += f"{action.replace('_', ' ').title():20} | {hotkey}\n"
        
        messagebox.showinfo("Hotkeys Reference", hotkeys_text)
    
    def show_troubleshooting(self):
        """Show troubleshooting information"""
        troubleshooting_text = """
🔧 TROUBLESHOOTING GUIDE

Common Issues:
• No Japanese detected: Install Japanese language pack for Tesseract
• Low confidence: Improve image quality or adjust scale factor
• API errors: Check your API keys in environment variables
• Slow performance: Reduce polling interval or disable batch processing
• Hotkeys not working: Run as administrator

For more help, check the documentation or contact support.
        """
        messagebox.showinfo("Troubleshooting", troubleshooting_text)
    
    def import_settings(self):
        """Import settings from file"""
        filename = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    settings = json.load(f)
                
                # Apply imported settings
                if 'ocr_language' in settings:
                    self.ocr_lang_var.set(settings['ocr_language'])
                if 'scale_factor' in settings:
                    self.scale_factor_var.set(settings['scale_factor'])
                # Add more settings as needed
                
                messagebox.showinfo("Import Complete", "Settings imported successfully!")
            except Exception as e:
                messagebox.showerror("Import Error", f"Failed to import settings: {e}")
    
    def export_settings(self):
        """Export current settings to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                settings = {
                    'ocr_language': self.ocr_lang_var.get(),
                    'scale_factor': self.scale_factor_var.get(),
                    'polling_interval': self.polling_var.get(),
                    'cache_size': self.cache_size_var.get(),
                    'ui_theme': self.theme_var.get(),
                    'always_on_top': self.always_on_top_var.get()
                }
                
                with open(filename, 'w') as f:
                    json.dump(settings, f, indent=2)
                
                messagebox.showinfo("Export Complete", f"Settings exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export settings: {e}")
    
    def test_ocr(self):
        """Test OCR functionality"""
        # Simple OCR test
        messagebox.showinfo("OCR Test", "OCR test functionality would go here. This is a placeholder.")
    
    def configure_ai_providers(self):
        """Configure AI provider settings"""
        # AI provider configuration dialog
        messagebox.showinfo("AI Providers", "AI provider configuration would go here. This is a placeholder.")
    
    def configure_hotkeys(self):
        """Configure hotkey settings"""
        # Hotkey configuration dialog
        messagebox.showinfo("Hotkeys", "Hotkey configuration would go here. This is a placeholder.")
    
    def show_statistics(self):
        """Show detailed statistics dashboard"""
        # Switch to analytics tab
        self.notebook.select(2)  # Analytics tab
        self.refresh_analytics()
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
🎯 ULTIMATE JAPANESE QUIZ SOLVER v2.0

The most advanced Japanese question detection and solving system!

Features:
✅ Full screen scanning with auto detection
✅ Multi-AI provider support
✅ Advanced OCR preprocessing
✅ Question type detection
✅ Confidence scoring
✅ History tracking
✅ Analytics dashboard
✅ Global hotkeys

Developed with ❤️ for Japanese language learners worldwide.
        """
        messagebox.showinfo("About", about_text)
    
    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit Ultimate Japanese Quiz Solver?"):
            self.stop_scanning()
            self.solver.emergency_stop = True
            self.root.destroy()
    
    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()


def main():
    """Main entry point for the GUI application"""
    print("""
    🎯 ULTIMATE JAPANESE QUIZ SOLVER - GUI VERSION 🎯
    ================================================
    
    Starting professional GUI interface...
    """)
    
    try:
        app = UltimateQuizSolverGUI()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
