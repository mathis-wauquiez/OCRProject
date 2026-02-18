import os
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Union, Optional, Any, List, Dict, Callable, Protocol, runtime_checkable
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
import logging
from enum import Enum

# Try to import PDF generation libraries
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ReportError(Exception):
    """Base exception for report-related errors"""
    pass


class ReportGenerationError(ReportError):
    """Exception raised when report generation fails"""
    pass


class MissingDependencyError(ReportError):
    """Exception raised when required dependency is missing"""
    pass


class ValidationError(ReportError):
    """Exception raised when validation fails"""
    pass


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with custom formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger()


# ============================================================================
# CONFIGURATION
# ============================================================================

class Theme(Enum):
    """Available themes for reports"""
    DEFAULT = "default"
    DARK = "dark"
    MINIMAL = "minimal"
    PROFESSIONAL = "professional"


@dataclass
class ReportConfig:
    """Configuration for report generation"""
    # Page settings
    page_size: tuple = (8.5, 11)  # inches
    dpi: int = 300
    
    # Image settings
    max_image_size: tuple = (1920, 1080)
    image_quality: int = 95
    compress_images: bool = True
    
    # Table settings
    max_table_rows: int = 50
    table_style: Optional[Dict] = None
    
    # PDF settings
    include_toc: bool = True
    embed_fonts: bool = True
    
    # HTML settings
    include_katex: bool = True
    theme: Theme = Theme.DEFAULT
    custom_css: Optional[str] = None
    
    # General settings
    show_progress: bool = True
    auto_close_figures: bool = True
    
    # Output settings
    output_format: str = 'png'  # for embedded figures
    
    def __post_init__(self):
        """Validate configuration"""
        if self.dpi < 72 or self.dpi > 600:
            logger.warning(f"DPI {self.dpi} is outside recommended range (72-600)")
        
        if self.image_quality < 1 or self.image_quality > 100:
            raise ValueError("image_quality must be between 1 and 100")
        
        if isinstance(self.theme, str):
            self.theme = Theme(self.theme)


# ============================================================================
# PROTOCOLS AND DATA CLASSES
# ============================================================================

@runtime_checkable
class Reportable(Protocol):
    """Protocol for objects that can be reported"""
    def to_report_item(self) -> 'ReportItem':
        """Convert object to ReportItem"""
        ...


@dataclass
class ReportItem:
    """Data class for report items with validation"""
    content_type: str
    content: Any
    title: Optional[str] = None
    width: Optional[float] = None
    height: Optional[float] = None
    metadata: Optional[Dict] = field(default_factory=dict)

    def __post_init__(self):
        """Validate report item"""
        valid_types = {'figure', 'text', 'image', 'table', 'raw_html', 'lazy'}
        if self.content_type not in valid_types:
            raise ValidationError(
                f"Invalid content_type '{self.content_type}'. "
                f"Must be one of: {valid_types}"
            )

        if self.metadata is None:
            self.metadata = {}


@dataclass
class ReportSection:
    """A named section that groups multiple report items under one collapsible header.

    Use via the ``AutoReport.section()`` context manager::

        with report.section("Results"):
            report.report_figure(fig)
            report.report_table(df)
    """
    title: str
    items: List[ReportItem] = field(default_factory=list)


@dataclass
class ReportMetadata:
    """Metadata for the report"""
    creation_date: datetime = field(default_factory=datetime.now)
    item_count: int = 0
    report_id: str = ""
    version: str = "2.0"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'creation_date': self.creation_date.isoformat(),
            'item_count': self.item_count,
            'report_id': self.report_id,
            'version': self.version
        }


# ============================================================================
# HTML TEMPLATES WITH COLLAPSIBLE SECTIONS
# ============================================================================

class HTMLTemplates:
    """HTML templates for different themes"""
    
    DEFAULT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
            background-color: #f9f9f9;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0 0 10px 0;
        }
        .metadata {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        /* Collapsible Section Styles - FIXED FOR LARGE CONTENT */
        .section {
            background: white;
            margin-bottom: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: visible;
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            cursor: pointer;
            user-select: none;
            transition: background 0.3s ease;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        .section-header:hover {
            background: linear-gradient(135deg, #7688f0 0%, #8555b2 100%);
        }
        
        .section-title {
            margin: 0;
            font-size: 1.3em;
            font-weight: 600;
        }
        
        .section-toggle {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .toggle-icon {
            width: 24px;
            height: 24px;
            border: 2px solid white;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            transition: transform 0.3s ease;
        }
        
        .section.collapsed .toggle-icon {
            transform: rotate(0deg);
        }
        
        .section.expanded .toggle-icon {
            transform: rotate(90deg);
        }
        
        /* FIXED: Use display instead of max-height for large sections */
        .section-content {
            display: none;
            padding: 0 25px;
            overflow: visible;
        }
        
        .section.expanded .section-content {
            display: block;
            padding: 25px;
        }
        
        .section-content-inner {
            /* Content can grow freely */
        }
        
        /* Optional: Add smooth fade-in animation */
        .section.expanded .section-content {
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        /* Expand/Collapse All Controls */
        .controls {
            background: white;
            padding: 15px 25px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .control-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .control-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .control-btn:active {
            transform: translateY(0);
        }
        
        /* Rest of the styles */
        .figure {
            text-align: center;
            margin: 20px 0;
        }
        .figure img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .table-container {
            overflow-x: auto;
            margin: 20px 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            background: white;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #e9ecef;
        }
        .text-content {
            color: #444;
            line-height: 1.8;
        }
        pre {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            border-left: 4px solid #667eea;
        }
        .katex-equation {
            text-align: center;
            margin: 20px 0;
            font-size: 1.2em;
        }
    </style>
    {% if include_katex %}
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
            onload="renderMathInElement(document.body);"></script>
    {% endif %}
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <div class="metadata">
            <p><strong>Generated:</strong> {{ creation_date }}</p>
            <p><strong>Author:</strong> {{ author }}</p>
            <p><strong>Items:</strong> {{ item_count }}</p>
        </div>
    </div>
    
    <div class="controls">
        <button class="control-btn" onclick="expandAll()">📂 Expand All</button>
        <button class="control-btn" onclick="collapseAll()">📁 Collapse All</button>
        <span style="margin-left: auto; color: #666; font-size: 0.9em;">
            Click section headers to toggle
        </span>
    </div>
    
    {% for section in sections %}
    <div class="section collapsed" id="section-{{ loop.index }}">
        <div class="section-header" onclick="toggleSection({{ loop.index }})">
            <h2 class="section-title">{{ section.title }}</h2>
            <div class="section-toggle">
                <span class="toggle-text">Click to expand</span>
                <div class="toggle-icon">▶</div>
            </div>
        </div>
        <div class="section-content">
            <div class="section-content-inner">
                {{ section.content | safe }}
            </div>
        </div>
    </div>
    {% endfor %}
    
    <script>
        // Toggle individual section
        function toggleSection(index) {
            const section = document.getElementById('section-' + index);
            const toggleText = section.querySelector('.toggle-text');
            
            section.classList.toggle('collapsed');
            section.classList.toggle('expanded');
            
            if (section.classList.contains('expanded')) {
                toggleText.textContent = 'Click to collapse';
            } else {
                toggleText.textContent = 'Click to expand';
            }
        }
        
        // Expand all sections
        function expandAll() {
            document.querySelectorAll('.section').forEach(section => {
                section.classList.remove('collapsed');
                section.classList.add('expanded');
                const toggleText = section.querySelector('.toggle-text');
                if (toggleText) toggleText.textContent = 'Click to collapse';
            });
        }
        
        // Collapse all sections
        function collapseAll() {
            document.querySelectorAll('.section').forEach(section => {
                section.classList.remove('expanded');
                section.classList.add('collapsed');
                const toggleText = section.querySelector('.toggle-text');
                if (toggleText) toggleText.textContent = 'Click to expand';
            });
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
                e.preventDefault();
                expandAll();
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'w') {
                e.preventDefault();
                collapseAll();
            }
        });
    </script>
</body>
</html>
"""    
    DARK_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        
        /* Collapsible Section Styles */
        .section {
            background: #2a2a2a;
            margin-bottom: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            overflow: visible;
            transition: all 0.3s ease;
        }
        
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 25px;
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            cursor: pointer;
            user-select: none;
            transition: background 0.3s ease;
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .section-header:hover {
            background: linear-gradient(135deg, #34495e 0%, #3d566e 100%);
        }
        
        .section-title {
            margin: 0;
            font-size: 1.3em;
            font-weight: 600;
            color: #4a9eff;
        }
        
        .section-toggle {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .toggle-icon {
            width: 24px;
            height: 24px;
            border: 2px solid #4a9eff;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            transition: transform 0.3s ease;
            color: #4a9eff;
        }
        
        .section.collapsed .toggle-icon {
            transform: rotate(0deg);
        }
        
        .section.expanded .toggle-icon {
            transform: rotate(90deg);
        }
        
        .section-content {
            display: none;
            padding: 0 25px;
        }

        .section.expanded .section-content {
            display: block;
            padding: 25px;
        }
        
        /* Controls */
        .controls {
            background: #2a2a2a;
            padding: 15px 25px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .control-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        
        .control-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.4);
            background: linear-gradient(135deg, #34495e 0%, #3d566e 100%);
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            background: #2a2a2a;
        }
        th {
            background-color: #34495e;
            color: white;
        }
        th, td {
            border: 1px solid #444;
            padding: 12px;
        }
        tr:nth-child(even) {
            background-color: #333;
        }
        tr:hover {
            background-color: #3a3a3a;
        }
        .figure {
            text-align: center;
            margin: 20px 0;
        }
        .figure img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <div class="metadata">
            <p><strong>Generated:</strong> {{ creation_date }}</p>
            <p><strong>Author:</strong> {{ author }}</p>
            <p><strong>Items:</strong> {{ item_count }}</p>
        </div>
    </div>
    
    <div class="controls">
        <button class="control-btn" onclick="expandAll()">📂 Expand All</button>
        <button class="control-btn" onclick="collapseAll()">📁 Collapse All</button>
        <span style="margin-left: auto; color: #888; font-size: 0.9em;">
            Click section headers to toggle
        </span>
    </div>
    
    {% for section in sections %}
    <div class="section collapsed" id="section-{{ loop.index }}">
        <div class="section-header" onclick="toggleSection({{ loop.index }})">
            <h2 class="section-title">{{ section.title }}</h2>
            <div class="section-toggle">
                <span class="toggle-text">Click to expand</span>
                <div class="toggle-icon">▶</div>
            </div>
        </div>
        <div class="section-content">
            <div class="section-content-inner">
                {{ section.content | safe }}
            </div>
        </div>
    </div>
    {% endfor %}
    
    <script>
        function toggleSection(index) {
            const section = document.getElementById('section-' + index);
            const toggleText = section.querySelector('.toggle-text');
            
            if (section.classList.contains('collapsed')) {
                section.classList.remove('collapsed');
                section.classList.add('expanded');
                toggleText.textContent = 'Click to collapse';
            } else {
                section.classList.remove('expanded');
                section.classList.add('collapsed');
                toggleText.textContent = 'Click to expand';
            }
        }
        
        function expandAll() {
            const sections = document.querySelectorAll('.section');
            sections.forEach(section => {
                section.classList.remove('collapsed');
                section.classList.add('expanded');
                const toggleText = section.querySelector('.toggle-text');
                if (toggleText) toggleText.textContent = 'Click to collapse';
            });
        }
        
        function collapseAll() {
            const sections = document.querySelectorAll('.section');
            sections.forEach(section => {
                section.classList.remove('expanded');
                section.classList.add('collapsed');
                const toggleText = section.querySelector('.toggle-text');
                if (toggleText) toggleText.textContent = 'Click to expand';
            });
        }
        
        document.addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
                e.preventDefault();
                expandAll();
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
                e.preventDefault();
                collapseAll();
            }
        });
    </script>
</body>
</html>
    """
    
    @classmethod
    def get_template(cls, theme: Theme) -> str:
        """Get template for specified theme"""
        if theme == Theme.DARK:
            return cls.DARK_TEMPLATE
        else:
            return cls.DEFAULT_TEMPLATE


# ============================================================================
# IMAGE UTILITIES
# ============================================================================

class ImageOptimizer:
    """Utilities for image optimization"""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.ImageOptimizer')
    
    def optimize_image(self, img: Image.Image) -> Image.Image:
        """Optimize image for report inclusion"""
        if not PIL_AVAILABLE:
            return img
        
        try:
            # Resize if too large
            if (img.size[0] > self.config.max_image_size[0] or 
                img.size[1] > self.config.max_image_size[1]):
                self.logger.debug(f"Resizing image from {img.size} to fit {self.config.max_image_size}")
                img.thumbnail(self.config.max_image_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'L', 'RGBA'):
                self.logger.debug(f"Converting image from {img.mode} to RGB")
                img = img.convert('RGB')
            
            return img
        except Exception as e:
            self.logger.error(f"Failed to optimize image: {e}")
            return img
    
    def image_to_base64(self, img: Image.Image, format: str = 'PNG') -> str:
        """Convert PIL Image to base64 string"""
        try:
            buf = io.BytesIO()
            
            # Optimize before saving
            img_optimized = self.optimize_image(img)
            
            # Save with quality settings
            if format.upper() in ('JPEG', 'JPG'):
                img_optimized.save(buf, format='JPEG', 
                                  quality=self.config.image_quality,
                                  optimize=self.config.compress_images)
            else:
                img_optimized.save(buf, format=format,
                                  optimize=self.config.compress_images)
            
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to convert image to base64: {e}")
            raise ReportGenerationError(f"Image conversion failed: {e}")


# ============================================================================
# MAIN AUTOREPORT CLASS
# ============================================================================

class AutoReport:
    """
    Enhanced automatic report generation with comprehensive features.
    
    Features:
    - Resource management with context managers
    - Comprehensive error handling and logging
    - Configuration system
    - Template-based HTML generation with collapsible sections
    - Progress tracking
    - Image optimization
    - Type validation
    """
    
    def __init__(self, 
                 title: str = "Auto Report",
                 author: str = "AutoReport System",
                 output_dir: Union[str, Path] = "./reports",
                 config: Optional[ReportConfig] = None,
                 log_level: int = logging.INFO):
        """
        Initialize the AutoReport class.
        
        Args:
            title: Title of the report
            author: Author of the report
            output_dir: Directory to save reports
            config: Report configuration object
            log_level: Logging level
        """
        self.title = title
        self.author = author
        self.output_dir = Path(output_dir)
        self.config = config or ReportConfig()
        self.items: List[ReportItem] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__ + '.AutoReport')
        self.logger.setLevel(log_level)
        
        # Create metadata
        self.metadata = ReportMetadata(
            report_id=datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        
        # Create output directory
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"AutoReport initialized. Report ID: {self.metadata.report_id}")
            self.logger.info(f"Output directory: {self.output_dir}")
        except Exception as e:
            raise ReportError(f"Failed to create output directory: {e}")
        
        # Initialize image optimizer
        self.image_optimizer = ImageOptimizer(self.config)

        # Track opened figures for cleanup
        self._figures_to_close: List[plt.Figure] = []

        # Cache rendered base64 strings keyed by figure id() to avoid
        # re-rendering the same figure on repeated generate_html() calls.
        self._figure_render_cache: Dict[int, str] = {}

        # Sections support: ordered content that may contain ReportSection or
        # standalone ReportItem entries.  The flat ``self.items`` list is kept
        # in sync for backward compatibility.
        self._content_order: List[Union[ReportItem, ReportSection]] = []
        self._current_section: Optional[ReportSection] = None
    
    # ========================================================================
    # CONTEXT MANAGERS
    # ========================================================================
    
    @contextmanager
    def auto_close_figures(self):
        """
        Context manager to automatically close matplotlib figures.
        
        Usage:
            with report.auto_close_figures():
                report.generate_pdf()
        """
        try:
            yield
        finally:
            if self.config.auto_close_figures:
                self.logger.debug(f"Closing {len(self._figures_to_close)} figures")
                for fig in self._figures_to_close:
                    try:
                        plt.close(fig)
                    except Exception as e:
                        self.logger.warning(f"Failed to close figure: {e}")
                self._figures_to_close.clear()
    
    @contextmanager
    def batch_add(self):
        """
        Context manager for batch adding items with single progress update.

        Usage:
            with report.batch_add():
                report.report(...)
                report.report(...)
        """
        initial_count = len(self.items)
        try:
            yield
        finally:
            added = len(self.items) - initial_count
            if added > 0:
                self.logger.info(f"Batch added {added} items to report")

    @contextmanager
    def section(self, title: str):
        """
        Context manager to group report items into a named section.

        All ``report_*`` calls made inside the block are collected under a
        single collapsible header in the HTML output.  Items added outside any
        section remain standalone (one collapsible header each), preserving
        full backward compatibility.

        Usage::

            with report.section("Analysis"):
                report.report_figure(fig, title="Distribution")
                report.report_table(df, title="Metrics")

            # This item is standalone
            report.report_text("Some note")

        Args:
            title: The section heading shown in the report.
        """
        if self._current_section is not None:
            raise ReportError(
                f"Cannot open section '{title}' – section "
                f"'{self._current_section.title}' is already open. "
                "Nested sections are not supported."
            )

        sec = ReportSection(title=title)
        self._content_order.append(sec)
        self._current_section = sec
        self.logger.info(f"Opened section: {title}")
        try:
            yield sec
        finally:
            self._current_section = None
            self.logger.info(
                f"Closed section: {title} ({len(sec.items)} items)"
            )

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _add_item(self, item: ReportItem):
        """Append *item* to the report, respecting the active section context.

        * Maintains the flat ``self.items`` list for backward compatibility.
        * Routes the item to the current ``ReportSection`` if one is open,
          otherwise appends it as a standalone entry in ``_content_order``.
        """
        self.items.append(item)
        self.metadata.item_count += 1
        if self._current_section is not None:
            self._current_section.items.append(item)
        else:
            self._content_order.append(item)

    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    def _validate_content(self, content: Any, content_type: str) -> bool:
        """Validate content matches expected type"""
        try:
            if content_type == 'figure' and MATPLOTLIB_AVAILABLE:
                return isinstance(content, plt.Figure)
            elif content_type == 'image' and PIL_AVAILABLE:
                return isinstance(content, (str, Image.Image))
            elif content_type == 'table':
                return isinstance(content, pd.DataFrame)
            elif content_type == 'text':
                return isinstance(content, str)
            return True
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False
    
    # ========================================================================
    # REPORTING METHODS
    # ========================================================================
    
    def report(self, 
               content: Any, 
               title: Optional[str] = None,
               **kwargs):
        """
        General reporting method with automatic type detection and validation.
        
        Args:
            content: Content to report
            title: Optional title
            **kwargs: Additional arguments
        """
        try:
            # Check if object implements Reportable protocol
            if isinstance(content, Reportable):
                item = content.to_report_item()
                self._add_item(item)
                self.logger.info(f"Added Reportable item: {item.title}")
                return
            
            # Auto-detect content type
            if MATPLOTLIB_AVAILABLE and isinstance(content, plt.Figure):
                self.report_figure(content, title=title, **kwargs)
            elif isinstance(content, str):
                # Check if it's an image file path
                if (Path(content).exists() and 
                    content.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))):
                    self.report_img(content, title=title, **kwargs)
                else:
                    self.report_text(content, title=title, **kwargs)
            elif PIL_AVAILABLE and isinstance(content, Image.Image):
                self.report_img(content, title=title, **kwargs)
            elif isinstance(content, pd.DataFrame):
                self.report_table(content, title=title, **kwargs)
            else:
                # Try to convert to string
                self.logger.warning(f"Unknown content type, converting to string: {type(content)}")
                self.report_text(str(content), title=title, **kwargs)
                
        except Exception as e:
            self.logger.error(f"Failed to add content to report: {e}")
            raise ReportError(f"Failed to add content: {e}")
    
    def report_figure(self, 
                      fig: plt.Figure, 
                      title: Optional[str] = None,
                      width: Optional[float] = None,
                      height: Optional[float] = None,
                      **kwargs):
        """Add a matplotlib figure to the report with validation."""
        if not MATPLOTLIB_AVAILABLE:
            raise MissingDependencyError(
                "matplotlib not available. Install with: pip install matplotlib"
            )
        
        if not isinstance(fig, plt.Figure):
            raise ValidationError(f"Expected plt.Figure, got {type(fig)}")
        
        try:
            if title is None and fig._suptitle is not None:
                title = fig._suptitle.get_text()
            
            item = ReportItem(
                content_type='figure',
                content=fig,
                title=title or 'Untitled Figure',
                width=width or fig.get_figwidth(),
                height=height or fig.get_figheight(),
                metadata={'kwargs': kwargs}
            )

            self._add_item(item)
            self._figures_to_close.append(fig)
            
            self.logger.info(f"✓ Added figure: {item.title}")
            
        except Exception as e:
            self.logger.error(f"Failed to add figure: {e}")
            raise ReportError(f"Failed to add figure: {e}")
    
    def report_text(self, 
                    text: str, 
                    title: Optional[str] = None,
                    is_katex: bool = False,
                    **kwargs):
        """Add text content to the report with validation."""
        if not isinstance(text, str):
            raise ValidationError(f"Expected str, got {type(text)}")
        
        try:
            item = ReportItem(
                content_type='text',
                content=text,
                title=title,
                metadata={'is_katex': is_katex, 'kwargs': kwargs}
            )

            self._add_item(item)
            
            preview = text[:50] + "..." if len(text) > 50 else text
            self.logger.info(f"✓ Added text: {title or preview}")
            
        except Exception as e:
            self.logger.error(f"Failed to add text: {e}")
            raise ReportError(f"Failed to add text: {e}")
    
    def report_img(self, 
                   img_source: Union[str, Path, Image.Image], 
                   title: Optional[str] = None,
                   width: Optional[float] = None,
                   height: Optional[float] = None,
                   **kwargs):
        """Add an image to the report with validation and optimization."""
        if not PIL_AVAILABLE:
            raise MissingDependencyError(
                "Pillow not available. Install with: pip install Pillow"
            )
        
        try:
            # Validate image source
            if isinstance(img_source, (str, Path)):
                img_path = Path(img_source)
                if not img_path.exists():
                    raise ValidationError(f"Image file not found: {img_path}")
            elif not isinstance(img_source, Image.Image):
                raise ValidationError(f"Expected str/Path/Image, got {type(img_source)}")
            
            item = ReportItem(
                content_type='image',
                content=img_source,
                title=title or 'Untitled Image',
                width=width,
                height=height,
                metadata={'kwargs': kwargs}
            )

            self._add_item(item)
            
            self.logger.info(f"✓ Added image: {item.title}")
            
        except Exception as e:
            self.logger.error(f"Failed to add image: {e}")
            raise ReportError(f"Failed to add image: {e}")
    
    def report_table(self, 
                     df: pd.DataFrame, 
                     title: Optional[str] = None,
                     max_rows: Optional[int] = None,
                     style: Optional[Dict] = None,
                     format_spec: Optional[Dict] = None,
                     **kwargs):
        """Add a pandas DataFrame with validation and formatting."""
        if not isinstance(df, pd.DataFrame):
            raise ValidationError(f"Expected pd.DataFrame, got {type(df)}")
        
        try:
            max_rows = max_rows or self.config.max_table_rows
            
            # Apply formatting if specified
            df_display = df.copy()
            if format_spec:
                for col, fmt in format_spec.items():
                    if col in df_display.columns:
                        df_display[col] = df_display[col].apply(fmt)
            
            # Truncate if too large
            if len(df_display) > max_rows:
                self.logger.info(f"Truncating table from {len(df_display)} to {max_rows} rows")
                df_display = df_display.head(max_rows)
            
            item = ReportItem(
                content_type='table',
                content=df_display,
                title=title or 'DataFrame',
                metadata={
                    'original_shape': df.shape,
                    'display_shape': df_display.shape,
                    'max_rows': max_rows,
                    'style': style or self.config.table_style,
                    'kwargs': kwargs
                }
            )

            self._add_item(item)
            
            self.logger.info(
                f"✓ Added table: {item.title} "
                f"({df.shape[0]} rows × {df.shape[1]} columns)"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add table: {e}")
            raise ReportError(f"Failed to add table: {e}")
    
    def report_raw_html(self, 
                       html: str, 
                       title: Optional[str] = None,
                       **kwargs):
        """Add raw HTML content with validation."""
        if not isinstance(html, str):
            raise ValidationError(f"Expected str, got {type(html)}")
        
        try:
            item = ReportItem(
                content_type='raw_html',
                content=html,
                title=title or 'HTML Section',
                metadata={'kwargs': kwargs}
            )

            self._add_item(item)
            
            self.logger.info(f"✓ Added HTML content: {item.title}")
            
        except Exception as e:
            self.logger.error(f"Failed to add HTML: {e}")
            raise ReportError(f"Failed to add HTML: {e}")
    
    # ========================================================================
    # GENERATION METHODS
    # ========================================================================
    
    def _get_progress_iterator(self, items: List, desc: str):
        """Get iterator with optional progress bar"""
        if self.config.show_progress and TQDM_AVAILABLE:
            return tqdm(items, desc=desc)
        return items
    
    def generate_pdf(self, 
                    filename: Optional[str] = None,
                    include_toc: Optional[bool] = None) -> Path:
        """
        Generate a PDF report with error handling and progress tracking.
        
        Args:
            filename: Output filename (optional)
            include_toc: Whether to include table of contents
            
        Returns:
            Path to generated PDF file
        """
        if not MATPLOTLIB_AVAILABLE:
            raise MissingDependencyError(
                "matplotlib required for PDF generation. "
                "Install with: pip install matplotlib"
            )
        
        include_toc = include_toc if include_toc is not None else self.config.include_toc
        
        if filename is None:
            filename = f"report_{self.metadata.report_id}.pdf"
        
        pdf_path = self.output_dir / filename
        
        try:
            self.logger.info(f"Starting PDF generation: {pdf_path}")
            
            with self.auto_close_figures():
                with PdfPages(pdf_path) as pdf:
                    # Title page
                    self._generate_pdf_title_page(pdf)
                    
                    # Table of contents
                    if include_toc and self.items:
                        self._generate_pdf_toc(pdf)
                    
                    # Content pages
                    items_iter = self._get_progress_iterator(
                        self.items, 
                        "Generating PDF"
                    )
                    
                    for item in items_iter:
                        self._add_pdf_item(pdf, item)
            
            self.logger.info(f"✓ PDF report generated: {pdf_path}")
            return pdf_path
            
        except Exception as e:
            self.logger.error(f"PDF generation failed: {e}")
            raise ReportGenerationError(f"Failed to generate PDF: {e}") from e
    
    def _generate_pdf_title_page(self, pdf: PdfPages):
        """Generate PDF title page"""
        fig, ax = plt.subplots(figsize=self.config.page_size)
        ax.axis('off')
        
        ax.text(0.5, 0.7, self.title, 
               fontsize=24, fontweight='bold',
               ha='center', va='center')
        ax.text(0.5, 0.6, 
               f"Generated: {self.metadata.creation_date.strftime('%Y-%m-%d %H:%M:%S')}", 
               fontsize=12, ha='center', va='center')
        ax.text(0.5, 0.5, f"Author: {self.author}", 
               fontsize=12, ha='center', va='center')
        ax.text(0.5, 0.4, f"Items: {self.metadata.item_count}", 
               fontsize=12, ha='center', va='center')
        
        pdf.savefig(fig, bbox_inches='tight', dpi=self.config.dpi)
        plt.close(fig)
    
    def _generate_pdf_toc(self, pdf: PdfPages):
        """Generate PDF table of contents"""
        fig, ax = plt.subplots(figsize=self.config.page_size)
        ax.axis('off')
        ax.text(0.1, 0.9, "Table of Contents", 
               fontsize=18, fontweight='bold')
        
        y_pos = 0.8
        for i, item in enumerate(self.items):
            title = item.title or f"Item {i+1}"
            ax.text(0.1, y_pos, f"{i+1}. {title}", fontsize=12)
            y_pos -= 0.05
            
            if y_pos < 0.1:
                pdf.savefig(fig, bbox_inches='tight', dpi=self.config.dpi)
                plt.close(fig)
                fig, ax = plt.subplots(figsize=self.config.page_size)
                ax.axis('off')
                y_pos = 0.9
        
        pdf.savefig(fig, bbox_inches='tight', dpi=self.config.dpi)
        plt.close(fig)
    
    def _add_pdf_item(self, pdf: PdfPages, item: ReportItem):
        """Add a single item to PDF"""
        try:
            if item.content_type == 'figure':
                pdf.savefig(item.content, bbox_inches='tight', dpi=self.config.dpi)
                
            elif item.content_type == 'text':
                fig = self._create_text_figure(item)
                pdf.savefig(fig, bbox_inches='tight', dpi=self.config.dpi)
                plt.close(fig)
                
            elif item.content_type == 'image' and PIL_AVAILABLE:
                fig = self._create_image_figure(item)
                pdf.savefig(fig, bbox_inches='tight', dpi=self.config.dpi)
                plt.close(fig)
                
            elif item.content_type == 'table':
                fig = self._create_table_figure(item)
                pdf.savefig(fig, bbox_inches='tight', dpi=self.config.dpi)
                plt.close(fig)
                
        except Exception as e:
            self.logger.error(f"Failed to add item '{item.title}' to PDF: {e}")
    
    def _create_text_figure(self, item: ReportItem) -> plt.Figure:
        """Create matplotlib figure for text content"""
        fig, ax = plt.subplots(figsize=self.config.page_size)
        ax.axis('off')
        
        y_pos = 0.9
        if item.title:
            ax.text(0.1, y_pos, item.title, 
                   fontsize=16, fontweight='bold')
            y_pos -= 0.05
        
        # Simple text wrapping
        text = item.content
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= 80:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        for line in lines[:40]:  # Limit lines per page
            ax.text(0.1, y_pos, line, fontsize=10, wrap=True)
            y_pos -= 0.04
        
        return fig
    
    def _create_image_figure(self, item: ReportItem) -> plt.Figure:
        """Create matplotlib figure for image content"""
        if isinstance(item.content, (str, Path)):
            img = Image.open(item.content)
        else:
            img = item.content
        
        # Optimize image
        img = self.image_optimizer.optimize_image(img)
        
        fig, ax = plt.subplots(figsize=self.config.page_size)
        ax.axis('off')
        
        if item.title:
            ax.set_title(item.title, fontsize=16, pad=20)
        
        ax.imshow(img)
        
        if isinstance(item.content, (str, Path)):
            img.close()
        
        return fig
    
    def _create_table_figure(self, item: ReportItem) -> plt.Figure:
        """Create matplotlib figure for table content"""
        fig, ax = plt.subplots(figsize=self.config.page_size)
        ax.axis('off')
        
        if item.title:
            ax.set_title(item.title, fontsize=16, pad=20)
        
        # Convert DataFrame to table
        df = item.content
        table_data = [df.columns.tolist()] + df.values.tolist()
        
        table = ax.table(cellText=table_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#667eea')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        return fig
    
    def generate_html(self, 
                     filename: Optional[str] = None,
                     include_katex: Optional[bool] = None,
                     theme: Optional[Theme] = None) -> Path:
        """
        Generate an HTML report using templates with collapsible sections.
        
        Args:
            filename: Output filename (optional)
            include_katex: Whether to include KaTeX
            theme: Theme to use
            
        Returns:
            Path to generated HTML file
        """
        include_katex = include_katex if include_katex is not None else self.config.include_katex
        theme = theme or self.config.theme
        
        if filename is None:
            filename = f"report_{self.metadata.report_id}.html"
        
        html_path = self.output_dir / filename
        
        try:
            self.logger.info(f"Starting HTML generation: {html_path}")

            # Prepare sections — respects ReportSection grouping when present
            sections = self._build_html_sections()
            
            # Prepare template context
            context = {
                'title': self.title,
                'author': self.author,
                'creation_date': self.metadata.creation_date.strftime('%Y-%m-%d %H:%M:%S'),
                'item_count': self.metadata.item_count,
                'sections': sections,
                'include_katex': include_katex
            }
            
            # Render template
            if JINJA2_AVAILABLE:
                template_str = HTMLTemplates.get_template(theme)
                if self.config.custom_css:
                    # Inject custom CSS
                    template_str = template_str.replace(
                        '</style>',
                        f'{self.config.custom_css}\n</style>'
                    )
                template = Template(template_str)
                html_content = template.render(**context)
            else:
                # Fallback to manual rendering
                html_content = self._render_html_manual(context, theme)
            
            # Write HTML file
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Close figures and clear render cache after a successful write
            if self.config.auto_close_figures:
                for fig in self._figures_to_close:
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
                self._figures_to_close.clear()
            self._figure_render_cache.clear()

            self.logger.info(f"✓ HTML report generated: {html_path}")
            return html_path

        except Exception as e:
            self.logger.error(f"HTML generation failed: {e}")
            raise ReportGenerationError(f"Failed to generate HTML: {e}") from e
    
    def _build_html_sections(self) -> List[Dict]:
        """Build the sections list for Jinja / manual HTML rendering.

        * If ``section()`` was used, ``_content_order`` mixes
          ``ReportSection`` (grouped) and standalone ``ReportItem`` entries.
        * If ``section()`` was never called, ``_content_order`` contains only
          ``ReportItem`` entries — each becomes its own collapsible section,
          which is identical to the old (pre-sections) behaviour.
        """
        # Fallback: nothing was ever added via report_* methods.
        if not self._content_order and not self.items:
            return []

        # If _content_order is empty but items exist (direct append by legacy
        # code), fall back to the flat list.
        source = self._content_order if self._content_order else self.items

        items_iter = self._get_progress_iterator(source, "Generating HTML")

        sections: List[Dict] = []
        for entry in items_iter:
            if isinstance(entry, ReportSection):
                content_parts: List[str] = []
                for item in entry.items:
                    if item.title:
                        content_parts.append(
                            f'<h3 style="color: #555; margin: 25px 0 10px 0; '
                            f'padding-bottom: 6px; border-bottom: 1px solid #eee;">'
                            f'{item.title}</h3>'
                        )
                    content_parts.append(self._render_html_item(item))
                sections.append({
                    'title': entry.title,
                    'content': '\n'.join(content_parts),
                })
            else:
                # Standalone ReportItem → its own collapsible section
                sections.append({
                    'title': entry.title or 'Untitled',
                    'content': self._render_html_item(entry),
                })
        return sections

    def _render_html_item(self, item: ReportItem) -> str:
        """Render a single item to HTML"""
        try:
            if item.content_type == 'figure' and MATPLOTLIB_AVAILABLE:
                fig_id = id(item.content)
                if fig_id not in self._figure_render_cache:
                    use_jpeg = self.config.output_format.lower() in ('jpeg', 'jpg')
                    fmt = 'jpeg' if use_jpeg else 'png'
                    buf = io.BytesIO()
                    save_kwargs = dict(bbox_inches='tight', dpi=self.config.dpi)
                    if use_jpeg:
                        save_kwargs['pil_kwargs'] = {'quality': self.config.image_quality, 'optimize': True}
                    item.content.savefig(buf, format=fmt, **save_kwargs)
                    buf.seek(0)
                    self._figure_render_cache[fig_id] = base64.b64encode(buf.read()).decode()
                img_str = self._figure_render_cache[fig_id]
                mime = 'jpeg' if self.config.output_format.lower() in ('jpeg', 'jpg') else 'png'
                return f'<div class="figure"><img src="data:image/{mime};base64,{img_str}" alt="{item.title}"></div>'
                
            elif item.content_type == 'text':
                if item.metadata.get('is_katex', False):
                    return f'<div class="katex-equation">{item.content}</div>'
                else:
                    content = item.content.replace('\n', '<br>').replace('  ', ' &nbsp;')
                    return f'<div class="text-content">{content}</div>'
                    
            elif item.content_type == 'image' and PIL_AVAILABLE:
                if isinstance(item.content, (str, Path)):
                    with open(item.content, 'rb') as f:
                        img_str = base64.b64encode(f.read()).decode()
                else:
                    img_str = self.image_optimizer.image_to_base64(item.content)
                return f'<div class="figure"><img src="data:image/png;base64,{img_str}" alt="{item.title}"></div>'
                    
            elif item.content_type == 'table':
                table_html = item.content.to_html(classes='dataframe', index=True)
                shape = item.metadata.get('original_shape', item.content.shape)
                return f'<div class="table-container">{table_html}</div><p><em>Table shape: {shape[0]} rows × {shape[1]} columns</em></p>'
                
            elif item.content_type == 'raw_html':
                return item.content
                
            return '<p>Unsupported content type</p>'
            
        except Exception as e:
            self.logger.error(f"Failed to render item '{item.title}': {e}")
            return f'<p>Error rendering item: {e}</p>'
    
    def _render_html_manual(self, context: Dict, theme: Theme) -> str:
        """Fallback manual HTML rendering without Jinja2"""
        self.logger.warning("Jinja2 not available, using manual HTML rendering")
        
        # Use template string with manual substitution
        template = HTMLTemplates.get_template(theme)
        
        # Simple manual replacement
        html = template.replace('{{ title }}', context['title'])
        html = html.replace('{{ author }}', context['author'])
        html = html.replace('{{ creation_date }}', context['creation_date'])
        html = html.replace('{{ item_count }}', str(context['item_count']))
        
        # Build sections
        sections_html = ''
        for i, section in enumerate(context['sections']):
            sections_html += f'''
    <div class="section collapsed" id="section-{i+1}">
        <div class="section-header" onclick="toggleSection({i+1})">
            <h2 class="section-title">{section["title"]}</h2>
            <div class="section-toggle">
                <span class="toggle-text">Click to expand</span>
                <div class="toggle-icon">▶</div>
            </div>
        </div>
        <div class="section-content">
            <div class="section-content-inner">
                {section["content"]}
            </div>
        </div>
    </div>
            '''
        
        # Find and replace sections placeholder
        html = html.replace('{% for section in sections %}', '')
        html = html.replace('{% endfor %}', sections_html)
        
        # Handle katex conditional
        if context.get('include_katex'):
            html = html.replace('{% if include_katex %}', '')
            html = html.replace('{% endif %}', '')
        else:
            # Remove katex section
            start = html.find('{% if include_katex %}')
            end = html.find('{% endif %}')
            if start != -1 and end != -1:
                html = html[:start] + html[end+len('{% endif %}'):]
        
        return html
    
    def _render_md_item(self, item: ReportItem, parts: List[str],
                        fig_counter: int, img_counter: int,
                        heading_level: str = '##') -> tuple:
        """Render a single *item* to Markdown, appending strings to *parts*.

        Returns the updated (fig_counter, img_counter) for filename uniqueness.
        """
        title = item.title or 'Untitled'
        parts.append(f"\n{heading_level} {title}\n\n")

        if item.content_type == 'figure':
            fig_counter += 1
            fig_filename = f"figure_{fig_counter}.png"
            fig_path = self.output_dir / fig_filename
            item.content.savefig(fig_path, bbox_inches='tight',
                                 dpi=self.config.dpi)
            parts.append(f"![Figure: {title}]({fig_filename})\n\n")

        elif item.content_type == 'text':
            if item.metadata.get('is_katex', False):
                parts.append(f"$${item.content}$$\n\n")
            else:
                parts.append(f"{item.content}\n\n")

        elif item.content_type == 'image':
            if isinstance(item.content, (str, Path)):
                fname = Path(item.content).name
                parts.append(f"![{title}]({fname})\n\n")
            else:
                img_counter += 1
                img_filename = f"image_{img_counter}.png"
                img_path = self.output_dir / img_filename
                item.content.save(img_path)
                parts.append(f"![{title}]({img_filename})\n\n")

        elif item.content_type == 'table':
            parts.append(f"{item.content.to_markdown()}\n\n")

        elif item.content_type == 'raw_html':
            parts.append(f"```html\n{item.content}\n```\n\n")

        return fig_counter, img_counter

    def generate_markdown(self,
                         filename: Optional[str] = None) -> Path:
        """
        Generate a Markdown report.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to generated Markdown file
        """
        if filename is None:
            filename = f"report_{self.metadata.report_id}.md"
        
        md_path = self.output_dir / filename
        
        try:
            self.logger.info(f"Starting Markdown generation: {md_path}")
            
            markdown = f"""# {self.title}

**Generated**: {self.metadata.creation_date.strftime('%Y-%m-%d %H:%M:%S')}  
**Author**: {self.author}  
**Items**: {self.metadata.item_count}

"""
            
            source = self._content_order if self._content_order else self.items
            items_iter = self._get_progress_iterator(source, "Generating Markdown")
            fig_counter = 0
            img_counter = 0

            for entry in items_iter:
                if isinstance(entry, ReportSection):
                    markdown += f"\n## {entry.title}\n\n"
                    for item in entry.items:
                        fig_counter, img_counter = self._render_md_item(
                            item, markdown_parts := [], fig_counter, img_counter,
                            heading_level='###',
                        )
                        markdown += ''.join(markdown_parts)
                else:
                    fig_counter, img_counter = self._render_md_item(
                        entry, markdown_parts := [], fig_counter, img_counter,
                        heading_level='##',
                    )
                    markdown += ''.join(markdown_parts)
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
            
            self.logger.info(f"✓ Markdown report generated: {md_path}")
            return md_path
            
        except Exception as e:
            self.logger.error(f"Markdown generation failed: {e}")
            raise ReportGenerationError(f"Failed to generate Markdown: {e}") from e
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def clear(self):
        """Clear all report items and close figures"""
        try:
            if self.config.auto_close_figures:
                for item in self.items:
                    if item.content_type == 'figure':
                        plt.close(item.content)
            
            self.items.clear()
            self._figures_to_close.clear()
            self._content_order.clear()
            self._current_section = None
            self.metadata.item_count = 0
            
            self.logger.info("✓ Report cleared")
            
        except Exception as e:
            self.logger.error(f"Failed to clear report: {e}")
    
    def summary(self):
        """Print a comprehensive summary of the report"""
        print(f"\n{'='*60}")
        print(f"REPORT SUMMARY: {self.title}")
        print(f"{'='*60}")
        print(f"Author:      {self.author}")
        print(f"Created:     {self.metadata.creation_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Report ID:   {self.metadata.report_id}")
        print(f"Items:       {self.metadata.item_count}")
        print(f"Output Dir:  {self.output_dir}")
        
        # Count by type
        type_counts = {}
        for item in self.items:
            type_counts[item.content_type] = type_counts.get(item.content_type, 0) + 1
        
        print(f"\nContent Types:")
        for content_type, count in sorted(type_counts.items()):
            print(f"  - {content_type:12s}: {count}")
        
        # Show section structure if sections were used, otherwise flat list
        has_sections = any(isinstance(e, ReportSection) for e in self._content_order)
        if has_sections:
            print(f"\nSections:")
            idx = 1
            for entry in self._content_order:
                if isinstance(entry, ReportSection):
                    print(f"  {idx:3d}. [section    ] {entry.title} ({len(entry.items)} items)")
                    for j, item in enumerate(entry.items, 1):
                        title = item.title or f"Item {j}"
                        print(f"       {j:3d}. [{item.content_type:10s}] {title}")
                    idx += 1
                else:
                    title = entry.title or f"Item {idx}"
                    print(f"  {idx:3d}. [{entry.content_type:10s}] {title}")
                    idx += 1
        else:
            print(f"\nItems:")
            for i, item in enumerate(self.items, 1):
                title = item.title or f"Item {i}"
                print(f"  {i:3d}. [{item.content_type:10s}] {title}")
        
        print(f"\nConfiguration:")
        print(f"  - DPI:              {self.config.dpi}")
        print(f"  - Max Image Size:   {self.config.max_image_size}")
        print(f"  - Theme:            {self.config.theme.value}")
        print(f"  - Progress Bar:     {self.config.show_progress}")
        print(f"  - Auto Close Figs:  {self.config.auto_close_figures}")
        print(f"  - Collapsible Sections: Yes (in HTML)")
        
        print(f"{'='*60}\n")
    
    def to_dict(self) -> Dict:
        """Serialize report metadata for testing/debugging"""
        def _serialize_entry(entry):
            if isinstance(entry, ReportSection):
                return {
                    'type': 'section',
                    'title': entry.title,
                    'items': [
                        {'type': it.content_type, 'title': it.title, 'metadata': it.metadata}
                        for it in entry.items
                    ],
                }
            return {'type': entry.content_type, 'title': entry.title, 'metadata': entry.metadata}

        content = self._content_order if self._content_order else self.items
        return {
            'title': self.title,
            'author': self.author,
            'metadata': self.metadata.to_dict(),
            'content': [_serialize_entry(e) for e in content],
            # Flat list kept for backward compatibility
            'items': [
                {
                    'type': item.content_type,
                    'title': item.title,
                    'metadata': item.metadata
                }
                for item in self.items
            ],
            'config': {
                'dpi': self.config.dpi,
                'theme': self.config.theme.value,
                'max_image_size': self.config.max_image_size
            }
        }