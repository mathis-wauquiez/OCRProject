"""Shared font configuration for all figure-generation scripts.

Sets up matplotlib to use a CJK-capable font for labels, titles, and
tick labels so that Chinese/Japanese characters render without the
"missing glyph" warning.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── Preferred CJK font families (in order of preference) ──────────
_CJK_PREFERRED = [
    'WenQuanYi Zen Hei',
    'WenQuanYi Micro Hei',
    'Noto Sans CJK SC',
    'Noto Sans CJK TC',
    'Noto Sans CJK JP',
    'IPAGothic',
    'IPAPGothic',
    'AR PL UMing TW',
    'AR PL UKai TW',
    'SimHei',
    'Microsoft YaHei',
]


def _find_cjk_font() -> str | None:
    """Return the name of the first available CJK-capable font, or None."""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_PREFERRED:
        if name in available:
            return name
    # Fallback: any font whose name suggests CJK support
    for name in available:
        if any(kw in name.lower() for kw in ('cjk', 'hei', 'gothic', 'ming')):
            return name
    return None


def _find_cjk_font_path() -> str | None:
    """Return the *file path* of the first available CJK font (for PIL)."""
    available = {f.name: f.fname for f in fm.fontManager.ttflist}
    for name in _CJK_PREFERRED:
        if name in available:
            return available[name]
    for name, path in available.items():
        if any(kw in name.lower() for kw in ('cjk', 'hei', 'gothic', 'ming')):
            return path
    return None


# The resolved CJK font name (may be None on systems without CJK fonts)
CJK_FONT_NAME = _find_cjk_font()
CJK_FONT_PATH = _find_cjk_font_path()


def configure_matplotlib_fonts():
    """Apply a unified CJK-capable font to the matplotlib rcParams.

    Call this once at the top of every figure-generation script, before
    any ``plt.figure()`` or ``fig.savefig()`` call.
    """
    if CJK_FONT_NAME is not None:
        # Prepend the CJK font to all font families so CJK glyphs are
        # found before DejaVu gives up.
        for family in ('sans-serif', 'serif', 'monospace'):
            current = plt.rcParams.get(f'font.{family}', [])
            if CJK_FONT_NAME not in current:
                plt.rcParams[f'font.{family}'] = [CJK_FONT_NAME] + list(current)
        # Default family → sans-serif (which now starts with the CJK font)
        plt.rcParams['font.family'] = 'sans-serif'
    # Suppress the missing-glyph warning that fires even when a fallback
    # is available (matplotlib < 3.9).
    import warnings
    warnings.filterwarnings('ignore', message='Glyph.*missing from.*font')
