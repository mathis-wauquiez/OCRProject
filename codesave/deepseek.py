"""
Traditional Chinese Character Meaning Lookup

A module that provides functions to look up meanings of Traditional Chinese
characters using the DeepSeek API, with caching and parallel processing support.
"""

import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI
from tqdm import tqdm


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CACHE_PATH = "deepseek_char_cache.json"
DEFAULT_MAX_WORKERS = 6  # keep conservative to avoid rate limits


# ---------------------------------------------------------------------
# Cache Management
# ---------------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
_CHAR_CACHE: dict[str, str] = {}


def _load_cache() -> dict[str, str]:
    """Load cache from disk if it exists."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache() -> None:
    """Save cache to disk."""
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_CHAR_CACHE, f, ensure_ascii=False, indent=2)


# Initialize cache on module load
_CHAR_CACHE = _load_cache()


# ---------------------------------------------------------------------
# DeepSeek Client
# ---------------------------------------------------------------------

def _get_client(api_key: Optional[str] = None) -> OpenAI:
    """Create and return a DeepSeek API client."""
    key = "sk-850474e409644f99be281abe6f962f3a"
    if not key:
        raise ValueError(
            "API key required. Pass api_key parameter or set DEEPSEEK_API_KEY environment variable."
        )
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


# ---------------------------------------------------------------------
# Single Character Lookup
# ---------------------------------------------------------------------

def get_character_meaning(
    char: str,
    client: Optional[OpenAI] = None,
    api_key: Optional[str] = None,
    use_cache: bool = True,
) -> str:
    """
    Get the meaning of a single Traditional Chinese character.

    Args:
        char: A single Chinese character.
        client: Optional pre-configured OpenAI client for DeepSeek.
        api_key: Optional API key (uses env var DEEPSEEK_API_KEY if not provided).
        use_cache: Whether to use disk caching (default: True).

    Returns:
        A string explanation of the character's meaning.

    Raises:
        ValueError: If input is not a single character.
    """
    if not isinstance(char, str) or len(char) != 1:
        raise ValueError("Input must be a single Chinese character.")

    # Check cache
    if use_cache:
        with _CACHE_LOCK:
            if char in _CHAR_CACHE:
                return _CHAR_CACHE[char]

    # Create client if not provided
    if client is None:
        client = _get_client(api_key)

    # Build prompt
    prompt = (
        "Explain the meaning of the following Traditional Chinese character.\n\n"
        "Requirements:\n"
        "- **Pronunciation:** Give the Mandarin pinyin (with tone marks or numbers) and optionally Cantonese jyutping\n"
        "- **Simplified form:** Show the Simplified Chinese equivalent (or note if it's the same)\n"
        "- **Primary meaning:** Give the main definition(s)\n"
        "- **Common usages:** Mention common words or contexts where this character appears\n"
        "- **Etymology/radicals:** Briefly note the radical and any relevant origin\n"
        "- Keep the explanation concise and well-structured\n\n"
        f"Character: {char}"
    )

    # API call
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a knowledgeable Chinese linguistics assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,  # Slightly increased for additional info
    )

    answer = response.choices[0].message.content.strip()

    # Update cache
    if use_cache:
        with _CACHE_LOCK:
            _CHAR_CACHE[char] = answer
            _save_cache()

    return answer

# ---------------------------------------------------------------------
# Batch Processing
# ---------------------------------------------------------------------

def get_character_meanings(
    characters: list[str],
    api_key: Optional[str] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    use_cache: bool = True,
    show_progress: bool = True,
) -> dict[str, str]:
    """
    Get meanings for multiple Traditional Chinese characters in parallel.

    Args:
        characters: List of Chinese characters to look up.
        api_key: Optional API key (uses env var DEEPSEEK_API_KEY if not provided).
        max_workers: Maximum number of parallel threads (default: 6).
        use_cache: Whether to use disk caching (default: True).
        show_progress: Whether to show a progress bar (default: True).

    Returns:
        A dictionary mapping each character to its meaning explanation.
        Characters that failed to process will have None as their value.

    Example:
        >>> meanings = get_character_meanings(['龍', '鳳', '虎'])
        >>> print(meanings['龍'])
    """
    # Deduplicate while preserving order
    unique_chars = list(dict.fromkeys(characters))

    # Create shared client
    client = _get_client(api_key)

    results: dict[str, Optional[str]] = {}

    def worker(char: str) -> tuple[str, Optional[str]]:
        """Worker function for parallel processing."""
        try:
            meaning = get_character_meaning(
                char, client=client, use_cache=use_cache
            )
            return char, meaning
        except Exception as e:
            print(f"Error processing '{char}': {e}")
            return char, None

    # Parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, char) for char in unique_chars]

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Looking up characters")

        for future in iterator:
            char, meaning = future.result()
            results[char] = meaning

    # Return results in original order, including duplicates
    return {char: results.get(char) for char in characters}


def get_character_meanings_ordered(
    characters: list[str],
    api_key: Optional[str] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    use_cache: bool = True,
    show_progress: bool = True,
) -> list[Optional[str]]:
    """
    Get meanings for multiple characters, returning results as a list in the same order.

    This is useful when you need to maintain alignment with the input list,
    such as when working with DataFrame columns.

    Args:
        characters: List of Chinese characters to look up.
        api_key: Optional API key (uses env var DEEPSEEK_API_KEY if not provided).
        max_workers: Maximum number of parallel threads (default: 6).
        use_cache: Whether to use disk caching (default: True).
        show_progress: Whether to show a progress bar (default: True).

    Returns:
        A list of meaning explanations in the same order as the input.
        Failed lookups will have None in their position.

    Example:
        >>> chars = ['龍', '鳳', '虎']
        >>> meanings = get_character_meanings_ordered(chars)
        >>> for char, meaning in zip(chars, meanings):
        ...     print(f"{char}: {meaning[:50]}...")
    """
    meanings_dict = get_character_meanings(
        characters,
        api_key=api_key,
        max_workers=max_workers,
        use_cache=use_cache,
        show_progress=show_progress,
    )
    return [meanings_dict.get(char) for char in characters]


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def clear_cache() -> None:
    """Clear the in-memory and disk cache."""
    global _CHAR_CACHE
    with _CACHE_LOCK:
        _CHAR_CACHE = {}
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)


def get_cached_characters() -> list[str]:
    """Return a list of all characters currently in the cache."""
    with _CACHE_LOCK:
        return list(_CHAR_CACHE.keys())

from IPython.display import display, HTML
import re

def md_to_html(text: str) -> str:
    """Simple markdown to HTML conversion for common patterns."""
    # Bold: **text** -> <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic: *text* -> <em>text</em>
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Line breaks
    text = text.replace('\n', '<br>')
    # List items: - item -> bullet
    text = re.sub(r'<br>\s*-\s+', '<br>• ', text)
    return text


def display_character_with_meaning(svg, meaning: str, char: str = None, conf : float = 1.0, thresh: float = 0.4, scale: float = 1.5):
    """
    Display an SVG character alongside its meaning in a nicely formatted layout.
    
    Args:
        svg: SVG object with to_string() method
        meaning: Character meaning text from DeepSeek (markdown)
        char: Optional character label to display
        scale: Scale factor for the SVG display
    """
    svg_str = svg.to_string()
    meaning_html = md_to_html(meaning)
    
    char_header = f"<h2 style='margin: 0 0 10px 0; color: #333;'>【{char}】 - 【{conf:.3f}】</h2>" if char else ""
    
    html = f"""
    <div style="display: flex; gap: 24px; align-items: flex-start; 
                padding: 16px; margin: 12px 0; 
                border: 1px solid #e0e0e0; border-radius: 8px;
                background: {'#fafafa' if conf > thresh else "#ffefef"};">
        <div style="flex-shrink: 0; padding: 8px; background: white; 
                    border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="width: {100 * scale}px; height: {100 * scale}px;">
                {svg_str}
            </div>
        </div>
        <div style="flex: 1; min-width: 0;">
            {char_header}
            <div style="color: #444; line-height: 1.6; font-size: 14px;">
                {meaning_html}
            </div>
        </div>
    </div>
    """
    display(HTML(html))
    return html


def display_character_or_error(svg, char: str, conf: float, thresh = 0.4, scale: float = 1.5):
    """
    Display a character with its meaning, or show an error message if OCR failed.
    """
    try:
        meaning = get_character_meaning(char=char)
        return display_character_with_meaning(svg, meaning, char=char, conf=conf, thresh=thresh, scale=scale)
    except ValueError:
        svg_str = svg.to_string()
        html = f"""
        <div style="display: flex; gap: 24px; align-items: flex-start;
                    padding: 16px; margin: 12px 0;
                    border: 1px solid #ffcdd2; border-radius: 8px;
                    background: #ffebee;">
            <div style="flex-shrink: 0; padding: 8px; background: white;
                        border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="width: {100 * scale}px; height: {100 * scale}px;">
                    {svg_str}
                </div>
            </div>
            <div style="flex: 1; color: #c62828; font-weight: 500;">
                ⚠️ Character not recognized properly by OCR
            </div>
        </div>
        """
        display(HTML(html))
        return html
# ---------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Example: Look up meanings for a list of characters
    sample_chars = ["龍", "鳳", "虎", "龜"]

    print("Looking up character meanings...")
    print("=" * 50)

    # Set your API key via environment variable or pass directly:
    # export DEEPSEEK_API_KEY="your-key-here"
    # Or: meanings = get_character_meanings(sample_chars, api_key="your-key")

    try:
        meanings = get_character_meanings(sample_chars)

        for char, meaning in meanings.items():
            print(f"\n【{char}】")
            print(meaning if meaning else "Failed to retrieve meaning")
            print("-" * 50)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your DEEPSEEK_API_KEY environment variable.")