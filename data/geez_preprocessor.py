"""
Ge'ez Script Normalization and Text Cleaning Module
Part of Tereguwami (ተርጓሚ) Natural Language and Translation Pipeline (§4.4, §8.4)

Provides rigorous orthographic normalization for Ethiopian languages written in the Ge'ez abugida,
eliminating homophone character divergence, stripping archaic Ethiopic word separators,
standardizing punctuation, and regularizing sub-word boundary representation.
"""

import re
import unicodedata
from typing import List, Optional, Dict


class GeezPreprocessor:
    """
    Standardized Ge'ez script preprocessor and orthographic normalizer.
    
    In formal and informal Ethiopian digital text, homophonic characters (e.g. ሰ and ሠ,
    ሀ, ሐ, and ኀ, አ and ዐ, ጸ and ፀ) introduce artificial vocabulary explosion.
    This normalizer canonicalizes character variants according to modern Ethiopian
    computational linguistics standards.
    """

    # Ethiopic punctuation marks defined in Unicode range U+1360 - U+137C
    ETHIOPIC_PUNCTUATION_PATTERN = re.compile(r"[\u1360-\u137C]")
    # Archaic word separator (U+1361 ETHIOPIC WORDSPACE ፡) and full stop (U+1362 ።)
    ETHIOPIC_WORD_SEPARATORS = re.compile(r"[፡|]")
    # General non-alphanumeric punctuation
    EXTRA_PUNCTUATION = re.compile(r"[\.,!?;:\"'«»\(\)\[\]\{\}\-_\\/+=@#$%^&*~`|]")

    # Canonical homophone mapping table (Ge'ez abugida character unification)
    HOMOPHONE_MAP: Dict[str, str] = {
        # 1. Ha family (ሐ, ኀ, ኃ, ኻ -> ሀ)
        "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
        "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
        "ኻ": "ሃ", "ኺ": "ሂ", "ኼ": "ሄ", "ኽ": "ህ", "ኾ": "ሆ", "ሗ": "ኋ",
        # 2. Se family (ሠ -> ሰ)
        "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ", "ሧ": "ሷ",
        # 3. A family (ዐ -> አ)
        "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
        # 4. Tse family (ፀ -> ጸ)
        "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
    }

    # Ethiopic numeral values
    ETHIOPIC_NUMERALS: Dict[str, int] = {
        "፩": 1, "፪": 2, "፫": 3, "፬": 4, "፭": 5, "፮": 6, "፯": 7, "፰": 8, "፱": 9,
        "፲": 10, "፳": 20, "፴": 30, "፵": 40, "፶": 50, "፷": 60, "፸": 70, "፹": 80, "፺": 90,
        "፻": 100, "፼": 10000
    }

    def __init__(self, normalize_homophones: bool = True, strip_punctuation: bool = True):
        self.normalize_homophones = normalize_homophones
        self.strip_punctuation = strip_punctuation

    def clean_word_separators(self, text: str) -> str:
        """Replace archaic Ethiopic word dividers (፡, |) with standard whitespace."""
        return self.ETHIOPIC_WORD_SEPARATORS.sub(" ", text)

    def normalize_characters(self, text: str) -> str:
        """Canonicalize variant Ge'ez homophones to primary character representations."""
        result = []
        for char in text:
            result.append(self.HOMOPHONE_MAP.get(char, char))
        return "".join(result)

    def remove_punctuation(self, text: str, preserve_sentence_boundary: bool = False) -> str:
        """Strip Ethiopic and standard punctuation marks."""
        if preserve_sentence_boundary:
            # Preserve sentence terminators (። -> .)
            text = text.replace("።", " . ")
        else:
            text = text.replace("።", " ")
        
        text = self.ETHIOPIC_PUNCTUATION_PATTERN.sub(" ", text)
        text = self.EXTRA_PUNCTUATION.sub(" ", text)
        return text

    def clean(self, raw_text: str, preserve_boundary: bool = False) -> str:
        """
        Full preprocessing pipeline:
        1. Unicode normalization (NFC)
        2. Ethiopic word divider replacement
        3. Homophone unification
        4. Punctuation cleaning
        5. Whitespace deduplication
        """
        if not raw_text or not isinstance(raw_text, str):
            return ""

        # Step 1: Unicode NFC normalization
        text = unicodedata.normalize("NFC", raw_text)

        # Step 2: Replace Ethiopic word separators
        text = self.clean_word_separators(text)

        # Step 3: Homophone normalization
        if self.normalize_homophones:
            text = self.normalize_characters(text)

        # Step 4: Punctuation stripping
        if self.strip_punctuation:
            text = self.remove_punctuation(text, preserve_sentence_boundary=preserve_boundary)

        # Step 5: Clean extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_words(self, text: str) -> List[str]:
        """Convenience method to clean and split into word tokens."""
        cleaned = self.clean(text)
        return [w for w in cleaned.split(" ") if w]


# Global singleton instance for high-throughput invocation
geez_preprocessor = GeezPreprocessor()
