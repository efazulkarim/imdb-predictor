# ============================================================
# TEXT PREPROCESSING
# ============================================================
"""
Script preprocessing and feature extraction module.
Handles cleaning and feature extraction from movie scripts.
"""

import re
import numpy as np


# Bumped whenever clean_text_for_sbert changes meaning, so that
# downstream embedding caches invalidate automatically.
SBERT_PREPROCESSING_VERSION = 2


class ScriptPreprocessor:
    """Advanced preprocessing for movie scripts."""

    @staticmethod
    def clean_text(text):
        """
        Aggressive normalization used for the structural / TF-IDF pipelines.

        Lowercases, strips stage directions, character names, timestamps,
        and most punctuation. Suitable for bag-of-words style features
        but NOT ideal as input to SBERT (which is trained on cased,
        fully-punctuated natural language).
        """
        # Convert to lowercase
        text = text.lower()

        # Remove stage directions [GUNSHOT], (CRYING), etc.
        text = re.sub(r'\[.*?\]', ' ', text)
        text = re.sub(r'\(.*?\)', ' ', text)

        # Remove character names (JOHN:, MARY:, etc.)
        text = re.sub(r'^[A-Z][A-Z\s]+:', ' ', text, flags=re.MULTILINE | re.IGNORECASE)

        # Remove timestamps and scene numbers
        text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?', ' ', text)
        text = re.sub(r'\bscene\s*\d+\b', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\bint\.?\b|\bext\.?\b', ' ', text, flags=re.IGNORECASE)

        # Keep only letters, numbers, and basic punctuation
        text = re.sub(r"[^a-zA-Z0-9'.,!?\s]", ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    @staticmethod
    def clean_text_for_sbert(text):
        """
        Light cleaning suitable as SBERT input.

        Preserves case and natural punctuation (which SBERT was trained
        on), but removes obvious screenplay artifacts that would otherwise
        dominate the embedding:
            - stage directions in [brackets] and (parens)
            - ALL-CAPS character cues at start of line ("JOHN:")
            - INT./EXT. scene heading prefixes
            - timestamps and scene numbers

        Compared to clean_text, this keeps:
            - case (so SBERT sees real sentences)
            - quotes, dashes, semicolons, etc. (sentence boundaries)
            - paragraph breaks
        """
        # Stage directions
        text = re.sub(r'\[.*?\]', ' ', text)
        text = re.sub(r'\(.*?\)', ' ', text)

        # Character cue lines: "JOHN:" or "MARY ANN:" at start of line.
        # Case-sensitive so we don't strip ordinary "Word:" inside dialogue.
        text = re.sub(r'^[A-Z][A-Z0-9\s]{1,30}:', ' ', text, flags=re.MULTILINE)

        # INT. / EXT. scene heading markers
        text = re.sub(r'\b(INT|EXT)\.?\s', ' ', text)

        # Timestamps and explicit scene numbers
        text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?', ' ', text)
        text = re.sub(r'\bscene\s*\d+\b', ' ', text, flags=re.IGNORECASE)

        # Normalize whitespace but preserve sentence structure
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = text.strip()

        return text

    @staticmethod
    def extract_features(raw_text):
        """Extract rich numerical features from script."""
        features = {}

        # === Basic Statistics ===
        features['char_count'] = len(raw_text)
        words = raw_text.split()
        features['word_count'] = len(words)
        lines = raw_text.split('\n')
        features['line_count'] = len(lines)

        # === Vocabulary Complexity ===
        if words:
            features['avg_word_length'] = np.mean([len(w) for w in words])
            features['unique_word_ratio'] = len(set(words)) / len(words)
            # Long words (8+ chars) ratio - indicates sophisticated vocabulary
            long_words = [w for w in words if len(w) >= 8]
            features['long_word_ratio'] = len(long_words) / len(words)
        else:
            features['avg_word_length'] = 0
            features['unique_word_ratio'] = 0
            features['long_word_ratio'] = 0

        # === Sentence Structure ===
        sentences = re.split(r'[.!?]+', raw_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        features['sentence_count'] = len(sentences)
        if sentences:
            sent_lengths = [len(s.split()) for s in sentences]
            features['avg_sentence_length'] = np.mean(sent_lengths)
            features['sentence_length_std'] = np.std(sent_lengths)
        else:
            features['avg_sentence_length'] = 0
            features['sentence_length_std'] = 0

        # === Dialogue Analysis ===
        # Character name patterns (ALL CAPS followed by colon)
        character_lines = re.findall(r'^[A-Z][A-Z\s]+:', raw_text, re.MULTILINE)
        features['dialogue_density'] = len(character_lines) / max(len(lines), 1)

        # Unique characters speaking
        unique_chars = set([c.strip(':').strip() for c in character_lines])
        features['unique_characters'] = len(unique_chars)

        # === Emotional Indicators ===
        features['exclamation_ratio'] = raw_text.count('!') / max(features['word_count'], 1) * 100
        features['question_ratio'] = raw_text.count('?') / max(features['word_count'], 1) * 100

        # === Action/Direction Density ===
        action_brackets = len(re.findall(r'\[.*?\]', raw_text))
        action_parens = len(re.findall(r'\(.*?\)', raw_text))
        features['action_density'] = (action_brackets + action_parens) / max(features['word_count'], 1) * 100

        # === Script Structure Indicators ===
        # Scene headings (INT., EXT.)
        scene_headings = len(re.findall(r'\b(INT|EXT)\.?\s', raw_text, re.IGNORECASE))
        features['scene_count'] = scene_headings

        # Pacing: words per scene (higher = slower pacing)
        features['words_per_scene'] = features['word_count'] / max(scene_headings, 1)

        return features

