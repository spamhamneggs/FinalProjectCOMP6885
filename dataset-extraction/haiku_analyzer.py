"""
Unified Haiku Feature Extraction and Quality Analyzer (v2.1)

Comprehensive haiku evaluation combining structural, linguistic, semantic,
imagery, and literary-specific metrics.

Features:
- Structural: syllable counts, 5-7-5 pattern compliance
- Linguistic: POS distribution, verb tenses, adjective/adverb density
- Semantic: SBERT embeddings for coherence, sentiment analysis
- Imagery: concreteness, nature/sensory classification, imagery categories
- Literary: kireji (cut-point) detection, originality, semantic density
- Quality Metrics: comprehensive scoring with unified output structure
- Batch Processing: efficient dataset evaluation with caching

Usage:
    analyzer = HaikuAnalyzer()
    scores = analyzer.score_haiku(line1, line2, line3)  # Single haiku
    df_results = analyzer.evaluate_dataset(df)          # Batch processing
"""

import pandas as pd
from string import punctuation
import numpy as np
import nltk
import pyphen
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import math
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('wordnet_ic', quiet=True)

# ============================================================================
# CONSTANTS & LOOKUP TABLES
# ============================================================================

# WordNet domain keywords for sensory/imagery classification
# These are used to identify synsets that belong to sensory modalities
SENSORY_DOMAIN_KEYWORDS = {
    'visual': [
        'color', 'light', 'bright', 'dark', 'shadow', 'glow', 'shine', 'glitter',
        'shape', 'form', 'image', 'sight', 'see', 'look', 'view', 'visual',
        'hue', 'shade', 'brilliance', 'luminosity'
    ],
    'auditory': [
        'sound', 'noise', 'silence', 'quiet', 'loud', 'roar', 'whisper', 'murmur',
        'music', 'sing', 'song', 'bell', 'chime', 'ring', 'melody', 'rhythm',
        'hear', 'listen', 'acoustic', 'sonic', 'phonetic', 'audio'
    ],
    'tactile': [
        'soft', 'hard', 'rough', 'smooth', 'texture', 'touch', 'feel', 'cold',
        'warm', 'hot', 'temperature', 'stroke', 'brush', 'graze', 'caress',
        'pressure', 'contact', 'tangible', 'palpable', 'tactile'
    ],
    'gustatory': [
        'taste', 'sweet', 'bitter', 'sour', 'salty', 'flavor', 'tongue',
        'palate', 'savory', 'delicious', 'bland', 'spicy', 'savor', 'sip'
    ],
    'olfactory': [
        'smell', 'scent', 'fragrance', 'odor', 'aroma', 'perfume', 'stench',
        'musty', 'nose', 'olfactory', 'sniff', 'breathe', 'inhale'
    ],
}

# Sensory verbs that enhance imagery
SENSORY_VERBS = {
    'glitter', 'gleam', 'glimmer', 'sparkle', 'shine', 'glow', 'shimmer',  # visual
    'rustle', 'whisper', 'murmur', 'chirp', 'buzz', 'ring', 'chime',  # auditory
    'caress', 'stroke', 'brush', 'graze', 'touch', 'clasp', 'grasp',  # tactile
    'taste', 'savor', 'sip', 'suck', 'chew',  # gustatory
    'smell', 'sniff', 'inhale', 'breathe',  # olfactory
}

# Natural/environmental nouns
NATURE_NOUNS = {
    'sky', 'cloud', 'sun', 'moon', 'star', 'rain', 'wind', 'snow', 'ice', 'water',
    'tree', 'flower', 'grass', 'leaf', 'leaves', 'branch', 'root', 'seed', 'bird',
    'fish', 'insect', 'butterfly', 'bee', 'beetle', 'stream', 'river', 'lake', 'ocean',
    'mountain', 'valley', 'hill', 'rock', 'stone', 'sand', 'soil', 'fire', 'lightning',
    'dawn', 'dusk', 'sunset', 'sunrise', 'autumn', 'spring', 'summer', 'winter', 'night',
    'forest', 'meadow', 'garden', 'path', 'road', 'shadow', 'light', 'darkness',
}


@dataclass
class HaikuQualityScores:
    """Unified, normalized haiku quality metrics (all 0.0-1.0 range)."""

    # Structural metrics
    syllable_accuracy: float = 0.0
    follows_575_pattern: bool = False

    # Coherence & juxtaposition
    semantic_coherence_sbert: float = 0.0  # embedding-based
    line1_line2_coherence: float = 0.0
    line2_line3_coherence: float = 0.0
    kireji_strength: float = 0.0  # cut-point/juxtaposition strength

    # Imagery
    imagery_concreteness: float = 0.0
    imagery_visual: float = 0.0
    imagery_auditory: float = 0.0
    imagery_tactile: float = 0.0
    imagery_gustatory: float = 0.0
    imagery_olfactory: float = 0.0
    nature_score: float = 0.0
    sensory_verb_emphasis: float = 0.0

    # Linguistic
    lexical_diversity: float = 0.0
    adjective_density: float = 0.0
    verb_density: float = 0.0
    noun_density: float = 0.0
    sensory_verb_density: float = 0.0

    # Semantic
    sentiment_balance: float = 0.0
    overall_sentiment: float = 0.0

    # Semantic density
    semantic_density: float = 0.0

    # Originality (if reference corpus available)
    originality_score: float = 0.0

    # Composite score
    composite_haiku_quality_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)

    def as_vector(self) -> np.ndarray:
        """Return a numeric vector of the main float features for ML/training.

        Order is fixed and includes booleans as floats.
        """
        vals = [
            float(self.syllable_accuracy),
            1.0 if self.follows_575_pattern else 0.0,
            float(self.semantic_coherence_sbert),
            float(self.line1_line2_coherence),
            float(self.line2_line3_coherence),
            float(self.kireji_strength),
            float(self.imagery_concreteness),
            float(self.imagery_visual),
            float(self.imagery_auditory),
            float(self.imagery_tactile),
            float(self.imagery_gustatory),
            float(self.imagery_olfactory),
            float(self.nature_score),
            float(self.sensory_verb_emphasis),
            float(self.lexical_diversity),
            float(self.adjective_density),
            float(self.verb_density),
            float(self.noun_density),
            float(self.sensory_verb_density),
            float(self.sentiment_balance),
            float(self.overall_sentiment),
            float(self.semantic_density),
            float(self.originality_score),
            float(self.composite_haiku_quality_score),
        ]
        return np.asarray(vals, dtype=float)


class HaikuAnalyzer:
    """Comprehensive haiku analyzer combining structural, linguistic, semantic, and quality metrics."""

    def __init__(self, target_syllables: Tuple[int, int, int] = (5, 7, 5)):
        """
        Initialize the HaikuAnalyzer.

        Args:
            target_syllables: Tuple of target syllable counts (default: 5-7-5)
        """
        self.target_syllables = target_syllables
        self.dic = pyphen.Pyphen(lang='en')
        # VADER removed because it produced noisy weighting.
        # Sentiment will be computed using SBERT-based positive/negative prototypes.
        self.stop_words = set(stopwords.words('english'))
        # Lemmatizer for robust matching (plurals, inflections)
        self.lem = WordNetLemmatizer()

        # Load SBERT model for semantic coherence
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Prepare SBERT-based sentiment prototypes (positive / negative seeds)
        try:
            pos_seeds = ["good", "happy", "joyful", "pleasant", "wonderful"]
            neg_seeds = ["bad", "sad", "angry", "terrible", "awful"]
            pos_embs = self._encode(pos_seeds, show_progress_bar=False)
            neg_embs = self._encode(neg_seeds, show_progress_bar=False)
            pos_embs = np.asarray(pos_embs)
            neg_embs = np.asarray(neg_embs)
            self._sentiment_positive_emb = np.mean(pos_embs, axis=0)
            self._sentiment_negative_emb = np.mean(neg_embs, axis=0)

            # Prepare SBERT-based subjectivity prototypes
            subj_seeds = ["emotional", "personal", "uncertain", "intimate", "subjective", "opinion"]
            obj_seeds = ["factual", "neutral", "observation", "objective", "descriptive", "report"]
            subj_embs = self._encode(subj_seeds, show_progress_bar=False)
            obj_embs = self._encode(obj_seeds, show_progress_bar=False)
            subj_embs = np.asarray(subj_embs)
            obj_embs = np.asarray(obj_embs)
            self._subjectivity_prototype_emb = np.mean(subj_embs, axis=0)
            self._objectivity_prototype_emb = np.mean(obj_embs, axis=0)
        except Exception:
            self._sentiment_positive_emb = None
            self._sentiment_negative_emb = None
            self._subjectivity_prototype_emb = None
            self._objectivity_prototype_emb = None

        # Cache for word concreteness scores
        self._concreteness_cache = {}
        self._brysbaert_data = None
        # Cache for WordNet synsets and combined definition texts
        self._synset_cache = {}
        self._synset_text_cache = {}

        # Load Brysbaert concreteness dataset if available
        self._load_brysbaert_concreteness()

        # Reference corpus for originality (will be populated if available)
        self._reference_embeddings = None
        self._reference_haikus = []

    def _load_brysbaert_concreteness(self):
        """Load Brysbaert et al. concreteness ratings from xlsx file."""
        try:
            cwd = Path.cwd()
            xlsx_path = cwd / 'vendor/Concreteness_ratings_Brysbaert_et_al_BRM.xlsx'

            if xlsx_path.exists():
                df = pd.read_excel(xlsx_path)
            else:
                df = None

            if df is not None:
                # Build dict: word -> concreteness (0-5 scale, normalize to 0-1)
                if 'Word' in df.columns and ('Conc.M' in df.columns or 'Conc.Mean' in df.columns):
                    conc_col = 'Conc.M' if 'Conc.M' in df.columns else 'Conc.Mean'
                    self._brysbaert_data = {}
                    for _, row in df.iterrows():
                        try:
                            word = str(row['Word']).lower()
                            concreteness = float(row[conc_col]) / 5.0  # Normalize to 0-1
                            self._brysbaert_data[word] = concreteness
                        except Exception:
                            continue
        except Exception as e:
            print(f"Warning: Could not load Brysbaert concreteness dataset: {e}")

    # ------------------------------------------------------------------
    # SBERT compatibility wrapper
    # ------------------------------------------------------------------
    def _encode(self, texts, **kwargs):
        """
        Encode texts using SBERT with forward-compatible API. Tries new
        `output_value="sentence_embedding"` first, falls back to
        `convert_to_numpy=True` for older versions.
        """
        try:
            return self.sbert_model.encode(texts, output_value="sentence_embedding", **kwargs)
        except TypeError:
            return self.sbert_model.encode(texts, convert_to_numpy=True, **kwargs)

    # ------------------------------------------------------------------
    # WordNet cached helpers
    # ------------------------------------------------------------------
    def _get_synsets_cached(self, word: str):
        """Return synsets for `word`, using an in-memory cache."""
        key = word.lower()
        if key in self._synset_cache:
            return self._synset_cache[key]
        syns = wordnet.synsets(key)
        self._synset_cache[key] = syns
        return syns

    def _get_synset_combined_text(self, word: str) -> str:
        """Return combined definition + lemma text for a word (cached)."""
        key = word.lower()
        if key in self._synset_text_cache:
            return self._synset_text_cache[key]
        syns = self._get_synsets_cached(key)
        if not syns:
            self._synset_text_cache[key] = ""
            return ""
        synset = syns[0]
        definition = synset.definition().lower()
        lemma_names = ' '.join([lem.name() for lem in synset.lemmas()]).lower()
        combined = f"{definition} {lemma_names}"
        self._synset_text_cache[key] = combined
        return combined

    # ------------------------------
    # SBERT sentiment helper
    # ------------------------------
    def _sentiment_score_from_embedding(self, emb: np.ndarray) -> float:
        """
        Compute a simple sentiment score from an embedding by comparing
        cosine similarity to positive and negative prototype embeddings.

        Returns value in [-1, 1] (positive = more positive sentiment).
        """
        try:
            if emb is None:
                return 0.0
            if isinstance(emb, (list, tuple)):
                emb = np.asarray(emb)
            emb = np.asarray(emb).ravel()
            if self._sentiment_positive_emb is None or self._sentiment_negative_emb is None:
                return 0.0
            pos = float(np.dot(emb, self._sentiment_positive_emb) /
                        (np.linalg.norm(emb) * np.linalg.norm(self._sentiment_positive_emb) + 1e-8))
            neg = float(np.dot(emb, self._sentiment_negative_emb) /
                        (np.linalg.norm(emb) * np.linalg.norm(self._sentiment_negative_emb) + 1e-8))
            return float(pos - neg)
        except Exception:
            return 0.0

    def _subjectivity_score_from_embedding(self, emb: np.ndarray) -> float:
        """
        Compute a subjectivity score from an embedding by comparing
        cosine similarity to subjectivity and objectivity prototype embeddings.

        Returns value in [0, 1] (higher = more subjective).
        """
        try:
            if emb is None:
                return 0.5 # Neutral
            if isinstance(emb, (list, tuple)):
                emb = np.asarray(emb)
            emb = np.asarray(emb).ravel()
            if self._subjectivity_prototype_emb is None or self._objectivity_prototype_emb is None:
                return 0.5 # Neutral

            subj_sim = float(np.dot(emb, self._subjectivity_prototype_emb) /
                         (np.linalg.norm(emb) * np.linalg.norm(self._subjectivity_prototype_emb) + 1e-8))
            obj_sim = float(np.dot(emb, self._objectivity_prototype_emb) /
                        (np.linalg.norm(emb) * np.linalg.norm(self._objectivity_prototype_emb) + 1e-8))

            # Sigmoid squashing function to map to [0, 1]
            def sigmoid(x):
                return 1 / (1 + math.exp(-x))

            # Difference in similarities, scaled to adjust sensitivity
            subjectivity = sigmoid((subj_sim - obj_sim) * 5)
            return float(subjectivity)
        except Exception:
            return 0.5 # Neutral on error

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def _get_wordnet_concreteness(self, word: str) -> float:
        """
        Determine concreteness of a word using Brysbaert dataset or WordNet.

        Brysbaert et al. (2014) ratings are preferred when available.
        Falls back to WordNet-based heuristic.

        Args:
            word: The word to score

        Returns:
            Concreteness score (0.0 = abstract, 1.0 = concrete)
        """
        if word in self._concreteness_cache:
            return self._concreteness_cache[word]

        word_lower = word.lower().strip('.,!?;:')

        # Try Brysbaert dataset first
        if self._brysbaert_data is not None and word_lower in self._brysbaert_data:
            score = self._brysbaert_data[word_lower]
            self._concreteness_cache[word] = score
            return score

        # Fall back to WordNet heuristic
        synsets = self._get_synsets_cached(word_lower)

        if not synsets:
            # Unknown words default to neutral
            score = 0.5
        else:
            # Analyze the most common synset (first one)
            synset = synsets[0]

            # Check POS (part of speech)
            pos = synset.pos()

            # Get hypernyms to check if word refers to concrete things
            hypernyms = synset.hypernyms()
            all_hypernym_names = set()
            for hyp in hypernyms:
                all_hypernym_names.add(hyp.name())

            # Concrete categories (hypernyms)
            concrete_hypernyms = {
                'entity.n.01', 'object.n.01', 'physical_entity.n.01',
                'living_thing.n.01', 'organism.n.01', 'animal.n.01',
                'plant.n.01', 'artifact.n.01', 'natural_object.n.01',
            }

            # Abstract categories (hypernyms)
            abstract_hypernyms = {
                'abstraction.n.01', 'concept.n.01', 'idea.n.01',
                'attribute.n.01', 'quality.n.01', 'property.n.01',
                'state.n.01', 'condition.n.01', 'emotion.n.01',
                'feeling.n.01', 'thought.n.01',
            }

            # Check hypernym overlap
            concrete_overlap = len(all_hypernym_names & concrete_hypernyms)
            abstract_overlap = len(all_hypernym_names & abstract_hypernyms)

            # Score based on POS and hypernyms
            if pos == 'n':  # Noun
                if concrete_overlap > 0:
                    score = 0.85  # Concrete noun
                elif abstract_overlap > 0:
                    score = 0.25  # Abstract noun
                else:
                    score = 0.6  # Neutral noun
            elif pos == 'a' or pos == 'r':  # Adjective or adverb
                if concrete_overlap > 0:
                    score = 0.75  # Sensory adjective
                elif abstract_overlap > 0:
                    score = 0.3  # Abstract adjective
                else:
                    score = 0.55  # Neutral adjective
            elif pos == 'v':  # Verb
                # Action verbs are more concrete than state verbs
                if concrete_overlap > 0:
                    score = 0.7  # Action verb
                elif abstract_overlap > 0:
                    score = 0.35  # State verb
                else:
                    score = 0.55  # Neutral verb
            else:  # Other POS
                score = 0.5

            # Adjust based on concreteness vs. abstractness balance
            if abstract_overlap > concrete_overlap:
                score = max(0.0, score - 0.2)
            elif concrete_overlap > abstract_overlap:
                score = min(1.0, score + 0.15)

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        self._concreteness_cache[word] = score

        return score

    def count_syllables(self, word: str) -> int:
        """Count syllables in a word using pyphen."""
        word = word.lower().strip(punctuation)
        if not word:
            return 0
        syllabified = self.dic.inserted(word)
        return max(1, len(syllabified.split('-')))

    def _handle_nan_lines(self, line1, line2, line3) -> bool:
        """Check if any lines are NaN."""
        return pd.isna(line1) or pd.isna(line2) or pd.isna(line3)

    # ============================================================================
    # STRUCTURAL FEATURES
    # ============================================================================

    def extract_structural_features(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """Extract structural features: syllable counts and 5-7-5 pattern."""
        if self._handle_nan_lines(line1, line2, line3):
            return {
                'line1_syllables': 0, 'line2_syllables': 0, 'line3_syllables': 0,
                'total_syllables': 0, 'follows_575_pattern': False, 'total_words': 0,
            }

        lines = [line1, line2, line3]
        syllable_counts = []
        total_syllables = 0

        for line in lines:
            words = line.split()
            line_syllables = sum(self.count_syllables(word) for word in words)
            syllable_counts.append(line_syllables)
            total_syllables += line_syllables

        total_words = sum(len(line.split()) for line in lines)
        follows_575 = (syllable_counts == [5, 7, 5])

        return {
            'line1_syllables': syllable_counts[0],
            'line2_syllables': syllable_counts[1],
            'line3_syllables': syllable_counts[2],
            'total_syllables': total_syllables,
            'follows_575_pattern': follows_575,
            'total_words': total_words,
        }

    # ============================================================================
    # LINGUISTIC FEATURES
    # ============================================================================

    def extract_linguistic_features(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """Extract linguistic features: POS distribution, verb tense, adjective density."""
        if self._handle_nan_lines(line1, line2, line3):
            return {
                'noun_count': 0, 'verb_count': 0, 'adjective_count': 0, 'adverb_count': 0,
                'adjective_density': 0, 'verb_density': 0, 'noun_density': 0,
                'past_tense_count': 0, 'present_tense_count': 0, 'gerund_count': 0,
            }

        full_text = f"{line1} {line2} {line3}"
        words = word_tokenize(full_text)
        pos_tags = pos_tag(words)

        pos_counts = {}
        for word, tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1

        total_tags = len(pos_tags)
        noun_count = pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)
        verb_count = sum(pos_counts.get(tag, 0) for tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        adjective_count = sum(pos_counts.get(tag, 0) for tag in ['JJ', 'JJR', 'JJS'])
        adverb_count = sum(pos_counts.get(tag, 0) for tag in ['RB', 'RBR', 'RBS'])

        past_tense = pos_counts.get('VBD', 0)
        present_tense = pos_counts.get('VB', 0) + pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)
        gerund = pos_counts.get('VBG', 0)

        return {
            'noun_count': noun_count,
            'verb_count': verb_count,
            'adjective_count': adjective_count,
            'adverb_count': adverb_count,
            'adjective_density': adjective_count / total_tags if total_tags > 0 else 0,
            'verb_density': verb_count / total_tags if total_tags > 0 else 0,
            'noun_density': noun_count / total_tags if total_tags > 0 else 0,
            'past_tense_count': past_tense,
            'present_tense_count': present_tense,
            'gerund_count': gerund,
        }

    def extract_contrast_features(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """Extract contrast/antonym features using WordNet."""
        if self._handle_nan_lines(line1, line2, line3):
            return {'has_contrast': False, 'contrast_pairs': 0}

        full_text = f"{line1} {line2} {line3}"
        text_lower = full_text.lower()
        words_in_text = set(word_tokenize(text_lower))

        contrasts_found = set()
        for word in words_in_text:
            synsets = self._get_synsets_cached(word)
            for synset in synsets:
                for lemma in synset.lemmas():
                    if lemma.antonyms():
                        for antonym in lemma.antonyms():
                            if antonym.name() in text_lower:
                                contrasts_found.add(f"{word}-{antonym.name()}")

        return {
            'has_contrast': len(contrasts_found) > 0,
            'contrast_pairs': len(contrasts_found),
        }

    # ============================================================================
    # SEMANTIC FEATURES
    # ============================================================================

    def extract_semantic_features(
        self,
        line1: str,
        line2: str,
        line3: str,
        current_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Extract semantic features: sentiment, polarity, subjectivity."""
        if self._handle_nan_lines(line1, line2, line3):
            return {
                'sentiment_positive': 0, 'sentiment_negative': 0, 'sentiment_neutral': 0,
                'polarity': 0, 'subjectivity': 0.5,
            }

        full_text = f"{line1} {line2} {line3}"
        # Use SBERT-based prototype similarity for sentiment and subjectivity
        try:
            if current_embedding is not None:
                emb = current_embedding
                if isinstance(emb, (list, tuple)):
                    emb = np.asarray(emb)[0]
            else:
                emb = self._encode(str(full_text).strip())
                if isinstance(emb, (list, tuple)):
                    emb = np.asarray(emb)[0]

            polarity = float(self._sentiment_score_from_embedding(emb))
            subjectivity = float(self._subjectivity_score_from_embedding(emb))

            # Normalize component scores to [0,1] for positive/negative proxies
            pos_norm = max(0.0, min(1.0, (polarity + 1.0) / 2.0))
            neg_norm = max(0.0, min(1.0, (1.0 - polarity) / 2.0))
            neutral = max(0.0, 1.0 - abs(polarity))

            return {
                'sentiment_positive': pos_norm,
                'sentiment_negative': neg_norm,
                'sentiment_neutral': neutral,
                'polarity': polarity,
                'subjectivity': subjectivity,
            }
        except Exception:
            return {
                'sentiment_positive': 0.0,
                'sentiment_negative': 0.0,
                'sentiment_neutral': 0.0,
                'polarity': 0.0,
                'subjectivity': 0.5, # Neutral
            }

    # ============================================================================
    # NEW: SEMANTIC EMBEDDINGS & KIREJI (Cut-Point) DETECTION
    # ============================================================================

    def score_semantic_coherence_sbert(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """
        Score semantic coherence using SBERT embeddings + cosine similarity.

        Replaces naive word-overlap with proper sentence embeddings.
        Matches haiku coherence patterns better.
        """
        lines = [line1, line2, line3]

        if any(pd.isna(line) for line in lines):
            return {
                'semantic_coherence_sbert': 0.0,
                'line1_line2_cosine': 0.0,
                'line2_line3_cosine': 0.0,
                'kireji_strength': 0.0,
                'contrast_shift': 0.0,
            }

        # Encode lines using compatibility wrapper
        try:
            embeddings = self._encode([str(line).strip() for line in lines])
        except Exception:
            return {
                'semantic_coherence_sbert': 0.0,
                'line1_line2_cosine': 0.0,
                'line2_line3_cosine': 0.0,
                'kireji_strength': 0.0,
                'contrast_shift': 0.0,
            }

        embeddings = np.asarray(embeddings)
        if embeddings.ndim == 1:
            # Single vector returned incorrectly
            return {
                'semantic_coherence_sbert': 0.0,
                'line1_line2_cosine': 0.0,
                'line2_line3_cosine': 0.0,
                'kireji_strength': 0.0,
                'contrast_shift': 0.0,
            }

        # Ensure we have three embeddings
        if embeddings.shape[0] < 3:
            return {
                'semantic_coherence_sbert': 0.0,
                'line1_line2_cosine': 0.0,
                'line2_line3_cosine': 0.0,
                'kireji_strength': 0.0,
                'contrast_shift': 0.0,
            }

        # Norm checks to avoid cosine noise
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms < 1e-6):
            return {
                'semantic_coherence_sbert': 0.0,
                'line1_line2_cosine': 0.0,
                'line2_line3_cosine': 0.0,
                'kireji_strength': 0.0,
                'contrast_shift': 0.0,
            }

        def cosine(a, b, na, nb):
            return float(np.dot(a, b) / (na * nb + 1e-6))

        coh_12_raw = cosine(embeddings[0], embeddings[1], norms[0], norms[1])
        coh_23_raw = cosine(embeddings[1], embeddings[2], norms[1], norms[2])
        # coh_13_raw kept out as it's not used for current measures

        # Scale cosine from [-1,1] to [0,1] to preserve direction while making positive
        coh_12 = max(0.0, min(1.0, (coh_12_raw + 1.0) / 2.0))
        coh_23 = max(0.0, min(1.0, (coh_23_raw + 1.0) / 2.0))
        # coh_13 not used in current measures but could be retained later

        # Contrastive image shift: use angles between embeddings
        angle_12 = float(np.arccos(np.clip(coh_12_raw, -1.0, 1.0)))
        angle_23 = float(np.arccos(np.clip(coh_23_raw, -1.0, 1.0)))
        contrast_shift = max(0.0, (angle_23 - angle_12) / math.pi)

        # Kireji strength: detect a meaningful cut between images
        # Use *difference* between coherence scores, not variance
        cut_raw = abs(coh_12 - coh_23)

        # Filter out noise: if both coherences are very low, no real kireji exists
        noise_floor = min(coh_12, coh_23)
        if noise_floor < 0.1:
            kireji_score = cut_raw * 0.3  # heavily reduce false positives
        else:
            kireji_score = cut_raw

        # Clamp to [0, 1]
        kireji_score = float(max(0.0, min(1.0, kireji_score)))

        overall_coherence = (coh_12 + coh_23) / 2.0

        return {
            'semantic_coherence_sbert': max(0.0, min(1.0, overall_coherence)),
            'line1_line2_cosine': coh_12,
            'line2_line3_cosine': coh_23,
            'kireji_strength': kireji_score,
            'contrast_shift': max(0.0, min(1.0, contrast_shift)),
        }

    # ============================================================================
    # NEW: IMAGERY CATEGORIZATION
    # ============================================================================

    def extract_imagery_categories(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """
        Classify imagery types using WordNet synset analysis.

        Inspects synsets for sensory domain keywords to determine:
        visual, auditory, tactile, gustatory, olfactory.

        Returns density scores for each category (0.0-1.0).
        """
        full_text = f"{line1} {line2} {line3}"
        words = [w.lower() for w in word_tokenize(full_text) if w.isalpha()]

        if not words:
            return {
                'imagery_visual': 0.0,
                'imagery_auditory': 0.0,
                'imagery_tactile': 0.0,
                'imagery_gustatory': 0.0,
                'imagery_olfactory': 0.0,
            }

        category_counts = {cat: 0 for cat in SENSORY_DOMAIN_KEYWORDS.keys()}

        # Analyze each word's synsets for sensory domains
        for word in words:
            # Use cached combined synset text to speed up lookups
            combined_text = self._get_synset_combined_text(word)
            if not combined_text:
                continue

            # Count all matching categories (a word can evoke multiple senses)
            for category, keywords in SENSORY_DOMAIN_KEYWORDS.items():
                if any(kw in combined_text for kw in keywords):
                    category_counts[category] += 1

        total_words = len(words)
        category_scores = {
            f'imagery_{cat}': min(1.0, count / max(1, total_words))
            for cat, count in category_counts.items()
        }

        return category_scores

    # ============================================================================
    # NEW: NATURE & SENSORY NOUN DETECTION
    # ============================================================================

    def score_nature_imagery(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """
        Score nature-specific imagery and sensory verb emphasis.

        Haiku traditionally feature natural objects and seasonal words (kigo).
        Detect: natural nouns, sensory verbs.
        """
        full_text = f"{line1} {line2} {line3}"
        raw_words = [w for w in word_tokenize(full_text) if w.isalpha()]
        words = [w.lower().strip('.,!?;:') for w in raw_words]

        if not words:
            return {
                'nature_score': 0.0,
                'sensory_verb_density': 0.0,
                'sensory_verb_emphasis': 0.0,
            }

        # Lemmatize nouns and verbs for robust matching
        nature_count = 0
        sensory_verb_count = 0
        for w in words:
            lemma_n = self.lem.lemmatize(w, 'n')
            if lemma_n in NATURE_NOUNS:
                nature_count += 1
            lemma_v = self.lem.lemmatize(w, 'v')
            if lemma_v in SENSORY_VERBS:
                sensory_verb_count += 1

        nature_score = min(1.0, nature_count / max(1, len(words)))
        sensory_verb_density = min(1.0, sensory_verb_count / max(1, len(words)))

        # Sensory verb emphasis: if present, boost; prefer non-zero
        sensory_emphasis = min(1.0, sensory_verb_count / max(1, len(words)) * 2.0)

        return {
            'nature_score': nature_score,
            'sensory_verb_density': sensory_verb_density,
            'sensory_verb_emphasis': sensory_emphasis,
        }

    # ============================================================================
    # NEW: SEMANTIC DENSITY (Information Density)
    # ============================================================================

    def score_semantic_density(
        self,
        line1: str,
        line2: str,
        line3: str,
        current_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Score semantic density: information content per token.

        Haiku quality correlates with semantic density.
        Uses inverse entropy approximation and compression ratio concept.

        High density = rare word combinations, fewer stopwords.
        """
        full_text = f"{line1} {line2} {line3}"

        if not full_text.strip():
            return {'semantic_density': 0.0}
        # New approach: density = 1 - normalized_entropy(PCA-projection of embedding)
        try:
            if current_embedding is not None:
                emb = current_embedding
                if isinstance(emb, (list, tuple)):
                    emb = np.asarray(emb).ravel()
            else:
                emb = self._encode(full_text)
                if isinstance(emb, (list, tuple)):
                    emb = np.asarray(emb)[0]
                emb = np.asarray(emb).ravel()

            # If a reference corpus of embeddings is available, fit PCA on it
            if self._reference_embeddings is not None and getattr(self._reference_embeddings, 'shape', (0,))[0] > 1:
                ref = np.asarray(self._reference_embeddings)
                # Center reference and compute PCA components via SVD
                try:
                    mean = np.mean(ref, axis=0)
                    ref_c = ref - mean
                    # Compute truncated SVD/PCA
                    u, s, vt = np.linalg.svd(ref_c, full_matrices=False)
                    components = vt  # shape (k, dim)
                    # Project current embedding (centered) onto components
                    v = emb - mean
                    scores = np.dot(components, v)
                    coeffs = np.abs(scores)
                    if coeffs.sum() <= 0:
                        return {'semantic_density': 0.0}
                    p = coeffs / coeffs.sum()
                except Exception:
                    # fallback to per-dimension distribution
                    coeffs = np.abs(emb)
                    if coeffs.sum() <= 0:
                        return {'semantic_density': 0.0}
                    p = coeffs / coeffs.sum()
            else:
                # No reference corpus: use absolute embedding component magnitudes
                coeffs = np.abs(emb)
                if coeffs.sum() <= 0:
                    return {'semantic_density': 0.0}
                p = coeffs / coeffs.sum()

            # Compute entropy of distribution p and normalize
            eps = 1e-12
            p_safe = p + eps
            H = -float(np.sum(p_safe * np.log(p_safe)))
            H_norm = H / float(np.log(len(p_safe))) if len(p_safe) > 1 else 0.0
            density = 1.0 - H_norm
            density = max(0.0, min(1.0, density))
            return {'semantic_density': density}
        except Exception:
            # Fallback lexical proxy when encoding fails
            words = word_tokenize(full_text)
            alpha_words = [w.lower() for w in words if w.isalpha()]
            if not alpha_words:
                return {'semantic_density': 0.0}
            stopword_ratio = sum(1 for w in alpha_words if w in self.stop_words) / len(alpha_words)
            vocab_richness = len(set(alpha_words)) / len(alpha_words)
            density = (vocab_richness * 0.7) + ((1.0 - stopword_ratio) * 0.3)
            return {'semantic_density': max(0.0, min(1.0, density))}

    # ============================================================================
    # NEW: ORIGINALITY / NOVELTY DETECTION
    # ============================================================================

    def score_originality(
        self,
        line1: str,
        line2: str,
        line3: str,
        current_embedding: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Score originality using reference corpus embeddings (if available).

        Requires reference corpus to be set via load_reference_corpus().
        Otherwise returns neutral score.
        """
        if self._reference_embeddings is None or len(self._reference_embeddings) == 0:
            return {'originality_score': 0.5}  # Neutral if no reference

        full_text = f"{line1} {line2} {line3}".strip()
        if not full_text:
            return {'originality_score': 0.0}

        # Use provided embedding if given (precomputed), otherwise encode
        if current_embedding is None:
            current_embedding = self._encode(full_text)
            if isinstance(current_embedding, (list, tuple)):
                current_embedding = np.asarray(current_embedding)[0]

        # Find nearest neighbor in reference corpus
        ref_norms = np.linalg.norm(self._reference_embeddings, axis=1)
        cur_norm = np.linalg.norm(current_embedding)
        if cur_norm < 1e-6:
            return {'originality_score': 0.5}
        similarities = np.dot(self._reference_embeddings, current_embedding) / (
            ref_norms * cur_norm + 1e-6
        )

        max_similarity = np.max(similarities)

        # Originality = 1 - max_similarity (more different = more original)
        originality = 1.0 - max_similarity

        return {'originality_score': max(0.0, min(1.0, originality))}

    def load_reference_corpus(self, haiku_list: List[str], batch_size: Optional[int] = None):
        """
        Load reference haiku corpus for originality scoring.

        Args:
            haiku_list: List of reference haiku texts (or formatted line1 line2 line3 strings)
        """
        if not haiku_list:
            return

        try:
            self._reference_haikus = haiku_list
            # Allow batched encoding for speed; default uses _encode wrapper
            # Allow specifying batch size for large corpora
            encode_kwargs = {'show_progress_bar': False}
            if batch_size is not None:
                encode_kwargs['batch_size'] = int(batch_size)
            self._reference_embeddings = self._encode(
                haiku_list,
                **encode_kwargs
            )
            # Convert to numpy array if needed
            self._reference_embeddings = np.asarray(self._reference_embeddings)
        except Exception as e:
            print(f"Warning: Could not load reference corpus: {e}")



    # ============================================================================
    # QUALITY METRICS (Scoring)
    # ============================================================================

    def score_syllable_accuracy(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """Score syllable accuracy for target pattern (default 5-7-5)."""
        lines = [line1, line2, line3]
        line_syllable_counts = []
        line_scores = []

        for line, target in zip(lines, self.target_syllables):
            if pd.isna(line) or not str(line).strip():
                line_syllable_counts.append(0)
                line_scores.append(0.0)
                continue

            words = word_tokenize(str(line))
            syllable_count = sum(self.count_syllables(w) for w in words)
            line_syllable_counts.append(syllable_count)

            deviation = abs(syllable_count - target)
            score = max(0.0, 1.0 - (deviation / target))
            line_scores.append(score)

        return {
            'line1_syllables': line_syllable_counts[0],
            'line2_syllables': line_syllable_counts[1],
            'line3_syllables': line_syllable_counts[2],
            'line1_score': line_scores[0],
            'line2_score': line_scores[1],
            'line3_score': line_scores[2],
            'overall_syllable_accuracy': np.mean(line_scores),
        }

    def score_imagery_concreteness(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """Score imagery and concreteness using WordNet synsets."""
        lines = [line1, line2, line3]
        concreteness_scores = []

        for line in lines:
            if pd.isna(line) or not str(line).strip():
                concreteness_scores.append(0.0)
                continue

            words = [w.lower() for w in word_tokenize(str(line)) if w.isalpha()]
            if not words:
                concreteness_scores.append(0.0)
                continue

            word_scores = []
            for word in words:
                score = self._get_wordnet_concreteness(word)
                word_scores.append(score)

            concreteness_scores.append(np.mean(word_scores))

        return {
            'line1_concreteness': concreteness_scores[0],
            'line2_concreteness': concreteness_scores[1],
            'line3_concreteness': concreteness_scores[2],
            'overall_imagery_concreteness': np.mean(concreteness_scores),
        }

    def score_lexical_diversity(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """Score lexical diversity using Type-Token Ratio (TTR)."""
        text = ' '.join(str(x) for x in [line1, line2, line3] if pd.notna(x))

        if not text.strip():
            return {'unique_words': 0, 'total_words': 0, 'ttr': 0.0, 'lexical_diversity': 0.0}

        words = [w.lower() for w in word_tokenize(text) if w.isalpha()]

        if not words:
            return {'unique_words': 0, 'total_words': 0, 'ttr': 0.0, 'lexical_diversity': 0.0}

        unique_words = len(set(words))
        total_words = len(words)

        # Herdan's C = log(V) / log(N) (less biased for short texts)
        if total_words <= 1 or unique_words <= 1:
            herdan_c = 0.0
        else:
            herdan_c = math.log(unique_words) / math.log(total_words)

        # Normalize Herdan's C via tanh to bound [0,1]
        lexical_div = float(np.tanh(herdan_c))

        return {
            'unique_words': unique_words,
            'total_words': total_words,
            'ttr': unique_words / total_words if total_words > 0 else 0.0,
            'lexical_diversity': max(0.0, min(1.0, lexical_div)),
        }

    def score_sentiment_balance(
        self,
        line1: str,
        line2: str,
        line3: str,
        empirical_mean: Optional[float] = None,
        full_haiku_embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Score sentiment balance using SBERT prototype sentiment analysis.

        Args:
            line1, line2, line3: Haiku lines
            empirical_mean: Empirical mean sentiment from training corpus.
                           If None, uses default 0.05 (slightly positive).
            full_haiku_embedding: Precomputed embedding of the full haiku text for overall sentiment.
        """
        lines = [line1, line2, line3]
        sentiments = []

        for line in lines:
            if pd.isna(line) or not str(line).strip():
                sentiments.append(0.0)
                continue
            try:
                emb = self._encode(str(line).strip())
                if isinstance(emb, (list, tuple)):
                    emb = np.asarray(emb)[0]
                sentiments.append(self._sentiment_score_from_embedding(emb))
            except Exception:
                sentiments.append(0.0)

        # Calculate overall sentiment using the full haiku embedding if provided
        if full_haiku_embedding is not None:
            if isinstance(full_haiku_embedding, (list, tuple)):
                full_haiku_embedding = np.asarray(full_haiku_embedding)[0]
            overall_sentiment = self._sentiment_score_from_embedding(full_haiku_embedding)
        else:
            overall_sentiment = np.mean(sentiments)

        # Use empirical mean if provided, otherwise default
        ideal_sentiment = empirical_mean if empirical_mean is not None else 0.05

        # Score: deviation from ideal
        sentiment_balance = 1.0 - abs(overall_sentiment - ideal_sentiment) / 2.0
        sentiment_balance = max(0.0, min(1.0, sentiment_balance))

        return {
            'line1_sentiment': sentiments[0],
            'line2_sentiment': sentiments[1],
            'line3_sentiment': sentiments[2],
            'overall_sentiment': overall_sentiment,
            'sentiment_balance': sentiment_balance,
        }

    # ============================================================================
    # COMPOSITE SCORING
    # ============================================================================

    def score_haiku(
        self,
        line1: str,
        line2: str,
        line3: str,
        weights: Optional[Dict[str, float]] = None,
        empirical_sentiment_mean: Optional[float] = None,
        current_embedding: Optional[np.ndarray] = None,
    ) -> HaikuQualityScores:
        """
        Compute comprehensive haiku quality score using all metrics.

        Returns unified HaikuQualityScores dataclass with normalized (0-1) scores.

        Args:
            line1, line2, line3: Haiku lines
            weights: Optional weights for composite score (default: balanced)
            empirical_sentiment_mean: Empirical mean sentiment for baseline calibration

        Returns:
            HaikuQualityScores object with all metrics
        """
        if weights is None:
            weights = {
                'syllable_accuracy': 0.15,
                'semantic_coherence': 0.15,
                'kireji': 0.10,
                'imagery': 0.15,
                'nature': 0.10,
                'lexical_diversity': 0.15,
                'sentiment': 0.03,
                'density': 0.10,
            }

        # Normalize weights to sum to 1.0
        total_w = sum(weights.values()) if weights else 0.0
        if total_w > 0:
            weights = {k: float(v) / float(total_w) for k, v in weights.items()}

        # Structural
        struct_scores = self.score_syllable_accuracy(line1, line2, line3)
        syll_accuracy = struct_scores['overall_syllable_accuracy']
        # Correct follows_575: exact syllable equality
        follows_575 = (
            struct_scores.get('line1_syllables', 0) == self.target_syllables[0]
            and struct_scores.get('line2_syllables', 0) == self.target_syllables[1]
            and struct_scores.get('line3_syllables', 0) == self.target_syllables[2]
        )

        # New SBERT coherence & kireji
        coherence_sbert = self.score_semantic_coherence_sbert(line1, line2, line3)
        sbert_coh = coherence_sbert['semantic_coherence_sbert']
        kireji = coherence_sbert['kireji_strength']

        # Imagery
        imagery_cats = self.extract_imagery_categories(line1, line2, line3)
        img_concreteness = self.score_imagery_concreteness(line1, line2, line3)
        img_score = img_concreteness['overall_imagery_concreteness']

        # Nature & sensory
        nature_scores = self.score_nature_imagery(line1, line2, line3)
        nature_score = nature_scores['nature_score']
        sensory_emphasis = nature_scores['sensory_verb_emphasis']

        # Linguistic
        ling_scores = self.extract_linguistic_features(line1, line2, line3)

        # Lexical diversity
        div_scores = self.score_lexical_diversity(line1, line2, line3)
        lex_div = div_scores['lexical_diversity']

        # Semantic density
        density_scores = self.score_semantic_density(line1, line2, line3, current_embedding=current_embedding)
        density = density_scores['semantic_density']

        # Sentiment
        sent_scores = self.score_sentiment_balance(
            line1, line2, line3,
            empirical_mean=empirical_sentiment_mean,
            full_haiku_embedding=current_embedding,
        )
        sent_balance = sent_scores['sentiment_balance']
        overall_sent = sent_scores['overall_sentiment']

        # Originality (if corpus loaded)
        orig_scores = self.score_originality(line1, line2, line3, current_embedding=current_embedding)
        originality = orig_scores['originality_score']

        # Composite score
        composite_score = (
            weights.get('syllable_accuracy') * syll_accuracy +
            weights.get('semantic_coherence') * sbert_coh +
            weights.get('kireji') * kireji +
            weights.get('imagery') * img_score +
            weights.get('nature') * nature_score +
            weights.get('lexical_diversity') * lex_div +
            weights.get('sentiment') * sent_balance +
            weights.get('density') * density
        )

        # Create unified output
        return HaikuQualityScores(
            # Structural
            syllable_accuracy=syll_accuracy,
            follows_575_pattern=follows_575,

            # Coherence
            semantic_coherence_sbert=sbert_coh,
            line1_line2_coherence=coherence_sbert['line1_line2_cosine'],
            line2_line3_coherence=coherence_sbert['line2_line3_cosine'],
            kireji_strength=kireji,

            # Imagery
            imagery_concreteness=img_score,
            imagery_visual=imagery_cats.get('imagery_visual', 0.0),
            imagery_auditory=imagery_cats.get('imagery_auditory', 0.0),
            imagery_tactile=imagery_cats.get('imagery_tactile', 0.0),
            imagery_gustatory=imagery_cats.get('imagery_gustatory', 0.0),
            imagery_olfactory=imagery_cats.get('imagery_olfactory', 0.0),
            nature_score=nature_score,
            sensory_verb_emphasis=sensory_emphasis,

            # Linguistic
            lexical_diversity=lex_div,
            adjective_density=ling_scores.get('adjective_density', 0.0),
            verb_density=ling_scores.get('verb_density', 0.0),
            noun_density=ling_scores.get('noun_density', 0.0),
            sensory_verb_density=nature_scores.get('sensory_verb_density', 0.0),

            # Semantic
            sentiment_balance=sent_balance,
            overall_sentiment=overall_sent,

            # Density
            semantic_density=density,

            # Originality
            originality_score=originality,

            # Composite
            composite_haiku_quality_score=composite_score,
        )

    # ============================================================================
    # BATCH PROCESSING
    # ============================================================================

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        line1_col: str = 'line1',
        line2_col: str = 'line2',
        line3_col: str = 'line3',
        extract_structural: bool = True,
        extract_linguistic: bool = True,
        extract_semantic: bool = True,
        extract_quality_scores: bool = True,
        extract_contrasts: bool = False,
        weights: Optional[Dict[str, float]] = None,
        verbose: bool = True,
        use_reference_corpus: bool = False,
        reference_corpus_df: Optional[pd.DataFrame] = None,
        sbert_batch_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Evaluate a dataset of haikus with comprehensive feature extraction.

        Args:
            df: DataFrame containing haiku lines
            line1_col, line2_col, line3_col: Column names for lines
            extract_structural: Extract structural features (syllables, word count)
            extract_linguistic: Extract linguistic features (POS, verb tense, density)
            extract_semantic: Extract semantic features (sentiment, polarity)
            extract_quality_scores: Compute quality metrics and unified HaikuQualityScores
            extract_contrasts: Extract contrast/antonym features (slower)
            weights: Optional weights for composite score
            verbose: Print progress information
            use_reference_corpus: Load reference corpus for originality scoring
            reference_corpus_df: DataFrame with reference haiku texts

        Returns:
            DataFrame with added feature columns (including unified quality scores)
        """
        if verbose:
            print(f"Processing {len(df)} haikus...")

        # Load reference corpus if requested
        if use_reference_corpus and reference_corpus_df is not None:
            ref_texts = []
            for _, row in reference_corpus_df.iterrows():
                text = ' '.join(str(x) for x in [
                    row.get(line1_col), row.get(line2_col), row.get(line3_col)
                ] if pd.notna(x))
                ref_texts.append(text)
            self.load_reference_corpus(ref_texts, batch_size=sbert_batch_size)
            if verbose:
                print(f"Loaded {len(ref_texts)} reference haiku for originality scoring")

        # Calculate empirical sentiment mean from dataset
        empirical_sentiment = 0.05  # default
        try:
            sentiments = []
            for _, row in df.iterrows():
                l1, l2, l3 = row[line1_col], row[line2_col], row[line3_col]
                full_text = ' '.join(str(x) for x in [l1, l2, l3] if pd.notna(x))
                if full_text.strip():
                    try:
                        emb = self._encode(full_text)
                        if isinstance(emb, (list, tuple)):
                            emb = np.asarray(emb)[0]
                        score = self._sentiment_score_from_embedding(emb)
                        sentiments.append(score)
                    except Exception:
                        continue
            if sentiments:
                empirical_sentiment = np.mean(sentiments)
        except Exception:
            pass

        # Precompute full-text embeddings for all haikus
        precomputed_haiku_embeddings = None
        if len(df) > 0:
            try:
                full_texts_for_embedding = []
                for _, row in df.iterrows():
                    text = ' '.join(str(x) for x in [row[line1_col], row[line2_col], row[line3_col]] if pd.notna(x))
                    full_texts_for_embedding.append(text)

                # Use batch_size from sbert_batch_size if provided, otherwise default to model's default
                encode_kwargs = {'show_progress_bar': False}
                if sbert_batch_size is not None:
                    encode_kwargs['batch_size'] = sbert_batch_size

                precomputed_haiku_embeddings = self._encode(full_texts_for_embedding, **encode_kwargs)
                precomputed_haiku_embeddings = np.asarray(precomputed_haiku_embeddings)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not precompute haiku embeddings in batch: {e}")
                precomputed_haiku_embeddings = None

        results = []
        errors = []

        # Add tqdm progress bar to the main loop
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing haikus", mininterval=10, disable=not verbose):
            try:
                l1 = row[line1_col]
                l2 = row[line2_col]
                l3 = row[line3_col]

                result_dict = {}

                # Get the precomputed embedding for the current haiku
                current_full_haiku_embedding = None
                if precomputed_haiku_embeddings is not None and idx < len(precomputed_haiku_embeddings):
                    current_full_haiku_embedding = precomputed_haiku_embeddings[idx]

                if extract_structural:
                    result_dict.update(self.extract_structural_features(l1, l2, l3))

                if extract_linguistic:
                    result_dict.update(self.extract_linguistic_features(l1, l2, l3))

                if extract_contrasts:
                    result_dict.update(self.extract_contrast_features(l1, l2, l3))

                if extract_semantic:
                    # Pass the precomputed embedding to extract_semantic_features
                    result_dict.update(self.extract_semantic_features(l1, l2, l3, current_embedding=current_full_haiku_embedding))

                if extract_quality_scores:
                    # Get unified HaikuQualityScores object
                    # Pass the precomputed embedding to score_haiku
                    quality_scores = self.score_haiku(
                        l1, l2, l3,
                        weights=weights,
                        empirical_sentiment_mean=empirical_sentiment,
                        current_embedding=current_full_haiku_embedding,
                    )
                    # Convert to dict and add to results
                    result_dict.update(quality_scores.to_dict())

                results.append(result_dict)

            except Exception as e:
                errors.append((idx, str(e)))
                if verbose:
                    print(f"  Error at row {idx}: {e}")
                # Append NaN result on error
                # Ensure all possible keys are present with NaN for consistency if an error occurs
                if extract_quality_scores:
                    # If quality_scores would have been computed, fill with NaNs for all keys
                    # This assumes HaikuQualityScores.to_dict() provides all keys
                    dummy_scores = HaikuQualityScores().to_dict()
                    for k in dummy_scores:
                        result_dict[k] = np.nan
                results.append(result_dict)

        scores_df = pd.DataFrame(results)
        result_df = pd.concat([df, scores_df], axis=1)

        if verbose:
            print(f"Processed {len(df) - len(errors)} haikus successfully")
            if errors:
                print(f"Encountered {len(errors)} errors")

        return result_df


if __name__ == '__main__':
    analyzer = HaikuAnalyzer()

    haikus = pd.read_csv("haiku_dataset_merged.csv")[['line1', 'line2', 'line3']]

    results_df = analyzer.evaluate_dataset(haikus)
    results_df.to_csv("haiku_analysis_results.csv", index=False)
    print("Results saved to haiku_analysis_results.csv")