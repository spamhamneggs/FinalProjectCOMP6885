"""
Haiku Quality Scorer

Implements quantitative metrics for evaluating haiku quality based on:
1. Syllable accuracy (structural metric)
2. Semantic coherence (embedding-based)
3. Imagery/concreteness
4. Lexical diversity
5. Sentiment balance
6. Language model perplexity
7. Composite quality score

Usage:
    scorer = HaikuQualityScorer()
    score = scorer.score_haiku(line1, line2, line3)
"""

import pandas as pd
import numpy as np
import nltk
import pyphen
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


class HaikuQualityScorer:
    """
    Comprehensive haiku quality scorer using multiple quantitative metrics.
    """
    
    def __init__(self, target_syllables: Tuple[int, int, int] = (5, 7, 5)):
        """
        Initialize the HaikuQualityScorer.
        
        Args:
            target_syllables: Tuple of target syllable counts (default: 5-7-5)
        """
        self.target_syllables = target_syllables
        self.dic = pyphen.Pyphen(lang='en')
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Concreteness lexicon (simplified subset based on Brysbaert et al. 2014)
        # In practice, you'd load a full lexicon; this is illustrative
        self.concreteness_words = self._load_concreteness_lexicon()
        
        # Abstract words (opposite of concrete)
        self.abstract_words = self._load_abstract_words()
        
    def _load_concreteness_lexicon(self) -> Dict[str, float]:
        """Load a simplified concreteness lexicon.
        
        Scores range from 0 (abstract) to 1 (concrete).
        
        Returns:
            Dictionary mapping words to concreteness scores.
        """
        # Simplified lexicon - in production, load from external resource
        concrete_words = {
            # Nature/sensory words (high concreteness)
            'dew': 0.95, 'water': 0.95, 'stone': 0.95, 'rock': 0.95,
            'tree': 0.95, 'leaf': 0.95, 'leaves': 0.95, 'branch': 0.95,
            'snow': 0.95, 'rain': 0.95, 'moon': 0.95, 'sun': 0.95,
            'bird': 0.95, 'butterfly': 0.95, 'flower': 0.95, 'rose': 0.95,
            'morning': 0.90, 'evening': 0.90, 'night': 0.90, 'dawn': 0.90,
            'wind': 0.90, 'breeze': 0.90, 'stream': 0.90, 'river': 0.90,
            'pond': 0.90, 'lake': 0.90, 'mountain': 0.90, 'valley': 0.90,
            'hand': 0.95, 'face': 0.95, 'eyes': 0.95, 'voice': 0.85,
            'sound': 0.75, 'light': 0.85, 'shadow': 0.85, 'color': 0.85,
            'red': 0.90, 'blue': 0.90, 'green': 0.90, 'white': 0.90,
            'sweet': 0.80, 'bitter': 0.80, 'warm': 0.80, 'cold': 0.80,
            'soft': 0.80, 'hard': 0.80, 'rough': 0.80, 'smooth': 0.80,
            'fragrant': 0.85, 'silence': 0.70, 'echo': 0.75, 'whisper': 0.75,
        }
        return concrete_words
    
    def _load_abstract_words(self) -> set:
        """Load a set of common abstract words."""
        abstract_words = {
            'feeling', 'feeling', 'emotion', 'thought', 'idea', 'concept',
            'time', 'life', 'death', 'love', 'hate', 'fear', 'joy', 'sadness',
            'beauty', 'truth', 'meaning', 'purpose', 'soul', 'spirit',
            'darkness', 'light', 'hope', 'despair', 'peace', 'chaos',
            'memory', 'dream', 'fate', 'destiny', 'eternity', 'infinity',
            'knowledge', 'wisdom', 'understanding', 'power', 'strength',
        }
        return abstract_words
    
    def count_syllables(self, word: str) -> int:
        """
        Count syllables in a word using pyphen hyphenation.
        
        Args:
            word: The word to count syllables for.
            
        Returns:
            Number of syllables (minimum 1).
        """
        word = word.lower().strip('.,!?;:')
        if not word:
            return 0
        syllabified = self.dic.inserted(word)
        syllable_count = len(syllabified.split('-'))
        return max(1, syllable_count)
    
    def score_syllable_accuracy(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """
        Score syllable accuracy for 5-7-5 pattern (or target pattern).
        
        Args:
            line1, line2, line3: The three lines of the haiku.
            
        Returns:
            Dictionary with individual line scores and overall accuracy.
        """
        lines = [line1, line2, line3]
        line_syllable_counts = []
        line_scores = []
        
        for line, target in zip(lines, self.target_syllables):
            if pd.isna(line) or not line.strip():
                line_syllable_counts.append(0)
                line_scores.append(0.0)
                continue
            
            words = word_tokenize(line)
            syllable_count = sum(self.count_syllables(w) for w in words)
            line_syllable_counts.append(syllable_count)
            
            # Calculate deviation score: 1 - (|actual - target| / target)
            # This heavily penalizes large deviations
            deviation = abs(syllable_count - target)
            score = max(0.0, 1.0 - (deviation / target))
            line_scores.append(score)
        
        overall_accuracy = np.mean(line_scores)
        
        return {
            'line1_syllables': line_syllable_counts[0],
            'line2_syllables': line_syllable_counts[1],
            'line3_syllables': line_syllable_counts[2],
            'line1_score': line_scores[0],
            'line2_score': line_scores[1],
            'line3_score': line_scores[2],
            'overall_syllable_accuracy': overall_accuracy,
        }
    
    def score_semantic_coherence(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """
        Score semantic coherence using simple word overlap heuristic.
        
        In production, use sentence embeddings (SBERT, USE) for better results.
        
        Args:
            line1, line2, line3: The three lines of the haiku.
            
        Returns:
            Dictionary with coherence metrics.
        """
        lines = [line1, line2, line3]
        
        # Simple word overlap metric (more sophisticated: use embeddings)
        def get_words(text):
            if pd.isna(text):
                return set()
            return set(w.lower() for w in word_tokenize(text) 
                      if w.isalpha() and w.lower() not in self.stop_words)
        
        words_list = [get_words(line) for line in lines]
        
        # Compute overlap between consecutive lines
        overlap_1_2 = len(words_list[0] & words_list[1]) / (len(words_list[0] | words_list[1]) + 1e-6)
        overlap_2_3 = len(words_list[1] & words_list[2]) / (len(words_list[1] | words_list[2]) + 1e-6)
        
        # Balance: some overlap is good (coherence), but too much is redundant
        # Ideal is moderate overlap (0.3-0.5 range)
        ideal_overlap = 0.4
        coherence_1_2 = 1.0 - abs(overlap_1_2 - ideal_overlap) / max(overlap_1_2, ideal_overlap, 1e-6)
        coherence_2_3 = 1.0 - abs(overlap_2_3 - ideal_overlap) / max(overlap_2_3, ideal_overlap, 1e-6)
        
        # Clamp to [0, 1]
        coherence_1_2 = max(0.0, min(1.0, coherence_1_2))
        coherence_2_3 = max(0.0, min(1.0, coherence_2_3))
        
        overall_coherence = (coherence_1_2 + coherence_2_3) / 2.0
        
        return {
            'line1_line2_overlap': overlap_1_2,
            'line2_line3_overlap': overlap_2_3,
            'coherence_1_2': coherence_1_2,
            'coherence_2_3': coherence_2_3,
            'overall_semantic_coherence': overall_coherence,
        }
    
    def score_imagery_concreteness(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """
        Score imagery and concreteness using a concreteness lexicon.
        
        Args:
            line1, line2, line3: The three lines of the haiku.
            
        Returns:
            Dictionary with concreteness scores.
        """
        lines = [line1, line2, line3]
        concreteness_scores = []
        
        for line in lines:
            if pd.isna(line) or not line.strip():
                concreteness_scores.append(0.0)
                continue
            
            words = [w.lower() for w in word_tokenize(line) if w.isalpha()]
            if not words:
                concreteness_scores.append(0.0)
                continue
            
            # Score each word's concreteness
            word_scores = []
            for word in words:
                if word in self.concreteness_words:
                    word_scores.append(self.concreteness_words[word])
                elif word in self.abstract_words:
                    word_scores.append(0.2)  # Penalize abstract words
                else:
                    # Default: neutral (0.5)
                    word_scores.append(0.5)
            
            line_concreteness = np.mean(word_scores)
            concreteness_scores.append(line_concreteness)
        
        overall_concreteness = np.mean(concreteness_scores)
        
        return {
            'line1_concreteness': concreteness_scores[0],
            'line2_concreteness': concreteness_scores[1],
            'line3_concreteness': concreteness_scores[2],
            'overall_imagery_concreteness': overall_concreteness,
        }
    
    def score_lexical_diversity(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """
        Score lexical diversity using Type-Token Ratio (TTR).
        
        Higher TTR indicates more varied vocabulary.
        
        Args:
            line1, line2, line3: The three lines of the haiku.
            
        Returns:
            Dictionary with diversity metrics.
        """
        text = ' '.join(str(x) for x in [line1, line2, line3] if pd.notna(x))
        
        if not text.strip():
            return {'ttr': 0.0, 'lexical_diversity': 0.0}
        
        words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
        
        if not words:
            return {'ttr': 0.0, 'lexical_diversity': 0.0}
        
        unique_words = len(set(words))
        total_words = len(words)
        
        # TTR: ratio of unique to total
        ttr = unique_words / total_words if total_words > 0 else 0.0
        
        # For very short texts (like haiku, ~10-15 words), TTR can be high
        # Normalize to a reasonable scale
        # Maximum TTR for haiku is typically 0.7-1.0
        normalized_diversity = min(1.0, ttr * 1.2)
        
        return {
            'unique_words': unique_words,
            'total_words': total_words,
            'ttr': ttr,
            'lexical_diversity': normalized_diversity,
        }
    
    def score_sentiment_balance(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """
        Score sentiment balance using VADER sentiment analysis.
        
        Args:
            line1, line2, line3: The three lines of the haiku.
            
        Returns:
            Dictionary with sentiment metrics.
        """
        lines = [line1, line2, line3]
        sentiments = []
        
        for line in lines:
            if pd.isna(line) or not line.strip():
                sentiments.append(0.0)
                continue
            
            scores = self.sia.polarity_scores(line)
            # Compound score ranges from -1 (negative) to 1 (positive)
            sentiments.append(scores['compound'])
        
        overall_sentiment = np.mean(sentiments)
        
        # Sentiment balance: prefer slightly positive or neutral
        # (not too extreme in either direction)
        # Ideal is around 0.0 to 0.3 (mildly positive/neutral)
        ideal_sentiment = 0.15
        sentiment_balance = 1.0 - abs(overall_sentiment - ideal_sentiment) / 2.0
        sentiment_balance = max(0.0, min(1.0, sentiment_balance))
        
        return {
            'line1_sentiment': sentiments[0],
            'line2_sentiment': sentiments[1],
            'line3_sentiment': sentiments[2],
            'overall_sentiment': overall_sentiment,
            'sentiment_balance': sentiment_balance,
        }
    
    def score_language_model_perplexity(
        self,
        line1: str,
        line2: str,
        line3: str
    ) -> Dict[str, float]:
        """
        Estimate language model perplexity using TextBlob (simplified).
        
        For production, use transformers library with GPT-2 or similar.
        
        Args:
            line1, line2, line3: The three lines of the haiku.
            
        Returns:
            Dictionary with perplexity-based scores.
        """
        text = ' '.join(str(x) for x in [line1, line2, line3] if pd.notna(x))
        
        if not text.strip():
            return {'grammatical_fluency': 0.0}
        
        blob = TextBlob(text)
        
        # TextBlob doesn't directly compute perplexity,
        # but we can use a simple heuristic based on sentence structure
        sentences = blob.sentences
        
        # Proxy: average sentence length and structure quality
        if not sentences:
            return {'grammatical_fluency': 0.0}
        
        # Prefer shorter, concise sentences (characteristic of haiku)
        avg_sentence_length = np.mean([len(s.words) for s in sentences])
        
        # Ideal sentence length for haiku is 5-10 words per line
        ideal_length = 7
        length_score = 1.0 - abs(avg_sentence_length - ideal_length) / (ideal_length + 1e-6)
        length_score = max(0.0, min(1.0, length_score))
        
        # Simple grammar check: assume TextBlob sentences are well-formed
        # (in production, use more sophisticated parsing)
        grammatical_fluency = length_score * 0.8 + 0.2  # Baseline confidence
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'grammatical_fluency': min(1.0, grammatical_fluency),
        }
    
    def score_haiku(
        self,
        line1: str,
        line2: str,
        line3: str,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive haiku quality score.
        
        Args:
            line1, line2, line3: The three lines of the haiku.
            weights: Optional dict of component weights.
                     Default: equal weights (0.2 each for 5 components).
        
        Returns:
            Dictionary with all component scores and composite quality score.
        """
        # Default equal weights
        if weights is None:
            weights = {
                'syllable_accuracy': 0.20,
                'semantic_coherence': 0.20,
                'imagery': 0.20,
                'lexical_diversity': 0.20,
                'sentiment': 0.10,
                'fluency': 0.10,
            }
        
        # Compute all component scores
        syllable_scores = self.score_syllable_accuracy(line1, line2, line3)
        coherence_scores = self.score_semantic_coherence(line1, line2, line3)
        imagery_scores = self.score_imagery_concreteness(line1, line2, line3)
        diversity_scores = self.score_lexical_diversity(line1, line2, line3)
        sentiment_scores = self.score_sentiment_balance(line1, line2, line3)
        fluency_scores = self.score_language_model_perplexity(line1, line2, line3)
        
        # Extract main scores
        syll_score = syllable_scores['overall_syllable_accuracy']
        coh_score = coherence_scores['overall_semantic_coherence']
        img_score = imagery_scores['overall_imagery_concreteness']
        lex_score = diversity_scores['lexical_diversity']
        sent_score = sentiment_scores['sentiment_balance']
        flu_score = fluency_scores['grammatical_fluency']
        
        # Compute composite score (weighted average)
        composite_score = (
            weights['syllable_accuracy'] * syll_score +
            weights['semantic_coherence'] * coh_score +
            weights['imagery'] * img_score +
            weights['lexical_diversity'] * lex_score +
            weights['sentiment'] * sent_score +
            weights['fluency'] * flu_score
        )
        
        # Compile all results
        all_scores = {
            **syllable_scores,
            **coherence_scores,
            **imagery_scores,
            **diversity_scores,
            **sentiment_scores,
            **fluency_scores,
            'composite_haiku_quality_score': composite_score,
        }
        
        return all_scores


def evaluate_haiku_dataset(
    df: pd.DataFrame,
    line1_col: str = 'line1',
    line2_col: str = 'line2',
    line3_col: str = 'line3',
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Evaluate a dataset of haikus and add quality scores.
    
    Args:
        df: DataFrame containing haiku lines.
        line1_col, line2_col, line3_col: Column names for the three lines.
        weights: Optional weights for component scores.
    
    Returns:
        DataFrame with added quality score columns.
    """
    scorer = HaikuQualityScorer()
    results = []
    
    for idx, row in df.iterrows():
        try:
            scores = scorer.score_haiku(
                row[line1_col],
                row[line2_col],
                row[line3_col],
                weights=weights
            )
            results.append(scores)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            # Return NaN for all scores on error
            results.append({
                'composite_haiku_quality_score': np.nan,
                'overall_syllable_accuracy': np.nan,
                'overall_semantic_coherence': np.nan,
                'overall_imagery_concreteness': np.nan,
                'lexical_diversity': np.nan,
                'sentiment_balance': np.nan,
                'grammatical_fluency': np.nan,
            })
    
    scores_df = pd.DataFrame(results)
    return pd.concat([df, scores_df], axis=1)


if __name__ == '__main__':
    # Example usage
    scorer = HaikuQualityScorer()
    
    # Example haikus
    haikus = [
        ('Morning dew drops', 'Glistening on emerald leaves', 'Sparrows take flight'),
        ('Silent autumn wind', 'Rustles through the empty trees', 'Life fades away'),
        ('Cherry blossoms bloom', 'Pink petals dance in spring breeze', 'Beauty fleeting fast'),
    ]
    
    print("=" * 80)
    print("HAIKU QUALITY EVALUATION")
    print("=" * 80)
    
    for i, (l1, l2, l3) in enumerate(haikus, 1):
        print(f"\nHaiku {i}:")
        print(f"  {l1}")
        print(f"  {l2}")
        print(f"  {l3}")
        
        scores = scorer.score_haiku(l1, l2, l3)
        
        print(f"\n  Syllable Accuracy: {scores['overall_syllable_accuracy']:.3f}")
        print(f"    Line 1 (5): {scores['line1_syllables']} syllables -> {scores['line1_score']:.3f}")
        print(f"    Line 2 (7): {scores['line2_syllables']} syllables -> {scores['line2_score']:.3f}")
        print(f"    Line 3 (5): {scores['line3_syllables']} syllables -> {scores['line3_score']:.3f}")
        
        print(f"\n  Semantic Coherence: {scores['overall_semantic_coherence']:.3f}")
        print(f"    Line 1<->2 Overlap: {scores['line1_line2_overlap']:.3f}")
        print(f"    Line 2<->3 Overlap: {scores['line2_line3_overlap']:.3f}")
        
        print(f"\n  Imagery/Concreteness: {scores['overall_imagery_concreteness']:.3f}")
        
        print(f"\n  Lexical Diversity (TTR): {scores['ttr']:.3f}")
        
        print(f"\n  Sentiment Balance: {scores['sentiment_balance']:.3f}")
        print(f"    Overall Sentiment: {scores['overall_sentiment']:.3f}")
        
        print(f"\n  Grammatical Fluency: {scores['grammatical_fluency']:.3f}")
        
        print(f"\n  >>> COMPOSITE QUALITY SCORE: {scores['composite_haiku_quality_score']:.3f} <<<")
        print("-" * 80)
