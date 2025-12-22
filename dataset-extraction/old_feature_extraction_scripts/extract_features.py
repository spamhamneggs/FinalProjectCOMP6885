import pandas as pd
import nltk
import pyphen
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Create a Pyphen object for English
dic = pyphen.Pyphen(lang='en')

def count_syllables(word):
    """Count syllables in a word using pyphen."""
    word = word.lower()
    # Insert hyphens into the word to indicate syllable breaks
    syllabified = dic.inserted(word)
    # Count the number of syllables by splitting on hyphens
    syllable_count = len(syllabified.split('-'))
    # Words have at least 1 syllable
    return max(1, syllable_count)

def extract_structural_features(line1, line2, line3):
    """Extract structural features: syllable counts and 5-7-5 pattern."""
    # Handle NaN values
    if pd.isna(line1) or pd.isna(line2) or pd.isna(line3):
        return {
            'line1_syllables': 0,
            'line2_syllables': 0,
            'line3_syllables': 0,
            'total_syllables': 0,
            'follows_575_pattern': False,
            'total_words': 0,
        }
    
    lines = [line1, line2, line3]
    syllable_counts = []
    total_syllables = 0
    
    for line in lines:
        words = line.split()
        line_syllables = sum(count_syllables(word) for word in words)
        syllable_counts.append(line_syllables)
        total_syllables += line_syllables
    
    # Check if follows 5-7-5 pattern
    follows_575 = (syllable_counts == [5, 7, 5])
    
    # Total word count
    total_words = sum(len(line.split()) for line in lines)
    
    return {
        'line1_syllables': syllable_counts[0],
        'line2_syllables': syllable_counts[1],
        'line3_syllables': syllable_counts[2],
        'total_syllables': total_syllables,
        'follows_575_pattern': follows_575,
        'total_words': total_words,
    }

def extract_linguistic_features(line1, line2, line3):
    """Extract linguistic features: POS distribution, verb tense, adjective density."""
    # Handle NaN values
    if pd.isna(line1) or pd.isna(line2) or pd.isna(line3):
        return {
            'noun_count': 0,
            'verb_count': 0,
            'adjective_count': 0,
            'adverb_count': 0,
            'adjective_density': 0,
            'verb_density': 0,
            'noun_density': 0,
            'past_tense_count': 0,
            'present_tense_count': 0,
            'gerund_count': 0,
        }
    
    full_text = f"{line1} {line2} {line3}"
    words = word_tokenize(full_text)
    pos_tags = pos_tag(words)
    
    # Count POS tags
    pos_counts = {}
    for word, tag in pos_tags:
        pos_counts[tag] = pos_counts.get(tag, 0) + 1
    
    # Common POS tags
    total_tags = len(pos_tags)
    noun_count = pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)
    verb_count = pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)
    adjective_count = pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0)
    adverb_count = pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + pos_counts.get('RBS', 0)
    
    # Verb tense distribution
    past_tense = pos_counts.get('VBD', 0)
    present_tense = pos_counts.get('VB', 0) + pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)
    gerund = pos_counts.get('VBG', 0)
    
    # Densities
    adjective_density = adjective_count / total_tags if total_tags > 0 else 0
    verb_density = verb_count / total_tags if total_tags > 0 else 0
    noun_density = noun_count / total_tags if total_tags > 0 else 0
    
    return {
        'noun_count': noun_count,
        'verb_count': verb_count,
        'adjective_count': adjective_count,
        'adverb_count': adverb_count,
        'adjective_density': adjective_density,
        'verb_density': verb_density,
        'noun_density': noun_density,
        'past_tense_count': past_tense,
        'present_tense_count': present_tense,
        'gerund_count': gerund,
    }

def extract_semantic_features(line1, line2, line3):
    """Extract semantic features: sentiment, contrast."""
    # Handle NaN values
    if pd.isna(line1) or pd.isna(line2) or pd.isna(line3):
        return {
            'sentiment_positive': 0,
            'sentiment_negative': 0,
            'sentiment_neutral': 0,
            'sentiment_compound': 0,
            'polarity': 0,
            'subjectivity': 0,
            'has_contrast': False,
            'contrast_pairs': 0,
        }
    
    full_text = f"{line1} {line2} {line3}"
    
    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(full_text)
    
    # TextBlob sentiment (polarity -1 to 1, subjectivity 0 to 1)
    blob = TextBlob(full_text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    text_lower = full_text.lower()
    
    # Detect contrasts/juxtaposition using WordNet antonyms
    words_in_text = set(word_tokenize(text_lower))
    contrasts_found = []
    
    for word in words_in_text:
        # Get all synsets for this word
        synsets = wordnet.synsets(word)
        for synset in synsets:
            # Get all lemmas for this synset and their antonyms
            for lemma in synset.lemmas():
                if lemma.antonyms():
                    for antonym in lemma.antonyms():
                        antonym_word = antonym.name()
                        # Check if antonym exists in text
                        if antonym_word in text_lower:
                            contrasts_found.append(f"{word}-{antonym_word}")
    
    # Remove duplicates
    contrasts_found = list(set(contrasts_found))
    has_contrast = len(contrasts_found) > 0
    
    return {
        'sentiment_positive': sentiment_scores['pos'],
        'sentiment_negative': sentiment_scores['neg'],
        'sentiment_neutral': sentiment_scores['neu'],
        'sentiment_compound': sentiment_scores['compound'],
        'polarity': polarity,
        'subjectivity': subjectivity,
        'has_contrast': has_contrast,
        'contrast_pairs': len(contrasts_found),
    }

def main():
    # Load the merged dataset
    df = pd.read_csv("haiku_dataset_merged.csv")
    
    # Remove rows with NaN values in any haiku line
    df = df.dropna(subset=['line1', 'line2', 'line3'])
    
    print(f"Processing {len(df)} haikus...")
    
    # Extract all features
    structural_features = df.apply(
        lambda row: extract_structural_features(row['line1'], row['line2'], row['line3']),
        axis=1
    )
    structural_df = pd.DataFrame(structural_features.tolist())
    
    linguistic_features = df.apply(
        lambda row: extract_linguistic_features(row['line1'], row['line2'], row['line3']),
        axis=1
    )
    linguistic_df = pd.DataFrame(linguistic_features.tolist())
    
    semantic_features = df.apply(
        lambda row: extract_semantic_features(row['line1'], row['line2'], row['line3']),
        axis=1
    )
    semantic_df = pd.DataFrame(semantic_features.tolist())
    
    # Combine all features
    result_df = pd.concat([df, structural_df, linguistic_df, semantic_df], axis=1)
    
    # Save to CSV
    result_df.to_csv("haiku_features.csv", index=False)
    print(f"[OK] Saved {len(result_df)} haikus with features to 'haiku_features.csv'")
    
    # Print summary statistics
    print("\n=== STRUCTURAL FEATURES ===")
    print(f"Average syllables per line 1: {structural_df['line1_syllables'].mean():.3f}")
    print(f"Average syllables per line 2: {structural_df['line2_syllables'].mean():.3f}")
    print(f"Average syllables per line 3: {structural_df['line3_syllables'].mean():.3f}")
    print(f"Haikus following 5-7-5 pattern: {structural_df['follows_575_pattern'].sum()} ({structural_df['follows_575_pattern'].sum()/len(structural_df)*100:.1f}%)")
    print(f"Average total words: {structural_df['total_words'].mean():.2f}")
    
    print("\n=== LINGUISTIC FEATURES ===")
    print(f"Average nouns per haiku: {linguistic_df['noun_count'].mean():.3f}")
    print(f"Average verbs per haiku: {linguistic_df['verb_count'].mean():.3f}")
    print(f"Average adjectives per haiku: {linguistic_df['adjective_count'].mean():.3f}")
    print(f"Average adjective density: {linguistic_df['adjective_density'].mean():.3f}")
    verb_total = linguistic_df['verb_count'].sum()
    if verb_total > 0:
        print(f"Past tense ratio: {(linguistic_df['past_tense_count'].sum() / verb_total):.3f}")
        print(f"Present tense ratio: {(linguistic_df['present_tense_count'].sum() / verb_total):.3f}")
    else:
        print("Past tense ratio: N/A (no verbs found)")
        print("Present tense ratio: N/A (no verbs found)")
    
    print("\n=== SEMANTIC FEATURES ===")
    print(f"Average sentiment (compound): {semantic_df['sentiment_compound'].mean():.3f}")
    print(f"Average polarity: {semantic_df['polarity'].mean():.3f}")
    print(f"Average subjectivity: {semantic_df['subjectivity'].mean():.3f}")
    print(f"Haikus with contrast: {semantic_df['has_contrast'].sum()} ({semantic_df['has_contrast'].sum()/len(semantic_df)*100:.1f}%)")
    
if __name__ == "__main__":
    main()
