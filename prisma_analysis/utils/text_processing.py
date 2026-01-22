"""
Text processing utilities for systematic reviews
"""

import re
import string
from typing import List, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextProcessor:
    """Processes text for analysis"""
    
    def __init__(self):
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def extract_key_phrases(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key phrases from text"""
        # Tokenize and clean
        tokens = self._tokenize_and_clean(text)
        
        # Get frequency distribution
        freq_dist = nltk.FreqDist(tokens)
        
        # Return most common phrases
        return [word for word, _ in freq_dist.most_common(top_n)]
    
    def extract_metrics(self, text: str) -> Dict[str, float]:
        """Extract performance metrics from text"""
        metrics = {}
        
        # Patterns for common metrics
        patterns = {
            'rmse': r'RMSE\s*[:=]?\s*([0-9]*\.?[0-9]+)',
            'mae': r'MAE\s*[:=]?\s*([0-9]*\.?[0-9]+)',
            'mape': r'MAPE\s*[:=]?\s*([0-9]*\.?[0-9]+)',
            'r2': r'R[²2]\s*[:=]?\s*([0-9]*\.?[0-9]+)',
            'accuracy': r'accuracy\s*[:=]?\s*([0-9]*\.?[0-9]+)%?'
        }
        
        for metric, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the first match
                try:
                    metrics[metric] = float(matches[0])
                except ValueError:
                    continue
        
        return metrics
    
    def identify_qml_approach(self, text: str) -> str:
        """Identify QML approach from text"""
        approaches = {
            'QNN': ['quantum neural', 'variational quantum', 'vqc', 'qnn'],
            'QSVM': ['quantum support vector', 'quantum kernel', 'qsvm'],
            'QELM': ['quantum extreme learning', 'qelm'],
            'QRC': ['quantum reservoir', 'qrc']
        }
        
        text_lower = text.lower()
        
        for approach, keywords in approaches.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return approach
        
        return 'Other'
    
    def _tokenize_and_clean(self, text: str) -> List[str]:
        """Tokenize and clean text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens
