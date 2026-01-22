"""
Deduplication utilities for systematic reviews
"""

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from typing import List, Dict, Tuple
import hashlib

class Deduplicator:
    """Handles deduplication of records"""
    
    def __init__(self, method: str = 'hybrid'):
        self.method = method
        self.statistics = {}
    
    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main deduplication method"""
        self.statistics['before'] = len(df)
        
        if self.method == 'strict':
            result = self._strict_deduplication(df)
        elif self.method == 'fuzzy':
            result = self._fuzzy_deduplication(df)
        elif self.method == 'hybrid':
            result = self._hybrid_deduplication(df)
        else:
            raise ValueError(f"Unknown deduplication method: {self.method}")
        
        self.statistics['after'] = len(result)
        self.statistics['removed'] = self.statistics['before'] - self.statistics['after']
        
        return result
    
    def _strict_deduplication(self, df: pd.DataFrame) -> pd.DataFrame:
        """Exact matching deduplication"""
        # Create hash for each record
        df['record_hash'] = df.apply(self._create_record_hash, axis=1)
        
        # Keep first occurrence of each hash
        return df.drop_duplicates(subset='record_hash', keep='first').drop(columns='record_hash')
    
    def _fuzzy_deduplication(self, df: pd.DataFrame, threshold: int = 90) -> pd.DataFrame:
        """Fuzzy matching deduplication"""
        # Simplified implementation
        duplicates = []
        
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                title_i = str(df.iloc[i]['title'])
                title_j = str(df.iloc[j]['title'])
                
                similarity = fuzz.ratio(title_i, title_j)
                
                if similarity >= threshold:
                    # Check authors and year
                    authors_i = str(df.iloc[i].get('authors', ''))
                    authors_j = str(df.iloc[j].get('authors', ''))
                    year_i = str(df.iloc[i].get('year', ''))
                    year_j = str(df.iloc[j].get('year', ''))
                    
                    if (fuzz.ratio(authors_i, authors_j) > 80 and 
                        year_i == year_j):
                        duplicates.append(j)
        
        # Remove duplicates
        mask = ~df.index.isin(set(duplicates))
        return df[mask].reset_index(drop=True)
    
    def _hybrid_deduplication(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hybrid deduplication using multiple strategies"""
        # Step 1: Exact matching on DOI
        df_no_doi_dups = df.drop_duplicates(subset='doi', keep='first')
        
        # Step 2: Fuzzy matching on titles
        df_fuzzy = self._fuzzy_deduplication(df_no_doi_dups)
        
        return df_fuzzy
    
    def _create_record_hash(self, row: pd.Series) -> str:
        """Create hash for record comparison"""
        # Combine key fields
        key_string = f"{row.get('title', '')}_{row.get('authors', '')}_{row.get('year', '')}"
        
        # Clean and normalize
        key_string = key_string.lower().strip()
        
        # Create hash
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_statistics(self) -> Dict:
        """Get deduplication statistics"""
        return self.statistics
