"""
PRISMA 2020 Selection Process Implementation
Handles identification, screening, eligibility, and inclusion phases
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PRISMASelection:
    """Implements PRISMA 2020 selection protocol"""
    
    def __init__(self, config_path: str = "config/eligibility_criteria.json"):
        """
        Initialize PRISMA selection process
        
        Args:
            config_path: Path to eligibility criteria JSON file
        """
        self.config = self._load_config(config_path)
        self.records = pd.DataFrame()
        self.selection_log = []
        self.inter_reviewer_stats = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load eligibility criteria from JSON configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def import_records(self, file_paths: Dict[str, str]) -> pd.DataFrame:
        """
        Import records from multiple database sources
        
        Args:
            file_paths: Dictionary mapping source names to file paths
            
        Returns:
            Combined DataFrame of all records
        """
        all_records = []
        
        for source, path in file_paths.items():
            logger.info(f"Importing records from {source}...")
            
            if path.endswith('.csv'):
                df = pd.read_csv(path, encoding='utf-8')
            elif path.endswith('.xlsx'):
                df = pd.read_excel(path)
            elif path.endswith('.bib'):
                df = self._parse_bibtex(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")
            
            # Add source identifier
            df['source_database'] = source
            df['import_timestamp'] = datetime.now()
            
            all_records.append(df)
        
        # Combine all records
        self.records = pd.concat(all_records, ignore_index=True)
        logger.info(f"Total records imported: {len(self.records)}")
        
        return self.records
    
    def deduplicate_records(self, method: str = "strict") -> pd.DataFrame:
        """
        Remove duplicate records using multiple strategies
        
        Args:
            method: 'strict' (exact matches), 'fuzzy' (similarity-based),
                   'hybrid' (combination)
        
        Returns:
            Deduplicated DataFrame
        """
        from .utils.deduplication import Deduplicator
        
        deduper = Deduplicator(method=method)
        self.records = deduper.deduplicate(self.records)
        
        # Log deduplication statistics
        stats = deduper.get_statistics()
        self.selection_log.append({
            'step': 'deduplication',
            'timestamp': datetime.now(),
            'method': method,
            'records_before': stats['before'],
            'records_after': stats['after'],
            'duplicates_removed': stats['removed']
        })
        
        return self.records
    
    def screen_titles_abstracts(self, 
                               reviewer_data: Dict[str, pd.DataFrame],
                               calculate_kappa: bool = True) -> pd.DataFrame:
        """
        Perform title/abstract screening with multiple reviewers
        
        Args:
            reviewer_data: Dictionary of reviewer names to DataFrames
            calculate_kappa: Whether to compute Cohen's kappa
        
        Returns:
            Screened records with consensus decisions
        """
        logger.info("Starting title/abstract screening...")
        
        # Combine reviewer decisions
        screened = self._combine_reviewer_decisions(reviewer_data)
        
        # Apply inclusion/exclusion criteria
        screened = self._apply_title_abstract_criteria(screened)
        
        # Calculate inter-reviewer agreement
        if calculate_kappa and len(reviewer_data) > 1:
            kappa = self._calculate_cohens_kappa(reviewer_data)
            self.inter_reviewer_stats['title_abstract_kappa'] = kappa
            logger.info(f"Cohen's kappa for title/abstract: {kappa:.3f}")
        
        # Log screening results
        included = screened[screened['include_ti_ab'] == True]
        excluded = screened[screened['include_ti_ab'] == False]
        
        self.selection_log.append({
            'step': 'title_abstract_screening',
            'timestamp': datetime.now(),
            'records_screened': len(screened),
            'records_included': len(included),
            'records_excluded': len(excluded),
            'kappa': self.inter_reviewer_stats.get('title_abstract_kappa', None)
        })
        
        return screened
    
    def assess_full_text_eligibility(self,
                                    full_texts: Dict[str, str],
                                    reviewer_decisions: Dict[str, Dict]) -> pd.DataFrame:
        """
        Assess full-text eligibility with documented reasons
        
        Args:
            full_texts: Mapping of record IDs to full text paths/content
            reviewer_decisions: Reviewer decisions with reasons
        
        Returns:
            Eligibility assessment results
        """
        logger.info("Starting full-text eligibility assessment...")
        
        eligibility_results = []
        
        for record_id, text_path in full_texts.items():
            # Parse full text (simplified - would use actual text parsing in practice)
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Collect all reviewer decisions for this record
            record_decisions = {}
            reasons = []
            
            for reviewer, decisions in reviewer_decisions.items():
                if record_id in decisions:
                    decision = decisions[record_id]
                    record_decisions[reviewer] = decision['include']
                    reasons.append(decision.get('reason', ''))
            
            # Determine consensus (majority vote, with third reviewer arbitration if needed)
            include_votes = sum(1 for d in record_decisions.values() if d)
            exclude_votes = len(record_decisions) - include_votes
            
            final_decision = include_votes > exclude_votes
            
            # Record eligibility assessment
            eligibility_results.append({
                'record_id': record_id,
                'reviewers': list(record_decisions.keys()),
                'include_votes': include_votes,
                'exclude_votes': exclude_votes,
                'final_decision': final_decision,
                'reasons': '; '.join(reasons),
                'exclusion_criteria': self._identify_exclusion_criteria(content, reasons)
            })
        
        # Create DataFrame and update records
        eligibility_df = pd.DataFrame(eligibility_results)
        
        # Log eligibility assessment
        included = eligibility_df[eligibility_df['final_decision'] == True]
        excluded = eligibility_df[eligibility_df['final_decision'] == False]
        
        self.selection_log.append({
            'step': 'full_text_assessment',
            'timestamp': datetime.now(),
            'full_texts_assessed': len(eligibility_df),
            'studies_included': len(included),
            'studies_excluded': len(excluded),
            'exclusion_reasons': eligibility_df['exclusion_criteria'].value_counts().to_dict()
        })
        
        return eligibility_df
    
    def generate_prisma_flow_data(self) -> Dict:
        """
        Generate data for PRISMA flow diagram
        
        Returns:
            Dictionary with counts for each PRISMA phase
        """
        flow_data = {
            'identification': {
                'database_records': len(self.records[self.records['source_database'].notna()]),
                'other_sources': len(self.records[self.records['source_database'].isna()]),
                'total_identified': len(self.records)
            },
            'screening': {
                'duplicates_removed': self._get_log_value('deduplication', 'duplicates_removed'),
                'records_screened': self._get_log_value('title_abstract_screening', 'records_screened'),
                'records_excluded_ti_ab': self._get_log_value('title_abstract_screening', 'records_excluded')
            },
            'eligibility': {
                'full_text_assessed': self._get_log_value('full_text_assessment', 'full_texts_assessed'),
                'full_text_excluded': self._get_log_value('full_text_assessment', 'studies_excluded'),
                'exclusion_reasons': self._get_exclusion_summary()
            },
            'included': {
                'total_included': self._get_log_value('full_text_assessment', 'studies_included'),
                'qualitative_synthesis': None,
                'quantitative_synthesis': None
            }
        }
        
        return flow_data
    
    def export_selection_report(self, output_dir: str = "reports") -> None:
        """Export comprehensive selection report"""
        report = {
            'metadata': {
                'review_title': 'Systematic Review of Quantum Machine Learning for Time Series Forecasting',
                'version': '1.0',
                'generation_date': datetime.now().isoformat(),
                'prisma_version': '2020'
            },
            'selection_process': self.selection_log,
            'inter_reviewer_agreement': self.inter_reviewer_stats,
            'final_counts': self.generate_prisma_flow_data(),
            'included_studies_list': self._get_included_studies_list()
        }
        
        # Save as JSON
        output_path = Path(output_dir) / f"prisma_selection_report_{datetime.now():%Y%m%d}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Selection report exported to: {output_path}")
    
    # Helper methods
    def _combine_reviewer_decisions(self, reviewer_data: Dict) -> pd.DataFrame:
        """Combine decisions from multiple reviewers"""
        # Implementation details...
        pass
    
    def _apply_title_abstract_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inclusion/exclusion criteria to title/abstract"""
        # Implementation details...
        pass
    
    def _calculate_cohens_kappa(self, reviewer_data: Dict) -> float:
        """Calculate Cohen's kappa for inter-reviewer agreement"""
        # Implementation details...
        pass
    
    def _identify_exclusion_criteria(self, content: str, reasons: List[str]) -> List[str]:
        """Identify which exclusion criteria were met"""
        # Implementation details...
        pass
    
    def _get_log_value(self, step: str, key: str) -> Optional[int]:
        """Retrieve value from selection log"""
        for entry in self.selection_log:
            if entry.get('step') == step:
                return entry.get(key)
        return None
    
    def _get_exclusion_summary(self) -> Dict:
        """Summarize exclusion reasons"""
        # Implementation details...
        pass
    
    def _get_included_studies_list(self) -> List[Dict]:
        """Get list of included studies"""
        # Implementation details...
        pass
    
    def _parse_bibtex(self, filepath: str) -> pd.DataFrame:
        """Parse BibTeX file into DataFrame"""
        # Implementation details...
        pass
