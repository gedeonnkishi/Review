"""
Quality assessment and risk of bias evaluation
Implements ROBIS, Cochrane, and custom QML-specific criteria
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

class RiskLevel(Enum):
    """Risk of bias levels"""
    LOW = "low"
    SOME_CONCERNS = "some_concerns"
    HIGH = "high"
    UNKNOWN = "unknown"

class QualityDomain(Enum):
    """Quality assessment domains"""
    ARCHITECTURE_CLARITY = "D1_architecture_clarity"
    BASELINE_RIGOR = "D2_baseline_rigor"
    HARDWARE_CLARITY = "D3_hardware_clarity"
    REPRODUCIBILITY = "D4_reproducibility"
    STATISTICAL_RIGOR = "D5_statistical_rigor"
    CONFLICT_INTEREST = "D6_conflict_interest"

class QualityAssessor:
    """Assesses quality and risk of bias for included studies"""
    
    def __init__(self, criteria_path: str = "config/quality_domains.json"):
        """
        Initialize quality assessor
        
        Args:
            criteria_path: Path to quality criteria JSON
        """
        self.criteria = self._load_criteria(criteria_path)
        self.assessments = {}
        self.inter_reviewer_agreement = {}
        
    def _load_criteria(self, criteria_path: str) -> Dict:
        """Load quality assessment criteria"""
        with open(criteria_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def assess_study(self, 
                    study_id: str,
                    study_data: Dict,
                    reviewer_id: str = "reviewer_1") -> Dict:
        """
        Assess quality for a single study
        
        Args:
            study_id: Unique study identifier
            study_data: Study characteristics and full text
            reviewer_id: Identifier of reviewer
        
        Returns:
            Quality assessment results
        """
        assessment = {
            'study_id': study_id,
            'reviewer_id': reviewer_id,
            'assessment_date': pd.Timestamp.now().isoformat(),
            'domains': {},
            'overall_risk': None,
            'comments': []
        }
        
        # Assess each domain
        for domain_name, domain_criteria in self.criteria['domains'].items():
            domain_assessment = self._assess_domain(
                domain_name, domain_criteria, study_data
            )
            assessment['domains'][domain_name] = domain_assessment
        
        # Determine overall risk
        assessment['overall_risk'] = self._determine_overall_risk(
            assessment['domains']
        )
        
        # Store assessment
        if study_id not in self.assessments:
            self.assessments[study_id] = []
        self.assessments[study_id].append(assessment)
        
        return assessment
    
    def compare_reviewer_assessments(self, study_id: str) -> Dict:
        """
        Compare assessments from multiple reviewers
        
        Args:
            study_id: Study to compare
        
        Returns:
            Comparison results including agreement statistics
        """
        if study_id not in self.assessments:
            raise ValueError(f"No assessments found for study {study_id}")
        
        assessments = self.assessments[study_id]
        
        if len(assessments) < 2:
            return {'warning': 'Only one reviewer assessment available'}
        
        # Extract domain assessments
        domain_results = {}
        for domain in QualityDomain:
            domain_name = domain.value
            reviewer_scores = []
            
            for assessment in assessments:
                if domain_name in assessment['domains']:
                    risk_level = assessment['domains'][domain_name]['risk_level']
                    # Convert to numeric for comparison
                    score = self._risk_to_numeric(risk_level)
                    reviewer_scores.append(score)
            
            if reviewer_scores:
                domain_results[domain_name] = {
                    'reviewer_scores': reviewer_scores,
                    'mean_score': np.mean(reviewer_scores),
                    'std_score': np.std(reviewer_scores),
                    'agreement': self._calculate_agreement(reviewer_scores)
                }
        
        # Calculate overall agreement
        overall_scores = []
        for assessment in assessments:
            overall_scores.append(
                self._risk_to_numeric(assessment['overall_risk'])
            )
        
        comparison = {
            'study_id': study_id,
            'num_reviewers': len(assessments),
            'domain_comparisons': domain_results,
            'overall_comparison': {
                'scores': overall_scores,
                'mean': np.mean(overall_scores),
                'std': np.std(overall_scores),
                'agreement': self._calculate_agreement(overall_scores)
            }
        }
        
        # Store for later analysis
        self.inter_reviewer_agreement[study_id] = comparison
        
        return comparison
    
    def generate_quality_summary(self) -> pd.DataFrame:
        """
        Generate summary of quality assessments across all studies
        
        Returns:
            DataFrame with quality summary
        """
        summary_data = []
        
        for study_id, assessments in self.assessments.items():
            # Use consensus assessment if multiple reviewers
            if len(assessments) > 1:
                # Take mode or average based on criteria
                consensus = self._calculate_consensus(assessments)
            else:
                consensus = assessments[0]
            
            # Extract domain scores
            domain_scores = {}
            for domain_name, domain_assessment in consensus['domains'].items():
                domain_scores[domain_name] = self._risk_to_numeric(
                    domain_assessment['risk_level']
                )
            
            summary_data.append({
                'study_id': study_id,
                'overall_risk': consensus['overall_risk'],
                'overall_score': self._risk_to_numeric(consensus['overall_risk']),
                **domain_scores
            })
        
        return pd.DataFrame(summary_data)
    
    def create_risk_of_bias_plot(self, 
                                output_path: str = None,
                                figsize: Tuple = (12, 8)) -> plt.Figure:
        """
        Create risk of bias visualization
        
        Args:
            output_path: Path to save figure
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        summary_df = self.generate_quality_summary()
        
        # Prepare data for visualization
        plot_data = []
        for _, row in summary_df.iterrows():
            for domain in QualityDomain:
                domain_name = domain.value
                if domain_name in row:
                    plot_data.append({
                        'Study': row['study_id'],
                        'Domain': domain_name.replace('_', ' ').title(),
                        'Risk Level': row[domain_name],
                        'Risk Label': self._numeric_to_risk(row[domain_name])
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Risk distribution by domain
        ax1 = axes[0]
        risk_counts = plot_df.groupby(['Domain', 'Risk Label']).size().unstack()
        risk_counts.plot(kind='bar', stacked=True, ax=ax1,
                        color={'Low': 'green', 'Some Concerns': 'yellow', 'High': 'red'})
        ax1.set_title('Risk of Bias Distribution by Domain')
        ax1.set_ylabel('Number of Studies')
        ax1.legend(title='Risk Level')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Overall risk summary
        ax2 = axes[1]
        overall_counts = summary_df['overall_risk'].value_counts()
        colors = [self._risk_to_color(r) for r in overall_counts.index]
        overall_counts.plot(kind='pie', ax=ax2, colors=colors, autopct='%1.1f%%')
        ax2.set_title('Overall Risk of Bias Distribution')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    # Helper methods
    def _assess_domain(self, 
                      domain_name: str,
                      domain_criteria: Dict,
                      study_data: Dict) -> Dict:
        """Assess a specific quality domain"""
        questions = domain_criteria.get('questions', [])
        responses = []
        total_score = 0
        max_score = 0
        
        for question in questions:
            response = self._evaluate_question(question, study_data)
            responses.append(response)
            
            if response.get('applicable', True):
                score = response.get('score', 0)
                weight = question.get('weight', 1.0)
                total_score += score * weight
                max_score += question.get('max_score', 1) * weight
        
        # Calculate risk level
        if max_score > 0:
            percentage = (total_score / max_score) * 100
            risk_level = self._score_to_risk_level(percentage, domain_criteria)
        else:
            risk_level = RiskLevel.UNKNOWN.value
        
        return {
            'risk_level': risk_level,
            'score': total_score,
            'max_score': max_score,
            'percentage': percentage if max_score > 0 else None,
            'responses': responses,
            'criteria_used': domain_criteria
        }
    
    def _evaluate_question(self, 
                          question: Dict,
                          study_data: Dict) -> Dict:
        """Evaluate a single assessment question"""
        # Implementation depends on question type
        question_type = question.get('type', 'boolean')
        
        if question_type == 'boolean':
            return self._evaluate_boolean_question(question, study_data)
        elif question_type == 'numeric':
            return self._evaluate_numeric_question(question, study_data)
        elif question_type == 'categorical':
            return self._evaluate_categorical_question(question, study_data)
        else:
            return {
                'question': question.get('text'),
                'score': 0,
                'applicable': False,
                'comment': f"Unknown question type: {question_type}"
            }
    
    def _evaluate_boolean_question(self, 
                                  question: Dict,
                                  study_data: Dict) -> Dict:
        """Evaluate yes/no question"""
        # Extract evidence from study data
        evidence = self._extract_evidence(
            question.get('evidence_fields', []),
            study_data
        )
        
        # Determine if criteria are met
        criteria_met = self._check_criteria(
            question.get('criteria', {}),
            evidence
        )
        
        score = question.get('score_if_true', 1) if criteria_met else 0
        
        return {
            'question': question.get('text'),
            'score': score,
            'applicable': True,
            'criteria_met': criteria_met,
            'evidence': evidence,
            'comment': question.get('comment', '')
        }
    
    def _extract_evidence(self, 
                         evidence_fields: List[str],
                         study_data: Dict) -> Dict:
        """Extract evidence for assessment"""
        evidence = {}
        
        for field in evidence_fields:
            if '.' in field:
                # Handle nested fields
                parts = field.split('.')
                current = study_data
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        current = None
                        break
                evidence[field] = current
            else:
                evidence[field] = study_data.get(field)
        
        return evidence
    
    def _check_criteria(self, 
                       criteria: Dict,
                       evidence: Dict) -> bool:
        """Check if criteria are met based on evidence"""
        # Implementation depends on criteria structure
        # This is a simplified version
        return True  # Placeholder
    
    def _score_to_risk_level(self, 
                           percentage: float,
                           domain_criteria: Dict) -> str:
        """Convert score percentage to risk level"""
        thresholds = domain_criteria.get('risk_thresholds', {})
        
        if percentage >= thresholds.get('low', 80):
            return RiskLevel.LOW.value
        elif percentage >= thresholds.get('some_concerns', 50):
            return RiskLevel.SOME_CONCERNS.value
        else:
            return RiskLevel.HIGH.value
    
    def _determine_overall_risk(self, domain_assessments: Dict) -> str:
        """Determine overall risk based on domain assessments"""
        # Use worst-case approach (any high risk -> overall high)
        risk_levels = [a['risk_level'] for a in domain_assessments.values()]
        
        if RiskLevel.HIGH.value in risk_levels:
            return RiskLevel.HIGH.value
        elif RiskLevel.SOME_CONCERNS.value in risk_levels:
            return RiskLevel.SOME_CONCERNS.value
        else:
            return RiskLevel.LOW.value
    
    def _risk_to_numeric(self, risk_level: str) -> float:
        """Convert risk level to numeric score (0-1)"""
        mapping = {
            RiskLevel.LOW.value: 1.0,
            RiskLevel.SOME_CONCERNS.value: 0.5,
            RiskLevel.HIGH.value: 0.0,
            RiskLevel.UNKNOWN.value: np.nan
        }
        return mapping.get(risk_level, np.nan)
    
    def _numeric_to_risk(self, score: float) -> str:
        """Convert numeric score to risk label"""
        if score >= 0.8:
            return "Low"
        elif score >= 0.5:
            return "Some Concerns"
        else:
            return "High"
    
    def _risk_to_color(self, risk_level: str) -> str:
        """Get color for risk level"""
        colors = {
            "Low": "green",
            "Some Concerns": "yellow",
            "High": "red",
            "Unknown": "gray"
        }
        return colors.get(risk_level, "gray")
    
    def _calculate_agreement(self, scores: List[float]) -> float:
        """Calculate agreement between reviewers"""
        if len(scores) < 2:
            return 1.0
        
        # Simple agreement measure (1 - normalized variance)
        variance = np.var(scores)
        max_variance = np.var([0, 1])  # Maximum possible variance for 0-1 scale
        
        if max_variance == 0:
            return 1.0
        
        return 1 - (variance / max_variance)
    
    def _calculate_consensus(self, assessments: List[Dict]) -> Dict:
        """Calculate consensus from multiple assessments"""
        # Use mode for categorical, mean for numeric
        consensus = assessments[0].copy()
        
        for domain_name in consensus['domains'].keys():
            domain_scores = []
            for assessment in assessments:
                if domain_name in assessment['domains']:
                    score = self._risk_to_numeric(
                        assessment['domains'][domain_name]['risk_level']
                    )
                    domain_scores.append(score)
            
            if domain_scores:
                mean_score = np.mean(domain_scores)
                consensus['domains'][domain_name]['risk_level'] = self._numeric_to_risk(
                    mean_score
                ).lower()
        
        # Recalculate overall risk
        consensus['overall_risk'] = self._determine_overall_risk(
            consensus['domains']
        )
        
        return consensus
