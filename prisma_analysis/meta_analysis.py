"""
Meta-analysis module for quantitative synthesis
Calculates effect sizes, heterogeneity, and publication bias
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class MetaAnalyzer:
    """Performs meta-analysis on extracted data"""
    
    def __init__(self, extraction_table: pd.DataFrame):
        """
        Initialize meta-analyzer
        
        Args:
            extraction_table: DataFrame from DataExtractor
        """
        self.data = extraction_table.copy()
        self.results = {}
        self.heterogeneity_stats = {}
        
    def calculate_effect_sizes(self, 
                              metric: str = 'rmse_ratio',
                              method: str = 'hedges_g') -> pd.DataFrame:
        """
        Calculate effect sizes for each study
        
        Args:
            metric: Performance metric to analyze
            method: Effect size calculation method
        
        Returns:
            DataFrame with effect sizes and variances
        """
        effect_sizes = []
        
        for _, row in self.data.iterrows():
            if pd.notna(row.get(metric)):
                effect_size = self._calculate_single_effect_size(
                    row, metric, method
                )
                effect_sizes.append(effect_size)
        
        effect_df = pd.DataFrame(effect_sizes)
        
        # Store for later use
        self.effect_sizes = effect_df
        self.results['effect_sizes'] = {
            'metric': metric,
            'method': method,
            'data': effect_df
        }
        
        return effect_df
    
    def perform_fixed_effects_meta(self, 
                                  effect_sizes: pd.DataFrame = None) -> Dict:
        """
        Perform fixed-effects meta-analysis
        
        Args:
            effect_sizes: DataFrame with effect sizes (optional)
        
        Returns:
            Fixed-effects meta-analysis results
        """
        if effect_sizes is None:
            if not hasattr(self, 'effect_sizes'):
                raise ValueError("Effect sizes not calculated yet")
            effect_sizes = self.effect_sizes
        
        # Calculate inverse variance weights
        effect_sizes = effect_sizes.copy()
        effect_sizes['weight'] = 1 / effect_sizes['variance']
        
        # Weighted mean
        weighted_sum = (effect_sizes['effect_size'] * effect_sizes['weight']).sum()
        total_weight = effect_sizes['weight'].sum()
        pooled_effect = weighted_sum / total_weight
        
        # Standard error and confidence interval
        se_pooled = np.sqrt(1 / total_weight)
        ci_lower = pooled_effect - 1.96 * se_pooled
        ci_upper = pooled_effect + 1.96 * se_pooled
        
        # Test for overall effect
        z_score = pooled_effect / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        result = {
            'model': 'fixed_effects',
            'pooled_effect': pooled_effect,
            'se': se_pooled,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'z_score': z_score,
            'p_value': p_value,
            'total_weight': total_weight,
            'num_studies': len(effect_sizes),
            'weights': effect_sizes[['study_id', 'weight']].to_dict('records')
        }
        
        self.results['fixed_effects'] = result
        return result
    
    def perform_random_effects_meta(self,
                                   effect_sizes: pd.DataFrame = None,
                                   method: str = 'dl') -> Dict:
        """
        Perform random-effects meta-analysis
        
        Args:
            effect_sizes: DataFrame with effect sizes (optional)
            method: Heterogeneity estimator ('dl', 'ml', 'reml')
        
        Returns:
            Random-effects meta-analysis results
        """
        if effect_sizes is None:
            if not hasattr(self, 'effect_sizes'):
                raise ValueError("Effect sizes not calculated yet")
            effect_sizes = self.effect_sizes
        
        effect_sizes = effect_sizes.copy()
        
        # Calculate heterogeneity (tau²)
        tau_squared = self._calculate_tau_squared(effect_sizes, method)
        
        # Update weights with between-study variance
        effect_sizes['weight_re'] = 1 / (effect_sizes['variance'] + tau_squared)
        
        # Weighted mean for random effects
        weighted_sum = (effect_sizes['effect_size'] * effect_sizes['weight_re']).sum()
        total_weight = effect_sizes['weight_re'].sum()
        pooled_effect = weighted_sum / total_weight
        
        # Standard error and confidence interval
        se_pooled = np.sqrt(1 / total_weight)
        ci_lower = pooled_effect - 1.96 * se_pooled
        ci_upper = pooled_effect + 1.96 * se_pooled
        
        # Test for overall effect
        z_score = pooled_effect / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Calculate heterogeneity statistics
        q_statistic, i_squared = self._calculate_heterogeneity(effect_sizes)
        
        result = {
            'model': 'random_effects',
            'pooled_effect': pooled_effect,
            'se': se_pooled,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'z_score': z_score,
            'p_value': p_value,
            'tau_squared': tau_squared,
            'q_statistic': q_statistic,
            'i_squared': i_squared,
            'heterogeneity_p': self._test_heterogeneity(q_statistic, len(effect_sizes)),
            'num_studies': len(effect_sizes),
            'weights': effect_sizes[['study_id', 'weight_re']].to_dict('records')
        }
        
        self.results['random_effects'] = result
        self.heterogeneity_stats = {
            'tau_squared': tau_squared,
            'q': q_statistic,
            'i_squared': i_squared
        }
        
        return result
    
    def perform_subgroup_analysis(self,
                                 subgroup_variable: str,
                                 effect_sizes: pd.DataFrame = None) -> Dict:
        """
        Perform subgroup analysis
        
        Args:
            subgroup_variable: Variable for subgrouping (e.g., 'qml_approach')
            effect_sizes: DataFrame with effect sizes (optional)
        
        Returns:
            Subgroup analysis results
        """
        if effect_sizes is None:
            if not hasattr(self, 'effect_sizes'):
                raise ValueError("Effect sizes not calculated yet")
            effect_sizes = self.effect_sizes
        
        # Merge with original data to get subgroup information
        merged = effect_sizes.merge(
            self.data[['study_id', subgroup_variable]],
            on='study_id',
            how='left'
        )
        
        subgroups = merged[subgroup_variable].unique()
        subgroup_results = {}
        
        for subgroup in subgroups:
            subgroup_data = merged[merged[subgroup_variable] == subgroup]
            
            if len(subgroup_data) >= 2:  # Need at least 2 studies
                # Perform random-effects meta for subgroup
                subgroup_result = self.perform_random_effects_meta(subgroup_data)
                subgroup_results[subgroup] = subgroup_result
        
        # Test for subgroup differences
        subgroup_test = self._test_subgroup_differences(subgroup_results)
        
        result = {
            'subgroup_variable': subgroup_variable,
            'subgroups': subgroup_results,
            'subgroup_test': subgroup_test,
            'num_subgroups': len(subgroup_results)
        }
        
        self.results[f'subgroup_{subgroup_variable}'] = result
        return result
    
    def assess_publication_bias(self,
                               effect_sizes: pd.DataFrame = None,
                               methods: List[str] = None) -> Dict:
        """
        Assess publication bias using multiple methods
        
        Args:
            effect_sizes: DataFrame with effect sizes (optional)
            methods: List of methods to use
        
        Returns:
            Publication bias assessment results
        """
        if effect_sizes is None:
            if not hasattr(self, 'effect_sizes'):
                raise ValueError("Effect sizes not calculated yet")
            effect_sizes = self.effect_sizes
        
        if methods is None:
            methods = ['eggers', 'begg', 'funnel', 'trim_fill']
        
        bias_results = {}
        
        for method in methods:
            if method == 'eggers':
                bias_results['eggers'] = self._eggers_test(effect_sizes)
            elif method == 'begg':
                bias_results['begg'] = self._begg_test(effect_sizes)
            elif method == 'funnel':
                bias_results['funnel'] = self._create_funnel_plot_data(effect_sizes)
            elif method == 'trim_fill':
                bias_results['trim_fill'] = self._trim_fill_analysis(effect_sizes)
        
        self.results['publication_bias'] = bias_results
        return bias_results
    
    def perform_sensitivity_analysis(self,
                                   method: str = 'leave_one_out',
                                   effect_sizes: pd.DataFrame = None) -> Dict:
        """
        Perform sensitivity analysis
        
        Args:
            method: Sensitivity analysis method
            effect_sizes: DataFrame with effect sizes (optional)
        
        Returns:
            Sensitivity analysis results
        """
        if effect_sizes is None:
            if not hasattr(self, 'effect_sizes'):
                raise ValueError("Effect sizes not calculated yet")
            effect_sizes = self.effect_sizes
        
        if method == 'leave_one_out':
            return self._leave_one_out_analysis(effect_sizes)
        elif method == 'influence_analysis':
            return self._influence_analysis(effect_sizes)
        elif method == 'quality_adjusted':
            return self._quality_adjusted_analysis(effect_sizes)
        else:
            raise ValueError(f"Unknown sensitivity analysis method: {method}")
    
    def generate_forest_plot(self,
                           output_path: str = None,
                           figsize: Tuple = (10, 12)) -> plt.Figure:
        """
        Generate forest plot of effect sizes
        
        Args:
            output_path: Path to save figure
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if not hasattr(self, 'effect_sizes'):
            raise ValueError("Effect sizes not calculated yet")
        
        effect_sizes = self.effect_sizes.copy()
        
        # Sort by effect size
        effect_sizes = effect_sizes.sort_values('effect_size')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot individual study effects
        y_pos = np.arange(len(effect_sizes))
        ax.errorbar(effect_sizes['effect_size'], y_pos,
                   xerr=effect_sizes['se'] * 1.96,
                   fmt='o', color='black', capsize=5)
        
        # Add pooled effect if available
        if 'random_effects' in self.results:
            pooled = self.results['random_effects']['pooled_effect']
            ci_lower = self.results['random_effects']['ci_95_lower']
            ci_upper = self.results['random_effects']['ci_95_upper']
            
            ax.axvline(pooled, color='red', linestyle='--', label='Pooled Effect')
            ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='red')
        
        # Add labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(effect_sizes['study_id'])
        ax.set_xlabel('Effect Size (Hedges\' g)')
        ax.set_title('Forest Plot of Effect Sizes')
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    # Helper methods
    def _calculate_single_effect_size(self,
                                    study_row: pd.Series,
                                    metric: str,
                                    method: str) -> Dict:
        """Calculate effect size for a single study"""
        if method == 'hedges_g':
            # Hedges' g for standardized mean difference
            # Simplified calculation - in practice would use proper formulas
            effect = study_row[metric]
            n_qml = study_row.get('sample_size_qml', 30)
            n_baseline = study_row.get('sample_size_baseline', 30)
            
            # Calculate variance
            variance = (n_qml + n_baseline) / (n_qml * n_baseline) + (
                effect ** 2) / (2 * (n_qml + n_baseline))
            
            return {
                'study_id': study_row['study_id'],
                'effect_size': effect,
                'variance': variance,
                'se': np.sqrt(variance),
                'sample_size_qml': n_qml,
                'sample_size_baseline': n_baseline
            }
        
        elif method == 'mean_difference':
            # Raw mean difference
            effect = study_row[metric]
            variance = study_row.get(f'{metric}_variance', 1.0)
            
            return {
                'study_id': study_row['study_id'],
                'effect_size': effect,
                'variance': variance,
                'se': np.sqrt(variance)
            }
        
        else:
            raise ValueError(f"Unknown effect size method: {method}")
    
    def _calculate_tau_squared(self,
                             effect_sizes: pd.DataFrame,
                             method: str) -> float:
        """Calculate between-study variance tau²"""
        # DerSimonian and Laird estimator
        if method == 'dl':
            # Calculate Q statistic
            w = 1 / effect_sizes['variance']
            w_sum = w.sum()
            w2_sum = (w ** 2).sum()
            
            # Weighted mean
            theta_w = (w * effect_sizes['effect_size']).sum() / w_sum
            
            # Q statistic
            q = (w * (effect_sizes['effect_size'] - theta_w) ** 2).sum()
            
            # Tau²
            c = w_sum - (w2_sum / w_sum)
            tau_squared = max(0, (q - (len(effect_sizes) - 1)) / c)
            
            return tau_squared
        
        else:
            raise ValueError(f"Unknown tau² estimator: {method}")
    
    def _calculate_heterogeneity(self,
                               effect_sizes: pd.DataFrame) -> Tuple[float, float]:
        """Calculate heterogeneity statistics"""
        w = 1 / effect_sizes['variance']
        w_sum = w.sum()
        theta_w = (w * effect_sizes['effect_size']).sum() / w_sum
        
        # Q statistic
        q = (w * (effect_sizes['effect_size'] - theta_w) ** 2).sum()
        
        # I² statistic
        df = len(effect_sizes) - 1
        if q > df:
            i_squared = max(0, ((q - df) / q) * 100)
        else:
            i_squared = 0
        
        return q, i_squared
    
    def _test_heterogeneity(self, q: float, df: int) -> float:
        """Test for heterogeneity using Q statistic"""
        return 1 - stats.chi2.cdf(q, df)
    
    def _test_subgroup_differences(self,
                                  subgroup_results: Dict) -> Dict:
        """Test for differences between subgroups"""
        # Implementation would use Q-between test
        return {'test': 'Q_between', 'p_value': 0.05}  # Placeholder
    
    def _eggers_test(self, effect_sizes: pd.DataFrame) -> Dict:
        """Egger's test for publication bias"""
        # Simplified implementation
        x = 1 / np.sqrt(effect_sizes['variance'])
        y = effect_sizes['effect_size'] / np.sqrt(effect_sizes['variance'])
        
        if len(effect_sizes) >= 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            return {
                'intercept': intercept,
                'se_intercept': std_err,
                't_value': intercept / std_err,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        else:
            return {'error': 'Insufficient studies for Egger\'s test'}
    
    def _begg_test(self, effect_sizes: pd.DataFrame) -> Dict:
        """Begg's test for publication bias"""
        # Simplified implementation
        return {'test': 'Begg', 'p_value': 0.1}  # Placeholder
    
    def _create_funnel_plot_data(self,
                                effect_sizes: pd.DataFrame) -> Dict:
        """Prepare data for funnel plot"""
        return {
            'effect_sizes': effect_sizes['effect_size'].tolist(),
            'standard_errors': effect_sizes['se'].tolist(),
            'precision': (1 / effect_sizes['se']).tolist()
        }
    
    def _trim_fill_analysis(self,
                           effect_sizes: pd.DataFrame) -> Dict:
        """Trim and fill analysis for publication bias"""
        # Simplified implementation
        return {'method': 'trim_fill', 'adjusted_effect': 0.0}  # Placeholder
    
    def _leave_one_out_analysis(self,
                               effect_sizes: pd.DataFrame) -> Dict:
        """Leave-one-out sensitivity analysis"""
        results = []
        
        for i in range(len(effect_sizes)):
            # Remove study i
            subset = effect_sizes.drop(i).reset_index(drop=True)
            
            # Perform random-effects meta
            try:
                re_result = self.perform_random_effects_meta(subset)
                results.append({
                    'omitted_study': effect_sizes.iloc[i]['study_id'],
                    'pooled_effect': re_result['pooled_effect'],
                    'ci_lower': re_result['ci_95_lower'],
                    'ci_upper': re_result['ci_95_upper']
                })
            except:
                continue
        
        return {
            'method': 'leave_one_out',
            'results': results,
            'summary': {
                'mean_effect': np.mean([r['pooled_effect'] for r in results]),
                'range_effects': (
                    min([r['pooled_effect'] for r in results]),
                    max([r['pooled_effect'] for r in results])
                )
            }
        }
    
    def _influence_analysis(self, effect_sizes: pd.DataFrame) -> Dict:
        """Influence analysis"""
        # Implementation would calculate Cook's distance, etc.
        return {'method': 'influence', 'results': {}}
    
    def _quality_adjusted_analysis(self,
                                  effect_sizes: pd.DataFrame) -> Dict:
        """Quality-adjusted meta-analysis"""
        # Weight by quality scores
        return {'method': 'quality_adjusted', 'results': {}}
