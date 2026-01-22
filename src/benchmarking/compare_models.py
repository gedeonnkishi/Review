"""
Benchmarking script for comparing CeNN with classical and quantum models.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

class ModelBenchmark:
    """
    Benchmark various forecasting models.
    """
    
    def __init__(self):
        self.results = {}
        
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Evaluate model predictions.
        
        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            model_name: Name of the model
            
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': self._calculate_mape(y_true, y_pred),
            'R2': self._calculate_r2(y_true, y_pred)
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def _calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Avoid division by zero
        mask = y_true != 0
        if np.any(mask):
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return np.nan
    
    def _calculate_r2(self, y_true, y_pred):
        """Calculate R-squared coefficient."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    def compare_models(self):
        """
        Compare all benchmarked models.
        
        Returns:
            DataFrame-like comparison table
        """
        comparison = []
        for model_name, metrics in self.results.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison.append(row)
        
        return comparison
    
    def time_execution(self, func, *args, **kwargs):
        """
        Time the execution of a function.
        
        Args:
            func: Function to time
            *args, **kwargs: Function arguments
            
        Returns:
            Execution time in seconds
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        return end_time - start_time, result
