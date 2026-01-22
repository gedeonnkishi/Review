"""
Utility functions for the CeNN framework.
"""

import os
import json
import yaml
import numpy as np
from typing import Dict, List, Any, Optional

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    _, ext = os.path.splitext(config_path)
    
    with open(config_path, 'r') as f:
        if ext.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif ext.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")

def save_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save experiment results to file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    _, ext = os.path.splitext(output_path)
    
    with open(output_path, 'w') as f:
        if ext.lower() == '.json':
            json.dump(results, f, indent=2)
        elif ext.lower() in ['.yaml', '.yml']:
            yaml.dump(results, f, default_flow_style=False)
        else:
            # Default to JSON
            json.dump(results, f, indent=2)

def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass

def calculate_spectral_similarity(signal1: np.ndarray, 
                                 signal2: np.ndarray, 
                                 sample_rate: float = 1.0) -> float:
    """
    Calculate spectral similarity between two signals.
    
    Args:
        signal1: First signal
        signal2: Second signal
        sample_rate: Sampling rate
        
    Returns:
        Spectral similarity index (0 to 1)
    """
    # Calculate power spectral density
    fft1 = np.fft.fft(signal1)
    fft2 = np.fft.fft(signal2)
    
    psd1 = np.abs(fft1) ** 2
    psd2 = np.abs(fft2) ** 2
    
    # Normalize
    psd1 = psd1 / np.sum(psd1)
    psd2 = psd2 / np.sum(psd2)
    
    # Calculate cosine similarity
    similarity = np.dot(psd1, psd2) / (np.linalg.norm(psd1) * np.norm(psd2))
    
    return float(similarity)

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate KL divergence between two distributions.
    
    Args:
        p: First distribution
        q: Second distribution
        epsilon: Small value to avoid log(0)
        
    Returns:
        KL divergence
    """
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    return np.sum(p * np.log(p / q))

def progress_bar(iteration: int, total: int, length: int = 50) -> str:
    """
    Create a progress bar string.
    
    Args:
        iteration: Current iteration
        total: Total iterations
        length: Length of progress bar
        
    Returns:
        Progress bar string
    """
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '░' * (length - filled_length)
    return f'|{bar}| {percent:.1f}% Complete'
