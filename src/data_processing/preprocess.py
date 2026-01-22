"""
Data preprocessing utilities for time series forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class TimeSeriesPreprocessor:
    """
    Preprocess time series data for forecasting.
    """
    
    def __init__(self, scaling_method='minmax'):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method: 'minmax' or 'standard'
        """
        self.scaling_method = scaling_method
        self.scaler = None
        
    def create_sliding_windows(self, data, window_size, horizon=1):
        """
        Create sliding windows for time series forecasting.
        
        Args:
            data: Time series data (1D array)
            window_size: Size of input window
            horizon: Forecasting horizon
            
        Returns:
            X, y: Input windows and target values
        """
        X, y = [], []
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size:i + window_size + horizon])
        
        return np.array(X), np.array(y)
    
    def scale_data(self, data, fit=True):
        """
        Scale time series data.
        
        Args:
            data: Time series data
            fit: Whether to fit the scaler
            
        Returns:
            Scaled data
        """
        if self.scaling_method == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            scaler = StandardScaler()
        
        if fit:
            self.scaler = scaler
            # Reshape for sklearn
            data_reshaped = data.reshape(-1, 1)
            return scaler.fit_transform(data_reshaped).flatten()
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            data_reshaped = data.reshape(-1, 1)
            return self.scaler.transform(data_reshaped).flatten()
    
    def inverse_scale(self, data):
        """
        Inverse scaling transformation.
        
        Args:
            data: Scaled data
            
        Returns:
            Original scale data
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted.")
        
        data_reshaped = data.reshape(-1, 1)
        return self.scaler.inverse_transform(data_reshaped).flatten()
    
    def add_temporal_features(self, data, timestamps=None):
        """
        Add temporal features to time series.
        
        Args:
            data: Time series data
            timestamps: Optional datetime index
            
        Returns:
            DataFrame with added features
        """
        df = pd.DataFrame({'value': data})
        
        if timestamps is not None:
            df.index = timestamps
            # Extract time features
            df['hour'] = df.index.hour
            df['dayofweek'] = df.index.dayofweek
            df['month'] = df.index.month
            df['dayofyear'] = df.index.dayofyear
            df['weekofyear'] = df.index.isocalendar().week
            
        # Statistical features
        df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
        df['rolling_std_7'] = df['value'].rolling(window=7).std()
        df['lag_1'] = df['value'].shift(1)
        df['lag_7'] = df['value'].shift(7)
        
        # Remove NaN values
        df = df.dropna()
        
        return df
