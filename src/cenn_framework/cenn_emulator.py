"""
Main CeNN Emulator class for quantum-inspired time series forecasting.
"""

import numpy as np
from typing import List, Tuple, Optional
from .cenn_cell import CeNNCell

class CeNNEmulator:
    """
    Quantum-inspired CeNN Emulator for time series forecasting.
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (8, 8),
                 template_A: List[float] = None,
                 template_B: List[float] = None,
                 activation: str = 'tanh',
                 alpha: float = 1.0):
        """
        Initialize CeNN Emulator.
        
        Args:
            grid_size: Size of CeNN grid (rows, columns)
            template_A: Feedback template [left, center, right]
            template_B: Control template [left, center, right]
            activation: Activation function
            alpha: Decay rate parameter
        """
        self.grid_size = grid_size
        self.rows, self.cols = grid_size
        self.template_A = template_A or [0.4, 1.0, 0.4]
        self.template_B = template_B or [0.2, 0.5, 0.2]
        self.activation = activation
        self.alpha = alpha
        
        # Initialize grid of cells
        self.grid = self._initialize_grid()
        
        # History for analysis
        self.state_history = []
        self.output_history = []
        
    def _initialize_grid(self) -> np.ndarray:
        """Initialize grid of CeNN cells."""
        grid = np.empty(self.grid_size, dtype=object)
        for i in range(self.rows):
            for j in range(self.cols):
                grid[i, j] = CeNNCell(
                    alpha=self.alpha,
                    template_A=self.template_A,
                    template_B=self.template_B,
                    activation=self.activation
                )
        return grid
    
    def reset_grid(self) -> None:
        """Reset all cells in the grid."""
        for i in range(self.rows):
            for j in range(self.cols):
                self.grid[i, j].reset()
        
        # Clear history
        self.state_history = []
        self.output_history = []
    
    def get_neighbors(self, i: int, j: int) -> Tuple[List[float], List[float]]:
        """
        Get outputs and inputs from neighboring cells.
        
        Args:
            i: Row index
            j: Column index
            
        Returns:
            Tuple of (neighbor_outputs, neighbor_inputs)
        """
        # Default: left, self, right neighbors (for 1D simulation)
        # In 2D grid, we consider horizontal neighbors for simplicity
        neighbor_outputs = []
        neighbor_inputs = []
        
        # Left neighbor
        if j > 0:
            neighbor_outputs.append(self.grid[i, j-1].output)
            neighbor_inputs.append(self.grid[i, j-1].state)
        else:
            neighbor_outputs.append(0.0)
            neighbor_inputs.append(0.0)
            
        # Self
        neighbor_outputs.append(self.grid[i, j].output)
        neighbor_inputs.append(self.grid[i, j].state)
        
        # Right neighbor
        if j < self.cols - 1:
            neighbor_outputs.append(self.grid[i, j+1].output)
            neighbor_inputs.append(self.grid[i, j+1].state)
        else:
            neighbor_outputs.append(0.0)
            neighbor_inputs.append(0.0)
            
        return neighbor_outputs, neighbor_inputs
    
    def step(self, inputs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform one time step of CeNN evolution.
        
        Args:
            inputs: External inputs to cells (same shape as grid)
            
        Returns:
            Grid outputs after step
        """
        if inputs is None:
            inputs = np.zeros(self.grid_size)
        
        new_states = np.zeros(self.grid_size)
        new_outputs = np.zeros(self.grid_size)
        
        # Update each cell
        for i in range(self.rows):
            for j in range(self.cols):
                neighbor_outputs, neighbor_inputs = self.get_neighbors(i, j)
                
                state, output = self.grid[i, j].forward(
                    x=self.grid[i, j].state,
                    neighbor_outputs=neighbor_outputs,
                    neighbor_inputs=neighbor_inputs,
                    bias=inputs[i, j]
                )
                
                new_states[i, j] = state
                new_outputs[i, j] = output
        
        # Store history
        self.state_history.append(new_states.copy())
        self.output_history.append(new_outputs.copy())
        
        return new_outputs
    
    def evolve(self, steps: int, inputs: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """
        Evolve CeNN for multiple steps.
        
        Args:
            steps: Number of time steps
            inputs: List of input arrays for each step
            
        Returns:
            Final outputs
        """
        if inputs is None:
            inputs = [None] * steps
        elif len(inputs) != steps:
            raise ValueError(f"Expected {steps} input arrays, got {len(inputs)}")
        
        for t in range(steps):
            self.step(inputs[t])
        
        return self.get_output()
    
    def get_output(self) -> np.ndarray:
        """Get current grid output."""
        outputs = np.zeros(self.grid_size)
        for i in range(self.rows):
            for j in range(self.cols):
                outputs[i, j] = self.grid[i, j].output
        return outputs
    
    def get_state(self) -> np.ndarray:
        """Get current grid state."""
        states = np.zeros(self.grid_size)
        for i in range(self.rows):
            for j in range(self.cols):
                states[i, j] = self.grid[i, j].state
        return states
    
    def forecast(self, series: np.ndarray, 
                 forecast_horizon: int = 24,
                 window_size: int = 24) -> np.ndarray:
        """
        Forecast time series using CeNN.
        
        Args:
            series: Input time series (1D array)
            forecast_horizon: Number of steps to forecast
            window_size: Size of input window
            
        Returns:
            Forecasted values
        """
        if len(series) < window_size:
            raise ValueError(f"Series length ({len(series)}) must be >= window_size ({window_size})")
        
        # Reset grid
        self.reset_grid()
        
        # Use last window as initial condition
        last_window = series[-window_size:]
        
        # Encode window into grid (simplified encoding)
        encoded_input = self._encode_timeseries(last_window)
        
        # Evolve CeNN
        predictions = []
        for t in range(forecast_horizon):
            # Step with encoded input
            output = self.step(encoded_input)
            
            # Extract prediction (simplified: mean of grid)
            prediction = np.mean(output)
            predictions.append(prediction)
            
            # Update encoded input for next step (shift window)
            encoded_input = np.roll(encoded_input, -1)
            encoded_input[-1] = prediction
        
        return np.array(predictions)
    
    def _encode_timeseries(self, series: np.ndarray) -> np.ndarray:
        """
        Encode time series into grid input.
        
        Args:
            series: Time series to encode
            
        Returns:
            Encoded grid
        """
        encoded = np.zeros(self.grid_size)
        
        # Simple encoding: reshape if possible, otherwise interpolate
        if len(series) == self.rows * self.cols:
            encoded = series.reshape(self.grid_size)
        else:
            # Linear interpolation
            from scipy import interpolate
            x_old = np.linspace(0, 1, len(series))
            x_new = np.linspace(0, 1, self.rows * self.cols)
            f = interpolate.interp1d(x_old, series, kind='linear')
            encoded = f(x_new).reshape(self.grid_size)
        
        return encoded
