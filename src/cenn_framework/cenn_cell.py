"""
CeNN Cell Implementation - Quantum-inspired cellular neural network cell.
"""

import numpy as np
import torch

class CeNNCell:
    """
    A single CeNN cell implementing quantum-inspired dynamics.
    """
    
    def __init__(self, alpha=1.0, template_A=None, template_B=None, activation='tanh'):
        """
        Initialize CeNN cell.
        
        Args:
            alpha: Decay rate parameter
            template_A: Feedback template (3-element list)
            template_B: Control template (3-element list)
            activation: Activation function ('tanh', 'sigmoid', 'relu')
        """
        self.alpha = alpha
        self.template_A = template_A or [0.4, 1.0, 0.4]
        self.template_B = template_B or [0.2, 0.5, 0.2]
        self.activation = activation
        
        # State variables
        self.state = 0.0
        self.output = 0.0
        
    def forward(self, x, neighbor_outputs, neighbor_inputs, bias=0.0):
        """
        Forward pass of CeNN cell.
        
        Args:
            x: Internal state
            neighbor_outputs: List of neighbor outputs [left, self, right]
            neighbor_inputs: List of neighbor inputs [left, self, right]
            bias: Bias term
            
        Returns:
            Updated state and output
        """
        # Template operations
        feedback = sum(a * y for a, y in zip(self.template_A, neighbor_outputs))
        control = sum(b * u for b, u in zip(self.template_B, neighbor_inputs))
        
        # State update (discrete approximation)
        dx = -self.alpha * x + feedback + control + bias
        self.state = x + dx
        
        # Apply activation
        if self.activation == 'tanh':
            self.output = np.tanh(self.state)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-self.state))
        elif self.activation == 'relu':
            self.output = max(0, self.state)
        else:
            self.output = self.state
            
        return self.state, self.output
    
    def reset(self):
        """Reset cell state."""
        self.state = 0.0
        self.output = 0.0
