from src.cenn_framework import CeNNEmulator
import numpy as np

# Test the CeNN emulator
print("Testing CeNN Emulator...")

# Create emulator
emulator = CeNNEmulator(
    grid_size=(8, 8),
    template_A=[0.4, 1.0, 0.4],
    template_B=[0.2, 0.5, 0.2],
    activation='tanh'
)

# Create test time series
time_series = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)

# Test forecasting
try:
    predictions = emulator.forecast(
        series=time_series,
        forecast_horizon=24,
        window_size=24
    )
    print(f"✓ Forecast successful!")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  First 5 predictions: {predictions[:5]}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n✓ All tests completed!")
