# Neuro-Symbolic CeNN Framework for Quantum-Inspired Time Series Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## ⚠️ Important Notice: Nature of This Repository

**This repository contains CLASSICAL EMULATIONS of quantum-inspired machine learning models.** 
No code is executed on actual quantum hardware. The goal is to provide a reproducible framework for studying quantum-inspired machine learning techniques that can run on classical hardware (CPU/GPU).

### What This Repository Contains:
- ✅ **Classical emulation** of quantum-inspired Cellular Neural Networks (CeNN)
- ✅ **GPU-accelerated** simulations of quantum-inspired dynamics
- ✅ **Benchmark comparisons** against simulated quantum circuits
- ✅ **Reproducible experiments** from our systematic review

### What This Repository Does NOT Contain:
- ❌ **Real quantum hardware** implementations
- ❌ **Execution on quantum processors** (IBM, Rigetti, etc.)
- ❌ **Physical quantum circuit** deployments

## 📋 Project Overview

This repository accompanies the research paper:

> **"Systematic Review of Quantum Machine Learning Techniques for Time Series Forecasting and a Neuro-Symbolic CeNN Emulation Framework"**

This work makes three main contributions:
1. **PRISMA-based systematic review** of QML-TSF studies
2. **Neuro-symbolic CeNN framework** for large-scale quantum-inspired emulation
3. **Complete reproducibility** package for all experiments

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/gedeonnkishi/Review.git
cd Review

# Create and activate conda environment
conda env create -f environment.yml
conda activate cenn-tsf

## Basic Usage
from src.cenn_framework import CeNNEmulator
# Initialize CeNN emulator
emulator = CeNNEmulator(
    grid_size=(8, 8),
    template_A=[0.4, 1.0, 0.4],
    template_B=[0.2, 0.5, 0.2],
    activation='tanh'
)

# Load your time series data
import numpy as np
time_series = np.load('data/your_timeseries.npy')

# Run forecasting
predictions = emulator.forecast(
    series=time_series,
    forecast_horizon=24,
    window_size=24
)

## 📁 Repository Structure
quantum-inspired-tsf-framework/
├── src/                          # Main source code
│   ├── cenn_framework/          # CeNN emulation framework
│   ├── benchmarking/            # Benchmarking scripts
│   ├── data_processing/         # Data loading and preprocessing
│   └── utils/                   # Utility functions
├── experiments/                  # Reproduce paper experiments
│   ├── paper_figures/           # Generate paper figures
│   ├── table_results/           # Generate result tables
│   └── ablation_studies/        # Ablation studies
├── notebooks/                    # Interactive tutorials
├── tests/                       # Unit tests
├── configs/                     # Configuration files
└── data/                        # Data processing scripts

## 📊 Reproducing Paper Results
To reproduce all experiments from the paper:
# Run main experiments
python experiments/paper_figures/generate_all_figures.py

# Generate result tables
python experiments/table_results/generate_tables.py

# Run ablation studies
python experiments/ablation_studies/run_ablation.py

📈 Key Results You Can Reproduce
Table 2: Comparative performance across domains (Energy, Finance, Traffic, Weather)

Table 3: Aggregated QML performance metrics (2019-2025)

Figure 4: GPU scaling performance

Figure 5: Sensitivity analysis (λ parameter)

Table 6: Direct comparison with representative QML-TSF studies

## 🔬 Scientific Methodology
Our framework implements:

1. CeNN State Dynamics
Continuous-time dynamics with linear complexity O(n):
ẋᵢ(t) = -αxᵢ(t) + Σ Aᵢⱼyⱼ(t) + Σ Bᵢⱼuⱼ(t) + Iᵢ
2. Neuro-Symbolic Integration
KL-divergence alignment with reference quantum distributions

Spectral Similarity Index (S) for dynamic fidelity

Symbolic constraints (causality, unitarity proxy, trend preservation)

3. GPU Optimization
Kernel fusion for reduced VRAM bandwidth

Multi-GPU scaling via NCCL (>92% efficiency)

CUDA-accelerated CeNN cell updates
## 📝 Citation
If you use this code in your research, please cite our paper:
@article{nkishidjenga2025systematic,
  title={Systematic Review of Quantum Machine Learning Techniques for Time Series Forecasting and a Neuro-Symbolic CeNN Emulation Framework},
  author={Nkishi Djenga, Gédéon and Maruba Kambale, Exaucé and Bieto Bisuta, Hughes and Witesyavwirwa Kambale, Vianney and Kyamakya, Kyandoghere},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing
Contributions are welcome! Please read our Contributing Guidelines for details.

## 🙏 Acknowledgments
University of Kinshasa (UNIKIN) for research support

Tshwane University of Technology (TUT) for computational resources

University of Klagenfurt for academic guidance

All authors of the reviewed studies in our systematic review
## ❓ FAQ
Q: Is this real quantum computing?
A: No, this is classical emulation of quantum-inspired models. It runs entirely on classical hardware.

Q: Can I run this on quantum hardware?
A: No, the framework is designed for GPU/CPU emulation. For real quantum implementations, see Qiskit or PennyLane documentation.

Q: How does this compare to real QML implementations?
A: Our emulation provides 55× speedup over simulated VQCs but doesn't claim quantum advantage. It's a practical alternative for large-scale forecasting.
Q: What datasets are included?
A: We provide scripts to download standard benchmarks (GEFCom2014, etc.) but not the datasets themselves due to licensing.
