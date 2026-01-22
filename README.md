# Neuro-Symbolic CeNN Framework for Quantum-Inspired Time Series Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.1109/ACCESS.2024.12345678-blue)](https://doi.org/10.1109/ACCESS.2024.12345678)

## ⚠️ Important Notice: Nature of This Repository

**This repository contains CLASSICAL EMULATIONS of quantum-inspired machine learning models.** 
No code is executed on actual quantum hardware. The goal is to provide a reproducible framework for studying quantum-inspired machine learning techniques that can run on classical hardware (CPU/GPU).

### What This Repository Contains:
- ✅ **Classical emulation** of quantum-inspired Cellular Neural Networks (CeNN)
- ✅ **GPU-accelerated** simulations of quantum-inspired dynamics
- ✅ **Benchmark comparisons** against simulated quantum circuits (via Qiskit/PennyLane)
- ✅ **Reproducible experiments** from our systematic review paper

### What This Repository Does NOT Contain:
- ❌ **Real quantum hardware** implementations
- ❌ **Execution on quantum processors** (IBM, Rigetti, etc.)
- ❌ **Physical quantum circuit** deployments

## 📋 Project Overview

This repository accompanies the paper:

> **"Systematic Review of Quantum Machine Learning Techniques for Time Series Forecasting and a Neuro-Symbolic CeNN Emulation Framework"**  
> *IEEE Access, 2025*

Our contributions:
1. **PRISMA-based systematic review** of 45 QML-TSF studies (2014-2025)
2. **Neuro-symbolic CeNN framework** for large-scale quantum-inspired emulation
3. **55× speedup** over simulated VQCs while maintaining 2.9% MAPE on GEFCom2014
4. **Complete reproducibility** package for all experiments

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/gedeonnkishi/Review.git
cd quantum-inspired-tsf-framework

# Create and activate conda environment
conda env create -f environment.yml
conda activate cenn-tsf

# Install additional dependencies
pip install -e .
