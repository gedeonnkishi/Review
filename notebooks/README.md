# Interactive Notebooks

This directory contains Jupyter notebooks for interactive exploration of the CeNN framework.

## Notebooks Overview

### 1. `01_getting_started.ipynb`
**Beginner-friendly introduction** to the CeNN framework.

**What you'll learn:**
- Basic installation and setup
- Creating and configuring CeNN emulators
- Visualizing CeNN grid dynamics
- Time series forecasting with CeNN
- Benchmarking against classical methods
- Spectral analysis of predictions

**Prerequisites:** Basic Python knowledge

**Estimated time:** 30-45 minutes

### 2. `02_reproduce_paper.ipynb`
**Complete reproduction** of paper results.

**What you'll reproduce:**
- All tables from the paper (Tables 2, 3, 6)
- All key figures (Figures 2, 4, 5, 7, 11)
- Ablation studies and sensitivity analysis
- Computational efficiency results
- Reproducibility validation

**Prerequisites:** Completion of Notebook 01 or familiarity with CeNN

**Estimated time:** 60-90 minutes

### 3. `03_custom_dataset.ipynb`
**Template for using your own data** with CeNN.

**What you'll learn:**
- Loading data from various formats (CSV, Excel, JSON)
- Preprocessing time series data
- Customizing CeNN for your specific use case
- Exporting and analyzing results

**Prerequisites:** Your own time series data

**Estimated time:** 30-60 minutes

## Quick Start

### Running Notebooks Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/quantum-inspired-tsf-framework.git
   cd quantum-inspired-tsf-framework
2. **Create and activate environment:**
   ```bash
conda env create -f environment.yml
conda activate cenn-tsf
3. **pip install jupyter notebook**
```bash
pip install jupyter notebook
4. **Launch Jupyter:**
```bash
jupyter notebook
5. **Open notebooks in your browser**

**Running on Google Colab**
Upload notebooks to Google Drive

Open in Google Colab

Install dependencies:

```bash
!pip install numpy pandas matplotlib seaborn scikit-learn scipy
**Notebook Structure**
Each notebook follows this structure:

Setup: Import libraries and configure environment

Data Preparation: Load and preprocess data

Experimentation: Run CeNN models

Visualization: Plot results

Analysis: Interpret findings

Export: Save results for later use

**Interactive Features**
The notebooks include:

Progress bars for long-running operations

Interactive widgets for parameter tuning

Real-time visualization of results

Export functionality for saving figures and data

**Tips for Best Results**
Run cells in order: Notebooks are designed to be executed sequentially

Modify parameters: Experiment with different CeNN configurations

Save outputs: Export interesting results for documentation

Check dependencies: Ensure all required packages are installed

Monitor resources: Some operations may be memory-intensive

**Troubleshooting**
Common Issues:
Import errors: Make sure you've added the src directory to your Python path

Memory issues: Reduce grid size or batch size for large datasets

Slow execution: Use GPU acceleration if available

Plotting errors: Ensure matplotlib is properly installed

**Getting Help:**
Check the main README.md for installation instructions

Review the documentation in docs/

Open an issue on GitHub for bugs

Contact the authors for research questions

**Citation**
If you use these notebooks in your research, please cite:
```bash
@article{nkishidjenga2025systematic,
  title={Systematic Review of Quantum Machine Learning Techniques for Time Series Forecasting and a Neuro-Symbolic CeNN Emulation Framework},
  author={Nkishi Djenga, Gédéon and Maruba Kambale, Exaucé and Bieto Bisuta, Hughes and Witesyavwirwa Kambale, Vianney and Kyamakya, Kyandoghere},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
**License**
These notebooks are released under the MIT License. See the main LICENSE file for details.
