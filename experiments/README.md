# Experiments

This directory contains code to reproduce experiments from the paper:

> **"Systematic Review of Quantum Machine Learning Techniques for Time Series Forecasting and a Neuro-Symbolic CeNN Emulation Framework"**

## Directory Structure
experiments/
├── paper_figures/ # Generate paper figures
│ ├── init.py
│ ├── generate_all_figures.py # Main script for figures
│ └── plot_utils.py
├── table_results/ # Generate result tables
│ ├── init.py
│ ├── generate_tables.py # Main script for tables
│ └── table_formatters.py
└── ablation_studies/ # Ablation studies
├── init.py
├── run_ablation.py # Main script for ablation studies
└── ablation_config.yaml


## Quick Start

### 1. Generate All Figures

```bash
python experiments/paper_figures/generate_all_figures.py
This will generate:

Figure 2: Classical ML pipeline for time series forecasting

Figure 4: Quantum-Classical transition diagram

Figure 5: Bloch sphere representation

Figure 7: VQC pipeline diagram

Figure 11: GPU scaling performance

2. Generate All Tables
python experiments/table_results/generate_tables.py
This will generate:

Table 2: Comparative performance across domains

Table 3: Aggregated QML performance metrics

Table 6: Direct comparison with QML-TSF studies

Configuration Table: CeNN hyperparameters per dataset

3. Run Ablation Studies
python experiments/ablation_studies/run_ablation.py
This will run:

Template A parameter study

Activation function comparison

Regularization parameter sensitivity

Grid size analysis

Output Files
Generated files will be saved in:
figures/ - All generated figures (PNG and PDF)

tables/ - Generated tables (Markdown and LaTeX)

ablation_results/ - Ablation study results and plots
Dependencies
Make sure you have installed the required packages:
pip install numpy pandas matplotlib seaborn scipy scikit-learn tabulate
For 3D plots (Figure 5), you need:
pip install matplotlib
Reproducibility
All scripts use fixed random seeds for reproducibility. To ensure consistent results:

Set set_random_seed(42) at the beginning of each script

Use the same Python environment

Use the same package versions

Customization
You can modify the scripts to:
Use your own datasets

Change experimental parameters

Add new figures or tables

Extend ablation studies

Citation
If you use this code in your research, please cite our paper:
@article{nkishidjenga2025systematic,
  title={Systematic Review of Quantum Machine Learning Techniques for Time Series Forecasting and a Neuro-Symbolic CeNN Emulation Framework},
  author={Nkishi Djenga, Gédéon and Maruba Kambale, Exaucé and Bieto Bisuta, Hughes and Witesyavwirwa Kambale, Vianney and Kyamakya, Kyandoghere},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
License
This code is released under the MIT License. See the main LICENSE file for details.
