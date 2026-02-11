# Welcome to the TSFEL tutorial at BIOSTEC 2026!

This tutorial demonstrates time series feature extraction using TSFEL through a guided hands-on notebook.

You can complete the assignment using one of the following environments:

- 🟡 Google Colab
- 🟢 Local environment
- 🔵 Binder


## 🟡 Google Colab

No installation required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fraunhoferportugal/tsfel-tutorials/blob/master/BIOSTEC-2026/feature_extraction_example_colab.ipynb)

Open the notebook and run all cells sequentially.

## 🟢 Run Locally

### Set up a dedicated conda environment 
```bash
conda create -n <environment_name> python=3.10 -y
conda activate <environment_name>
pip install tsfel jupyter
```

### Clone the repository and launch Jupyter notebook

```bash
git clone https://github.com/fraunhoferportugal/tsfel-tutorials.git
cd tsfel-tutorials/BIOSTEC-2026
jupyter feature_extraction_example.ipynb
```

## 🔵 Binder

Launch a temporary Jupyter environment in your browser:

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fraunhoferportugal/tsfel-tutorials/HEAD)

After launching:

1. Navigate to `BIOSTEC-2026/`
2. Open `feature_extraction_example.ipynb`
3. Execute the notebook cells in order

> Note: Binder sessions are temporary. Download results if needed.
