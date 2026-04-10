# AMOMA-DPK: Trading Off Privacy, Utility, and Stability in Privacy-Preserving K-Means Clustering Using an Adaptive Multi-Objective Memetic Algorithm

This repository contains the code used for the journal paper submitted to **WWWJ (World Wide Web Journal)**:

**Trading Off Privacy, Utility, and Stability in Privacy-Preserving $K$-Means Clustering Using an Adaptive Multi-Objective Memetic Algorithm**

The project implements the proposed **AMOMA-DPK** framework and the scripts used to reproduce the main experimental results and figures in the paper.

## Repository Structure

```text
├── dataset/                                      # Datasets used in the experiments
├── result/                                       # Saved experimental results
├── 1.AMOMA_DPK.py                                # Main implementation of AMOMA-DPK and experiment pipeline
├── 2.Figure1_pareto_front_3D&2D.py               # Generate 3D and 2D Pareto-front visualizations
├── 3.Figure2_HyperValue_comparison_3D.py         # Generate 3D hypervolume comparison results/figures
├── 4.Figure3_compare_HV_MemoryLength.py          # Hyperparameter analysis: memory length
├── 5.Figure4_compare_HV_ImbalancedFactor.py      # Hyperparameter analysis: imbalanced local-search factor
```

## Requirements

* Python >= 3.10

* Main Python packages:

  ```bash
  pip install numpy pandas matplotlib scikit-learn scipy
  ```

* Standard library modules used in the code include:
  `os`, `glob`, `time`, `random`, `math`, `copy`, and `multiprocessing`

Depending on your environment and code version, you may need to install additional packages referenced in the scripts.

## Usage

### 1. Run the main experiment

Before running, configure the dataset name, seeds, privacy-budget settings, and other experiment parameters inside the script.

```bash
python 1.AMOMA_DPK.py
```

This script runs the main optimization procedure and stores the output files in the `result/` folder.

### 2. Generate figures

After the experiment results are available, run the plotting scripts below. Please adjust the input CSV path or related settings in each script if needed.

**Figure 1: Pareto fronts (3D and 2D projections)**

```bash
python 2.Figure1_pareto_front_3D&2D.py
```

**Figure 2: 3D hypervolume comparison**

```bash
python 3.Figure2_HyperValue_comparison_3D.py
```

**Figure 3: Hypervolume comparison for memory length**

```bash
python 4.Figure3_compare_HV_MemoryLength.py
```

**Figure 4: Hypervolume comparison for imbalanced factor**

```bash
python 5.Figure3_compare_HV_ImbalancedFactor.py
```

Generated figures are typically saved to a newly created output folder defined inside the corresponding script.

## Datasets

The `dataset/` folder contains the datasets used in the experiments. In the paper, the experiments are conducted on two real-world datasets. They are also publicly available from:

* Heart Disease Dataset (UCI Machine Learning Repository) https://archive.ics.uci.edu/dataset/45/heart+disease

* Diabetes Classification Dataset (Kaggle) https://www.kaggle.com/datasets/simaanjali/diabetes-classification-dataset

Please make sure the dataset filenames and paths in the scripts match the files placed in the `dataset/` directory.

## Notes

* The main script includes the implementation of the proposed adaptive multi-objective memetic optimization framework for differentially private $K$-means clustering.

* The figure scripts are designed to reproduce the corresponding plots used in the paper.

* Some script parameters are intentionally left configurable so that users can rerun the experiments under different seeds, privacy budgets, or hyperparameter settings.

## Citation

If you use this code in your research, please cite the corresponding journal paper after publication.

## License

This repository is shared for research and academic use. Please add your preferred license statement here if the project will be released publicly under a specific license.
