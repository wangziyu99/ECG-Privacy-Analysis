# ECG Signal Analysis and Classification

## Overview

This repository provides a comprehensive framework for analyzing and classifying ECG signals using state-of-the-art preprocessing techniques and machine learning models. The primary goal is to extract PQRST features from ECG data and use them for classification tasks, leveraging models like Logistic Regression, Decision Trees, and Random Forests. The framework also integrates SHAP (SHapley Additive exPlanations) to explain model predictions.

---

## Key Features

- **ECG Signal Preprocessing**:
  - PQRST feature extraction using NeuroKit2.
  - Cleaning and delineating ECG signals for robust feature extraction.

- **Classification Models**:
  - Logistic Regression
  - Decision Tree
  - Random Forest

- **SHAP Explanations**:
  - Detailed feature impact visualization using SHAP.
  - Summary plots to interpret model predictions.

- **Scalable Workflow**:
  - Efficient preprocessing and classification pipeline designed for ECG datasets.

---

## Dependencies

Install the following dependencies before running the project:

```bash
pip install neurokit2 heartpy matplotlib wfdb scikit-learn shap
```

---

## Dataset

The framework is designed to process `.dat` files containing ECG signals. Ensure your dataset is structured and stored in a folder that the script can access. The project uses files from the MIT-BIH Arrhythmia Database as an example.

---

## Usage

### Step 1: Data Preprocessing

The script extracts PQRST features from raw ECG signals:

- **Inputs**: `.dat` files with ECG signals.
- **Outputs**: Processed feature arrays used for training and testing machine learning models.

### Step 2: Run the Analysis

Execute the script to perform preprocessing, classification, and SHAP analysis:

```bash
python run.py
```

### Step 3: Interpret Results

- **Classification Metrics**:
  - Accuracy
  - Confusion Matrix
- **SHAP Visualizations**:
  - Overall feature impact.
  - Detailed feature explanations for individual predictions.

---

## Code Structure

```
├── run.py               # Main script to execute the pipeline.
├── data/                # Directory to store the .dat files for ECG signals.
├── results/             # Directory to save outputs and SHAP visualizations.
└── README.md            # Documentation for the project.
```

---

## Example Output

### Classification Results

```
Random Forest:
Accuracy: 0.85
Confusion Matrix:
[[30  5]
 [ 4 26]]
```

### SHAP Visualization

- Bar chart showing overall feature importance.
- Detailed SHAP summary plots for individual predictions.

---

## Citation

If you use this project in your research or development, please cite it as follows:

```
@software{your_project,
  title={ECG Signal Analysis and Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/ecg-analysis}
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

For questions or collaboration, reach out to:

- **Your Name**
- **Email**: ziyuloveu@gmail.com
