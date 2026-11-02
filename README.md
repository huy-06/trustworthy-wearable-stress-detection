# Selective Classification and Probability Calibration for Trustworthy Stress Detection from Wrist-Worn Physiological Signals

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Framework: Scikit--Learn / XGBoost](https://img.shields.io/badge/Framework-XGBoost%20%7C%20LightGBM%20%7C%20PyTorch-orange.svg)](https://scikit-learn.org/)
[![Dataset: WESAD](https://img.shields.io/badge/Dataset-WESAD%20(Empatica%20E4)-brightgreen.svg)](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)
[![IEEE Standard Format](https://img.shields.io/badge/Manuscript-IEEE%20Format-red.svg)](Latex/main_en.tex)
[![Streamlit Demo](https://img.shields.io/badge/App-Streamlit%20Clinical%20Dashboard-FF4B4B.svg)](App/app.py)

---

## 📌 Overview

Continuous ambulatory stress monitoring via wrist-worn consumer wearables is heavily constrained by **motion artifacts and metabolic mimicry** (physical activity elevating heart rate and skin conductance), causing excessive false alarms and debilitating **alarm fatigue**. 

This research project introduces an end-to-end **Trustworthy Stress Detection Framework** evaluated on the public **WESAD (Empatica E4)** dataset ($N = 5,294$ windows across 15 participants) under rigorous **15-fold Leave-One-Subject-Out (LOSO)** cross-validation. The framework unites:
1. **Kinematic Context Integration:** Incorporating 6 tri-axial acceleration features alongside 19 autonomic physiological indicators (EDA, BVP/HRV, ST) to enable boosted decision trees to learn multi-modal interaction boundaries.
2. **Probability Calibration:** Isotonic regression alignment in a nested leak-free scheme, reducing Expected Calibration Error (ECE) from $0.0984 \rightarrow 0.0744$.
3. **Selective Classification (Learning with Rejection):** An abstention mechanism parameterized by confidence threshold $\tau = 0.80$, withholding decisions on physiologically ambiguous windows ($1 - \tau < \hat{p}_1 < \tau$).

### 🚀 Key Scientific Findings
* **Suppression of Motion False Alarms:** Motion-induced false alarm rate ($\text{FAR}_{\text{motion}}$) plummets by **$66.6\%$** (from $12.20\%$ down to **$4.07\%$**, $p = 0.0039$).
* **Clinical Diagnostic Discrimination:** Selective Macro $F_1$-score elevates from $88.13\% \rightarrow \mathbf{95.20\%}$ (Stress $F_1 = \mathbf{93.15\%}$, Selective Accuracy = $\mathbf{96.08\%}$) while preserving **$69.42\%$ operational coverage**.
* **Universal Architecture Generalization:** Evaluated across **8 distinct architectures** (7 tabular ML + 1 Deep Learning MLP); selective abstention consistently suppresses motion false alarms across all models (by $55.0\%$ to $90.0\%$).
* **Wearable Edge Feasibility:** Unlike heavy end-to-end deep networks, the 25-feature tabular XGBoost model can run directly on ultra-low-power microcontrollers (ARM Cortex-M4/M33 within a sub-milliwatt power budget).

---

## 🏗️ System Architecture & Workflow

```text
========================================================================================================
[EMPATICA E4 WRIST SENSORS]
├── BVP (64 Hz)  ──> Bandpass (0.5 - 8 Hz) ──> Inter-Beat Intervals (IBI) ──> HRV Metrics (SDNN, RMSSD, HF, LF)
├── EDA (4 Hz)   ──> High-Pass Filtering   ──> Tonic (SCL) & Phasic (SCR) ──> Tonic Mean, SCR Peaks, Phasic Energy
├── TEMP (4 Hz)  ──> Anti-aliasing Filter  ──> Thermal Dissipation        ──> Mean, Std, Max, Min, Slope
└── ACC (32 Hz)  ──> Euclidean Norm ||a||  ──> Kinematic Magnitude        ──> Mean, Var, Max, ZCR, Energy
========================================================================================================
                                             │
                                             ▼
[SLIDING WINDOW SEGMENTATION (60s Window, 5s Stride) ──> D = 25 Engineered Features]
                                             │
                                             ▼
[NESTED LEAVE-ONE-SUBJECT-OUT (15-FOLD LOSO) VALIDATION]
  ├── Outer Fold: 14 Training Subjects | 1 Completely Unseen Test Subject (S2..S17)
  └── Inner Fold: GroupKFold by Subject (Preventing window overlap leakage)
                                             │
                                             ▼
[PRIMARY ENSEMBLE CLASSIFIER (XGBoost / LightGBM / Random Forest)]
                                             │
                                             ▼ Raw Confidence Score
[NON-PARAMETRIC PROBABILITY CALIBRATION (Isotonic Regression)]
                                             │
                                             ▼ Well-Calibrated Posterior p̂₁ ∈ [0, 1]
[SELECTIVE CLASSIFICATION (Learning with Rejection @ Threshold τ = 0.80)]
  ├── If p̂₁ ≥ 0.80          ──> Predict Acute Psychosocial Stress (ŷ = 1)
  ├── If (1 - p̂₁) ≥ 0.80    ──> Predict Non-Stress / Relaxed Baseline (ŷ = 0)
  └── Otherwise (0.20 < p̂₁ < 0.80) ──> WITHHOLD DECISION [Uncertain / Abstain] (30.6% epochs)
========================================================================================================
                                             │
                                             ▼
[CLINICAL TRIAGE OUTPUT: Precision = 96.08% | FAR_motion = 4.07% (-66.6%) | Coverage = 69.42%]
```

---

## 📊 Comprehensive Experimental Benchmark (Matching Paper Table II & III)

### 1. Multi-Model Benchmark Across 8 Architectures (15-Fold LOSO)

| Algorithm | Full Macro $F_1$ | Full ECE | Full $\text{FAR}_{\text{motion}}$ | Calibrated ECE | Coverage ($\tau = 0.80$) | Selective Macro $F_1$ | Selective $\text{FAR}_{\text{motion}}$ | FAR Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Majority)** | 41.42% | 0.4140 | 0.00% | 0.0000 | 100.00% | 41.42% | 0.00% | -- |
| **Logistic Regression** | 88.00% | 0.0514 | 23.05% | 0.0521 | 72.48% | 92.23% | 10.17% | -55.9% |
| **Decision Tree** | 81.09% | 0.1542 | 22.71% | 0.1030 | 50.70% | 85.94% | 3.05% | -86.6% |
| **k-NN** | 85.48% | 0.0821 | 20.34% | 0.0512 | 64.94% | 93.69% | 9.15% | -55.0% |
| **AdaBoost** | 89.66% | 0.2710 | 13.90% | 0.0715 | 77.13% | 93.29% | 1.69% | -87.8% |
| **Random Forest** | 88.01% | 0.0620 | 13.90% | 0.0410 | 75.03% | 94.49% | 3.39% | -75.6% |
| **LightGBM** | 85.66% | 0.0915 | 13.56% | 0.0686 | 63.15% | 93.65% | **1.36%** | **-90.0%** |
| **MLP (Deep Learning)** | 88.95% | 0.0312 | 24.00% | 0.0336 | 75.16% | 95.38% | 5.84% | -75.7% |
| **XGBoost (Proposed)** | **88.13%** | **0.0612** | **12.20%** | **0.0744** | **69.42%** | **95.20%** | **4.07%** | **-66.6%** |

> **Pareto Trade-Off Note:** While LightGBM attains the lowest absolute $\text{FAR}_{\text{motion}}$ ($1.36\%$), it sacrifices significant clinical coverage ($63.15\%$) and diagnostic discrimination ($93.65\%$). XGBoost occupies the optimal Pareto frontier, retaining $+6.27\%$ higher usable monitoring windows and superior Stress $F_1$ ($93.15\%$).

### 2. Selective Operating Trade-Off Profile for XGBoost (Table III)

| Threshold Mode | Operational Coverage | Macro $F_1$ | Stress $F_1$ | ECE | $\text{FAR}_{\text{motion}}$ | Clinical Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Raw Baseline (100%)** | 100.0% | 88.13% | 84.99% | 0.0612 | 12.20% | Forced prediction, vulnerable to motion artifacts |
| **Calibrated (100%)** | 100.0% | 89.88% | 86.68% | 0.0744 | 10.85% | Better probability reliability, moderate FAR drop |
| **Selective ($\tau = 0.60$)** | 89.76% | 92.86% | 90.45% | 0.0534 | 7.46% | Initial abstention on highly uncertain epochs |
| **Selective ($\tau = 0.70$)** | 81.79% | 93.68% | 91.28% | 0.0406 | 4.41% | Balanced triage for ambulatory screening |
| **Optimal Selective ($\tau = 0.80$)** | **69.42%** | **95.20%** | **93.15%** | **0.0290** | **4.07%** | **Primary recommended clinical operating point** |
| **Conservative ($\tau = 0.90$)** | 50.38% | 96.93% | 95.54% | 0.0161 | 0.68% | High-security clinical audit mode |

---

## 📁 Repository Directory Structure

```text
├── README.md                          # Comprehensive project documentation
├── LICENSE                            # MIT Open-Source License
├── requirements.txt                   # Complete Python dependencies
├── .gitignore                         # Git exclusion rules
├── run_app.bat                        # Windows one-click launcher for Streamlit App
│
├── Dataset/                           # Preprocessed feature data
│   ├── wesad_wrist_features.csv       # 5,294 observation windows (Empatica E4)
│   ├── experiment_results_summary.json# Serialized metrics summary
│   └── README.md                      # Dataset specifications & feature dictionary
│
├── Notebook/                          # Sequential, fully-executed research notebooks
│   ├── 01_feature_extraction_wesad.ipynb
│   ├── 02_multimodel_benchmark.ipynb
│   ├── 03_loso_evaluation_and_calibration.ipynb
│   ├── 04_ablation_study_acceleration.ipynb
│   ├── 05_shap_interpretability_and_failure_analysis.ipynb
│   ├── 06_deep_learning_mlp_benchmark.ipynb
│   └── archive/                       # Early experimental prototypes
│
├── Models/                            # Serialized pipelines & inference modules
│   ├── trustworthy_xgb_pipeline.joblib# Trained calibrated XGBoost pipeline
│   ├── trustworthy_rf_pipeline.joblib # Trained Random Forest baseline pipeline
│   ├── xgboost_base_model.json        # Native XGBoost tree definitions
│   ├── config_inference.json          # Operating parameters & thresholds
│   └── predict.py                     # Standalone CLI inference module
│
├── Results/                           # Complete empirical raw outputs & CSVs
│   ├── table_benchmark_7_models.csv   # Benchmark across all algorithms
│   ├── LOSO/                          # 15-fold subject-level CSVs & predictions
│   ├── Ablation/                      # Multi-seed ablation CSVs & stats
│   ├── SHap/                          # TreeSHAP attribution & failure cases
│   └── DeepLearning/                  # MLP parametric baseline metrics & tests
│
├── App/                               # Interactive Clinical Decision Support Web App
│   └── app.py                         # Streamlit application with real-time triage
│
├── scripts/                           # Reproducibility automation scripts
│   └── generate_polished_figures.py   # Regenerates all 8 paper figures at 300 DPI
│
├── Latex/                             # Camera-ready publication manuscripts (IEEE format)
│   ├── main_en.tex                    # Final English IEEE Conference Manuscript
│   ├── main_vi.tex                    # Vietnamese Manuscript
│   ├── Images/                        # 300 DPI publication-grade vector/raster plots
│   └── archive/                       # Historical draft revisions
│
├── Paper/                             # Compiled PDF manuscripts
│   ├── DSR301m_Paper_EN.pdf           # Camera-ready English paper (PDF)
│   ├── DSR301m_Paper_VI.pdf           # Vietnamese version (PDF)
│   └── archive/                       # Historical PDF builds
│
└── Slide/                             # Research presentation materials
    └── DSR301m_Research_Presentation.pdf
```

---

## 💻 Quick Start & Reproducibility

### 1. Environment Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/huy-06/trustworthy-wearable-stress-detection.git
cd trustworthy-wearable-stress-detection

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Clinical Web Application

Launch the Streamlit interactive dashboard with one click:
* **Windows:** Double-click `run_app.bat` or run:
```bash
streamlit run App/app.py
```
* **Features included in the App:**
  * Interactive confidence threshold slider ($\tau \in [0.50, 0.95]$).
  * Real-time physiological sample injection & decision triage (*Stress*, *Non-Stress*, *Uncertain / Abstain*).
  * Feature importance breakdown with radar chart & dynamic SHAP explanation.
  * Motion context indicator highlighting kinetic artifact suppression.

### 3. Running Research Notebooks

The notebooks in `Notebook/` are self-contained and pre-executed with all outputs preserved:
* `01_feature_extraction_wesad.ipynb`: Processes raw WESAD pickle files into tabular windows.
* `02_multimodel_benchmark.ipynb`: Trains and evaluates the 7 tabular architectures.
* `03_loso_evaluation_and_calibration.ipynb`: Primary LOSO 15-fold evaluation, Isotonic calibration, and Risk-Coverage curve.
* `04_ablation_study_acceleration.ipynb`: Runs multi-seed ablation (ACC context vs. Physio-only).
* `05_shap_interpretability_and_failure_analysis.ipynb`: Global TreeSHAP attributions and failure audit on Subject S14.
* `06_deep_learning_mlp_benchmark.ipynb`: Evaluates the deep MLP baseline under identical conditions.

### 4. Regenerating High-Resolution Figures

To regenerate all 8 figures in `Latex/Images/` at 300 DPI with vector-crisp typography:

```bash
python scripts/generate_polished_figures.py
```

---

## 🔬 Citation & Academic Credit

If you find this codebase or research methodology beneficial for your work, please cite:

```bibtex
@inproceedings{dsr301m_trustworthy_stress_2026,
  author    = {DSR301m Research Team},
  title     = {Selective Classification and Probability Calibration for Trustworthy Stress Detection from Wrist-Worn Physiological Signals},
  booktitle = {Proceedings of the IEEE Conference on Biomedical and Health Informatics (BHI)},
  year      = {2026},
  pages     = {1--9},
  publisher = {IEEE}
}
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

