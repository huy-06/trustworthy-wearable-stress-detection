# WESAD Wrist-Based Physiological Feature Dataset

This directory contains the preprocessed, windowed, and synchronized feature dataset derived from the benchmark **WESAD (Wearable Stress and Affect Detection)** dataset [Schmidt et al., ICMI 2018], recorded using the medical-grade **Empatica E4** wristband.

---

## 1. Dataset Overview

* **File:** `wesad_wrist_features.csv`
* **Total Observation Windows:** $N = 5,294$ windows
* **Window Duration:** 60 seconds with 5-second step size (sliding window scheme).
* **Participants:** 15 authentic subjects ($S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S14, S15, S16, S17$).
  * *Note:* Subject $S12$ was omitted by the original WESAD dataset creators due to hardware sensor failure during data acquisition.
* **Ground Truth Classes:**
  * `0`: Non-Stress / Baseline / Amusement ($N = 3,744$, $70.7\%$)
  * `1`: Acute Psychosocial Stress ($N = 1,550$, $29.3\%$, Trier Social Stress Test - TSST)

---

## 2. Feature Specification (25 Engineered Features across 4 Sensor Channels)

| Modality | Sampling Rate ($f_s$) | Extracted Features ($D = 25$) | Description |
| :--- | :--- | :--- | :--- |
| **EDA** (Electrodermal Activity) | 4 Hz | `eda_tonic_mean`, `eda_tonic_std`, `eda_phasic_mean`, `eda_phasic_std`, `eda_phasic_max`, `eda_phasic_energy`, `eda_scr_peaks_count` | Continuous cvxEDA decomposition isolating tonic baseline drift from phasic Skin Conductance Responses (SCR). |
| **BVP / HRV** (Blood Volume Pulse) | 64 Hz | `hrv_mean_hr`, `hrv_sdnn`, `hrv_rmssd`, `hrv_pnn50`, `hrv_lf`, `hrv_hf`, `hrv_lf_hf` | Inter-Beat Intervals (IBI) time and frequency domain metrics reflecting autonomic sympathetic and parasympathetic regulation. |
| **TEMP** (Skin Temperature) | 4 Hz | `temp_mean`, `temp_std`, `temp_min`, `temp_max`, `temp_slope` | Peripheral vasoconstriction and thermal dissipation metrics. |
| **ACC** (Tri-axial Accelerometer) | 32 Hz | `acc_mag_mean`, `acc_mag_std`, `acc_mag_var`, `acc_mag_max`, `acc_mag_energy`, `acc_mag_zcr` | Kinematic acceleration Euclidean norm $||\mathbf{a}|| = \sqrt{a_x^2 + a_y^2 + a_z^2}$ capturing motion context. |

---

## 3. High-Motion Boundary Definition

Physical motion confounds are quantified via the acceleration magnitude variance:
$$\text{Var}(\text{ACC}) \ge \theta_{\text{motion}} \quad (\theta_{\text{motion}} = 80^{\text{th}}\text{ percentile of training folds})$$

This isolates 295 high-motion non-stress epochs ($N_{\text{motion}} = 295$) used to strictly compute the motion-induced false alarm rate ($\text{FAR}_{\text{motion}}$).

