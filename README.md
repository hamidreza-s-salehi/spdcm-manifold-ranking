# spdcm-manifold-ranking

**Predictive Parsimony in Resting-State Spectral Dynamic Causal Modeling: Regularized Manifold Ranking and Subspace Generative Embedding of Default Mode Network Hub Architectures**

**Authors:**  
**Hamidreza Sadeghsalehi**, **Fatemeh Ahmadi**, and **Arash Zare Sadeghi**

---

## Overview

In empirical resting-state spectral Dynamic Causal Modeling (spDCM), classical Random-Effects Bayesian Model Selection (RFX-BMS), balanced-prior family-wise BMS, analytical Bayesian Model Reduction (BMR), and second-level Parametric Empirical Bayes (PEB) frequently assign high expected posterior model frequency ($\langle r_1 \rangle > 0.81$) and near-unity exceedance probability ($\mathrm{XP}_1 \approx 1.00$) to the maximally complex, fully connected network architecture. This repository provides the computational, simulation, and visualization framework accompanying our study, showing how classical point-null Bayesian reduction and post-hoc regularized manifold ranking address complementary modeling objectives:

1. **Multivariate Savage–Dickey BMR, Balanced-Prior Family BMS, & Analytical PEB-Style Pooling Audits:** Overflow-safe evaluation of first-level analytical BMR and a custom analytical second-level PEB-style random-effects pooling comparator (evaluated under both variance-inflated plug-in and non-negative within-subject variance-subtracted method-of-moments estimators, alongside element-wise single-edge Savage–Dickey tests), demonstrating how simultaneous point-null pruning of 16 to 25 non-zero secondary couplings ($\sim 0.06\text{--}0.16\text{ Hz}$) induces cumulative quadratic Mahalanobis accuracy penalties that outweigh logarithmic Occam bonuses.
2. **Post-Hoc Regularized Manifold Ranking & Subspace Generative Embedding:**
   - **Shared-Dispersion Zero-Preserving Scaling:** Normalizes candidate parameter vectors by the unconstrained full-model cohort dispersion without model-specific mean centering so that structural zeros remain invariant at the origin without introducing differential dispersion scaling across hub and non-hub pathways.
   - **Rate-Distortion Self-Organizing Maps (SOM) & Principal-Subspace Gaussian Mixture Models (GMM):** Evaluates candidate topologies in a regularized 6D principal subspace ($p_k < N$, across prespecified diagonal [$p_3 = 38$] and BIC-preferred spherical [$p_3 = 23$] covariances, alongside tied, regularized full, and exploratory 36D full-space checks), penalized by active extrinsic directed degrees of freedom ($k_{\text{ext},m}$). The framework ranks previously inverted candidate architectures and can also be applied directly to BMR-reduced posteriors without separate de novo inversion of every sub-model.
3. **Multi-SNR Spectral-Domain Noise Perturbation Simulations & Misspecification Stress Tests:** Contrasts *Regime A (Strict Binary Sparsity)* against *Regime B (Graded Biological Hierarchy)* across $\text{SNR} \in \{0, 3, 6, 10\}\text{ dB}$ and heterogeneous subtype mixtures (50:50 and nearest-integer 25:75 and 75:25) using deterministically stratified 50/50 training ($n_{\text{train}} = 30$) and held-out test ($n_{\text{test}} = 30$) splits, including stress tests under regional hemodynamic transit-time jitter, colored innovation misspecification, and heavy-tailed Student-$t_4$ background coupling.
4. **Internal Consistency, Unstructured Linearized $L_1$ Surrogate, & External Cross-Spectral Transportability:** Evaluates 5-fold internal cross-validation (including a leave-one-model-out ensemble reference check), degree-matched ($k_{\text{ext}} = 5$) non-hub star controls, a 1,000-permutation Representational Similarity Analysis (RSA) random-subgraph null, and an unstructured linearized $L_1$ spectral regression surrogate in the primary IBID cohort ($N = 58$, plus an $N = 61$ BMR sensitivity analysis), alongside model-free observed Welch cross-spectral density validation in the public NYU/ADHD-200 resting-state cohort ($N = 40$; $n = 20$ typically developing controls and $n = 20$ individuals with ADHD) across within-cohort 5-fold cross-validation (Protocol A, uncalibrated and training-fold gain-calibrated), pure cross-cohort zero-shot prototype transfer from IBID without external model inversion (Protocol B), and within-subject reduced-model fits (Protocol C).

---

## Repository Structure

```text
spdcm-manifold-ranking/
├── data/
│   └── external_adhd_timeseries_40.pkl     # Extracted DMN ROI resting-state time series (NYU/ADHD-200, N = 40)
├── src/
│   ├── utils_spdcm.py                      # Biophysical CSD forward model, BMR, PEB, RFX-BMS, and statistical utilities
│   ├── step1_ibid_bmr_peb_and_manifold.py  # Primary cohort BMR/PEB audit, PCA dimensionality sweep, and SOM/GMM bootstrap
│   ├── step2_cross_spectral_cv_and_rsa.py  # 5-fold cross-spectral CV, 5-edge non-hub controls, L1 surrogate, edge audit, & RSA
│   ├── step3_biophysical_simulations.py    # Multi-SNR Regime A/B spectral simulations & misspecification stress tests
│   ├── step4_external_validation_adhd.py   # External Welch CSD Protocols A/B/C, nperseg/scan-length sensitivity, & LOSO transfer
│   └── step5_generate_figures.py           # Generates all 7 publication figures (300 DPI)
├── results/
│   └── spdcm_dmn_results.json              # Complete verified numerical outputs across all analyses
├── Figs/                                   # High-resolution publication figures (Fig1 to Fig7, 300 DPI PNG)
├── run_all.py                              # Master execution script (fixed global SEED = 42)
└── requirements.txt                        # Python package dependencies

```

---

## Installation & Reproducibility

1. **Clone the repository and install dependencies:**

```bash
git clone [https://github.com/hamidreza-s-salehi/spdcm-manifold-ranking.git](https://github.com/hamidreza-s-salehi/spdcm-manifold-ranking.git)
cd spdcm-manifold-ranking
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

2. **Run the complete analysis and figure-generation pipeline:**

```bash
python run_all.py

```

Running `run_all.py` executes all five modules with a fixed deterministic seed (`SEED = 42`), saves all verified numerical outputs to `results/spdcm_dmn_results.json`, and regenerates `Fig1` through `Fig7` in `Figs/`.

---

## Data Availability

* **Primary Cohort ($N = 58$, Iranian Brain Imaging Database [IBID]):** Resting-state fMRI data were obtained from Group 1 of the Iranian Brain Imaging Database (Batouli et al., 2021) hosted at the National Brain Mapping Laboratory (`https://nbml.ir`) and are governed by an institutional data-use agreement. If `data/model_space.mat` is placed in the `data/` directory, `run_all.py` executes the full primary IBID re-analysis; otherwise, `run_all.py` uses the precomputed IBID summary in `results/spdcm_dmn_results.json` and runs the spectral-domain simulations and public external cohort validation directly.
* **External Cohort ($N = 40$, NYU / ADHD-200 Repository):** Publicly available via Nilearn (`nilearn.datasets.fetch_adhd(n_subjects=40)`; Abraham et al., 2014). Pre-extracted 6-node DMN resting-state time series are included in `data/external_adhd_timeseries_40.pkl`.