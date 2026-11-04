# spdcm-manifold-ranking

**Regularized Manifold Ranking and Subspace Generative Embedding in Resting-State Spectral Dynamic Causal Modeling: Disentangling Graded Hub Hierarchies from Bayesian Over-Parameterization in the Default Mode Network**

**Authors:**  
**Hamidreza Sadeghsalehi**, **Fatemeh Ahmadi**, and **Arash Zare Sadeghi**

---

## Overview

In empirical resting-state spectral Dynamic Causal Modeling (spDCM), classical Random-Effects Bayesian Model Selection (RFX-BMS), analytical Bayesian Model Reduction (BMR), and hierarchical second-level Parametric Empirical Bayes (PEB) frequently assign near-unity posterior probability to the maximally complex, fully connected network architecture. This repository provides the computational, simulation, and visualization framework accompanying our study:

1. **Multivariate Savage-Dickey BMR & Second-Level PEB Audits:** Overflow-safe evaluation of first-level analytical BMR and hierarchical second-level PEB (`spm_dcm_peb_bmc`), demonstrating how non-zero secondary background couplings ($\sim 0.06\text{--}0.16\text{ Hz}$) induce quadratic Mahalanobis accuracy penalties under sharp point-null reduction priors.
2. **Post-Hoc Regularized Manifold Ranking & Subspace Generative Embedding:**
   - **Shared-Dispersion Zero-Preserving Scaling:** Normalizes candidate parameter vectors by the unconstrained full-model cohort dispersion without model-specific mean centering so that structural zeros remain invariant at the origin.
   - **Rate-Distortion Self-Organizing Maps (SOM) & Principal-Subspace Gaussian Mixture Models (GMM):** Evaluates candidate topologies in both a well-conditioned 6D principal subspace ($p_k < N$) and the full 36D parameter space, penalized strictly by active extrinsic directed degrees of freedom ($k_{\text{ext},m}$).
3. **Multi-SNR Biophysical Cross-Spectral Simulations:** Contrasts *Regime A (Strict Binary Sparsity)* against *Regime B (Graded Biological Hierarchy)* across $\text{SNR} \in \{0, 3, 6, 10\}\text{ dB}$ and heterogeneous subtype mixtures (25:75, 50:50, 75:25) using deterministically stratified 50/50 training and held-out test splits.
4. **Out-of-Sample Cross-Spectral Validation & Cross-Cohort Transportability:** Evaluates 5-fold out-of-sample cross-spectral predictions against canonical and degree-matched ($k_{\text{ext}} = 5$) non-hub star controls and against model-free observed Welch cross-spectral densities in the public NYU/ADHD-200 resting-state cohort ($N = 40$, including $n = 20$ typically developing controls).

---

## Repository Structure

```text
spdcm-manifold-ranking/
├── data/
│   └── external_adhd_timeseries_40.pkl     # Extracted DMN ROI resting-state time series (NYU/ADHD-200, N = 40)
├── src/
│   ├── utils_spdcm.py                      # Biophysical CSD forward model, BMR, PEB, RFX-BMS, and statistical utilities
│   ├── step1_ibid_bmr_peb_and_manifold.py  # Primary cohort BMR/PEB audit, PCA dimensionality sweep, and SOM/GMM bootstrap
│   ├── step2_cross_spectral_cv_and_rsa.py  # 5-fold cross-spectral CV, 5-edge non-hub controls, edge audit, and RSA null
│   ├── step3_biophysical_simulations.py    # Multi-SNR Regime A/B simulations and stratified held-out subtype recovery
│   ├── step4_external_validation_adhd.py   # External resting-state Welch CSD CV, nperseg sensitivity, and LOSO transfer
│   └── step5_generate_figures.py           # Generates all 7 publication figures (300 DPI)
├── results/
│   └── spdcm_dmn_results.json              # Complete verified numerical outputs across all analyses
├── Figs/                                   # High-resolution publication figures (Fig1 to Fig7, 300 DPI PNG)
├── run_all.py                              # Master execution script
└── requirements.txt                        # Python package dependencies
```

---

## Installation & Reproducibility

1. **Clone the repository and install dependencies:**
```bash
git clone https://github.com/hamidreza-s-salehi/spdcm-manifold-ranking.git
cd spdcm-manifold-ranking
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Run the analysis and figure-generation pipeline:**
```bash
python run_all.py
```

Running `run_all.py` executes the pipeline with a fixed deterministic seed (`SEED = 42`), saves all numerical outputs to `results/spdcm_dmn_results.json`, and generates `Fig1` through `Fig7` in `Figs/`.

---

## Data Availability

* **Primary Cohort ($N = 58$, Iranian Brain Imaging Database [IBID]):** Resting-state fMRI data were obtained from Group 1 of the Iranian Brain Imaging Database (Batouli et al., 2021) hosted at the National Brain Mapping Laboratory (`https://nbml.ir`) and are governed by an institutional data-use agreement. If `data/model_space.mat` is placed in the `data/` directory, `run_all.py` executes the full primary IBID re-analysis; otherwise, `run_all.py` uses the precomputed IBID summary in `results/spdcm_dmn_results.json` and runs the biophysical simulations and public external cohort validation directly.
* **External Cohort ($N = 40$, NYU / ADHD-200 Repository):** Publicly available via Nilearn (`nilearn.datasets.fetch_adhd(n_subjects=40)`; Abraham et al., 2014). Pre-extracted 6-node DMN resting-state time series are included in `data/external_adhd_timeseries_40.pkl`.
