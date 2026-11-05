"""
Biophysical CSD forward model, BMR, PEB, RFX-BMS, and statistical utilities.

This module provides reusable mathematical, statistical, and plotting routines
for resting-state spectral Dynamic Causal Modeling (spDCM).
"""

__author__ = "Hamidreza Sadeghsalehi, Fatemeh Ahmadi, Arash Zare Sadeghi"

import os
import shutil
import numpy as np
import scipy.signal as signal
from scipy.special import digamma
import matplotlib.pyplot as plt

# Global Seed for Reproducibility
SEED = 42

# Default Mode Network (DMN) Region Specifications (K = 6)
ROI_NAMES = [
    'PCC',                # Node 1: Posterior Cingulate Cortex
    'Frontal_Med_Orb_L',  # Node 2: Left Medial Orbitofrontal Cortex
    'Frontal_Med_Orb_R',  # Node 3: Right Medial Orbitofrontal Cortex
    'Lateral_Parietal',   # Node 4: Left Angular Gyrus / Lateral Parietal
    'Temporal_Mid_L',     # Node 5: Left Middle Temporal Gyrus
    'Temporal_sup_R'      # Node 6: Right Superior Temporal Gyrus
]

ROI_ABBR = ['PCC', 'mOFCL', 'mOFCR', 'LP', 'MTGL', 'STGR']
K_NODES = 6
K_EXT = [30, 5, 10, 5, 10, 14, 14]


def to_serializable(val):
    """Recursively convert numpy types to native Python types for JSON export."""
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    elif isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    elif isinstance(val, (np.ndarray, list, tuple)):
        return [to_serializable(x) for x in val]
    elif isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    elif isinstance(val, np.bool_):
        return bool(val)
    elif isinstance(val, complex):
        return {'real': float(val.real), 'imag': float(val.imag)}
    return val


def safe_savefig(fig, filename, dpi=300, bbox_inches='tight'):
    """
    Save figure safely to disk and mirror it to manuscript figures directory if present.
    """
    os.makedirs('Figs', exist_ok=True)
    
    root_path = os.path.join('Figs', os.path.basename(filename))
    fig.savefig(root_path, dpi=dpi, bbox_inches=bbox_inches)
    
    if os.path.exists('manuscript'):
        manuscript_figs = os.path.join('manuscript', 'Figs')
        os.makedirs(manuscript_figs, exist_ok=True)
        manuscript_path = os.path.join(manuscript_figs, os.path.basename(filename))
        try:
            shutil.copy2(root_path, manuscript_path)
        except Exception:
            pass


def holm_bonferroni(p_vals):
    """Step-down Holm-Bonferroni correction for multiple hypothesis testing."""
    p_vals = np.asarray(p_vals, dtype=float)
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]
    
    adj_p = np.zeros(n)
    cum_max = 0.0
    for i, p in enumerate(sorted_p):
        val = p * (n - i)
        cum_max = max(cum_max, val)
        adj_p[i] = min(1.0, cum_max)
        
    rev_indices = np.zeros(n, dtype=int)
    rev_indices[sorted_indices] = np.arange(n)
    return adj_p[rev_indices]


def run_ols(X, y, feature_names=None):
    """Ordinary Least Squares regression with t-statistics and p-values."""
    from scipy.stats import t
    n, p = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_pred = X @ beta
    residuals = y - y_pred
    dof = n - p
    sigma2 = np.sum(residuals**2) / dof
    var_beta = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(var_beta))
    t_stat = beta / (se + 1e-12)
    p_val = 2.0 * (1.0 - t.cdf(np.abs(t_stat), dof))
    r2 = 1.0 - (np.sum(residuals**2) / (np.sum((y - np.mean(y))**2) + 1e-12))
    
    return {
        'beta': beta.tolist(),
        'se': se.tolist(),
        't_stat': t_stat.tolist(),
        'p_val': p_val.tolist(),
        'r2': float(r2),
        'dof': int(dof),
        'feature_names': feature_names if feature_names is not None else [f"x{i}" for i in range(p)]
    }


def get_model_masks(data_path='data/model_space.mat'):
    """
    Returns boolean masks (6 x 6) for the seven candidate DMN architectures.
    Loads exact binary masks directly from data/model_space.mat if present,
    with an assertion that k_ext == [30, 5, 10, 5, 10, 14, 14].
    """
    if os.path.exists(data_path):
        import scipy.io as sio
        mat = sio.loadmat(data_path, simplify_cells=True)
        masks = [((np.abs(mat['subj'][0]['sess']['model'][m]['Ep']['A']) > 1e-5) & ~np.eye(6, dtype=bool)) for m in range(7)]
    else:
        # Exact verified binary masks matching model_space.mat
        m1 = ~np.eye(6, dtype=bool)
        m2 = np.zeros((6, 6), dtype=bool)
        m2[[0, 1, 2, 4, 5], 3] = True
        m3 = np.zeros((6, 6), dtype=bool)
        m3[[0, 1, 2, 4, 5], 3] = True
        m3[3, [0, 1, 2, 4, 5]] = True
        m4 = np.zeros((6, 6), dtype=bool)
        m4[[1, 2, 3, 4, 5], 0] = True
        m5 = np.zeros((6, 6), dtype=bool)
        m5[[1, 2, 3, 4, 5], 0] = True
        m5[0, [1, 2, 3, 4, 5]] = True
        m6 = np.zeros((6, 6), dtype=bool)
        m6_rows = [0, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 5, 5, 5]
        m6_cols = [3, 0, 2, 3, 0, 1, 3, 0, 0, 3, 5, 0, 3, 4]
        m6[m6_rows, m6_cols] = True
        m7 = np.zeros((6, 6), dtype=bool)
        m7_rows = [0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 5]
        m7_cols = [1, 2, 3, 4, 5, 0, 3, 0, 3, 0, 1, 2, 0, 0]
        m7[m7_rows, m7_cols] = True
        masks = [m1, m2, m3, m4, m5, m6, m7]

    k_counts = [int(np.sum(m)) for m in masks]
    assert k_counts == [30, 5, 10, 5, 10, 14, 14], f"Expected [30, 5, 10, 5, 10, 14, 14], got {k_counts}"
    return masks


def get_5edge_control_masks():
    """
    Returns boolean masks (6 x 6) for canonical hubs and non-hub star controls (all k_ext = 5).
    """
    masks_5edge = {
        'Model_4_PCC': np.zeros((6, 6), dtype=bool),
        'Model_2_LP': np.zeros((6, 6), dtype=bool),
        'Control_mOFCL': np.zeros((6, 6), dtype=bool),
        'Control_mOFCR': np.zeros((6, 6), dtype=bool),
        'Control_MTGL': np.zeros((6, 6), dtype=bool),
        'Control_STGR': np.zeros((6, 6), dtype=bool)
    }
    masks_5edge['Model_4_PCC'][[1, 2, 3, 4, 5], 0] = True
    masks_5edge['Model_2_LP'][[0, 1, 2, 4, 5], 3] = True
    masks_5edge['Control_mOFCL'][[0, 2, 3, 4, 5], 1] = True
    masks_5edge['Control_mOFCR'][[0, 1, 3, 4, 5], 2] = True
    masks_5edge['Control_MTGL'][[0, 1, 2, 3, 5], 4] = True
    masks_5edge['Control_STGR'][[0, 1, 2, 3, 4], 5] = True
    return masks_5edge


def biophysical_csd(A, freqs, tau_h=0.65, alpha_v=1.0, beta_v=1.0, sigma_obs=0.05):
    """
    Biophysical resting-state cross-spectral density forward model.
    Computes cross-spectral density Gy(w; A) using canonical SPM12 transfer functions.
    Matches verified implementation in run_dmn_q1_pipeline.py and run_major_revision_fixes.py.
    """
    K = A.shape[0]
    N_f = len(freqs)
    if np.allclose(np.diagonal(A), 0):
        J_sys = A - 0.5 * np.eye(K)
    else:
        J_sys = A
    Gy = np.zeros((N_f, K, K), dtype=complex)
    for f_idx, w in enumerate(freqs):
        H = 1.0 / (1.0 + 1j * w * tau_h)**2
        T = H * np.linalg.inv(1j * 2 * np.pi * w * np.eye(K) - J_sys)
        gv = alpha_v * (w ** (-beta_v)) * np.eye(K)
        ge = (sigma_obs**2) * np.eye(K)
        Gy[f_idx] = T @ gv @ T.conj().T + ge
    return Gy


def compute_exact_36d_bmr(qE, qC, pC_diag=1.0/64.0):
    """
    Computes exact 36D Savage-Dickey Bayesian Model Reduction (BMR) for models 1..7.
    Exact equation:
    Delta F = +0.5 * ln|pC_Z inv(qC_Z)| - 0.5 * qE_Z^T inv(qC_Z) qE_Z.
    """
    masks = get_model_masks()
    offdiag_all = ~np.eye(6, dtype=bool).flatten()
    dim = 36
    pC_mat = pC_diag * np.eye(dim)
    
    dF = np.zeros(7)
    occam = np.zeros(7)
    acc = np.zeros(7)
    
    for m in range(7):
        z_idx = np.where(~masks[m].flatten() & offdiag_all)[0]
        if len(z_idx) == 0:
            dF[m] = 0.0
            occam[m] = 0.0
            acc[m] = 0.0
        else:
            qC_z = qC[np.ix_(z_idx, z_idx)]
            qE_z = qE[z_idx]
            inv_qC_z = np.linalg.inv(qC_z)
            
            _, logdet_p = np.linalg.slogdet(pC_mat[np.ix_(z_idx, z_idx)])
            _, logdet_q = np.linalg.slogdet(qC_z)
            
            occ_val = 0.5 * (logdet_p - logdet_q)
            acc_val = -0.5 * (qE_z.T @ inv_qC_z @ qE_z)
            
            occam[m] = occ_val
            acc[m] = acc_val
            dF[m] = occ_val + acc_val
            
    return dF, occam, acc


def compute_second_level_peb(X_cohort, qC_within):
    """
    Implements SPM12 hierarchical second-level Parametric Empirical Bayes (spm_dcm_peb_bmc.m):
    qP^(2) = pP^(2) + sum_s (qC_s + Sigma_B)^(-1)
    """
    N, dim = X_cohort.shape
    var_between = np.var(X_cohort, axis=0, ddof=1)
    Sigma_B = np.diag(var_between) + (1.0 / 64.0) * np.eye(dim)
    
    pP_2 = 64.0 * np.eye(dim)
    V_s = qC_within + Sigma_B
    inv_V_s = np.linalg.inv(V_s)
    
    qP_2 = pP_2 + N * inv_V_s
    qC_2 = np.linalg.inv(qP_2)
    qE_2 = qC_2 @ (inv_V_s @ np.sum(X_cohort, axis=0))
    
    # Second-level analytical BMR across Models 1..7
    dF_peb, occam_peb, acc_peb = compute_exact_36d_bmr(qE_2, qC_2)
    post_prob = np.exp(dF_peb - np.max(dF_peb))
    post_prob /= np.sum(post_prob)
    
    return {
        'qE_second_level': qE_2.tolist(),
        'dF_PEB_nats': dF_peb.tolist(),
        'Occam_PEB_nats': occam_peb.tolist(),
        'AccLoss_PEB_nats': acc_peb.tolist(),
        'P_PEB_model_probability': post_prob.tolist(),
        'winning_model': int(np.argmax(post_prob) + 1),
        'P_M1_probability': float(post_prob[0])
    }


def run_rfx_bms(F, alpha0=None):
    """
    Random-Effects Bayesian Model Selection (RFX-BMS; Rigoux et al., 2014; Penny et al., 2010).
    Treats individual model attribution as a multinomial latent variable governed by Dirichlet distribution.
    Supports arbitrary prior Dirichlet parameters alpha0 (e.g. uniform model priors or balanced family priors).
    """
    N, M = F.shape
    if alpha0 is None:
        alpha0 = np.ones(M)
    else:
        alpha0 = np.asarray(alpha0, dtype=float)
    alpha = alpha0.copy()
    
    for _ in range(100):
        u = np.zeros((N, M))
        for s in range(N):
            log_u = F[s] + digamma(alpha) - digamma(np.sum(alpha))
            log_u -= np.max(log_u)
            u[s] = np.exp(log_u) / np.sum(np.exp(log_u))
        alpha = alpha0 + np.sum(u, axis=0)
        
    exp_r = alpha / np.sum(alpha)
    
    # Exceedance probabilities via Dirichlet Monte Carlo sampling
    rng = np.random.default_rng(SEED)
    samples = rng.dirichlet(alpha, size=10000)
    xp = np.mean(samples == np.max(samples, axis=1, keepdims=True), axis=0)
    
    return alpha, exp_r, xp
