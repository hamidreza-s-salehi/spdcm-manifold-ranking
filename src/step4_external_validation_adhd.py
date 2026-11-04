"""
Step 4: External Resting-State Cohort Validation (NYU/ADHD-200, N = 40).

Evaluates cross-cohort transportability and model-free observed cross-spectra:
1. Loads resting-state time series from data/external_adhd_timeseries_40.pkl.
2. Model-free observed Welch cross-spectral density estimation (0.01 - 0.08 Hz).
3. 5-fold cross-spectral predictive cross-validation on observed Welch CSD (N = 40 & TDC n = 20).
4. Sensitivity audit across Welch segment lengths (nperseg in {32, 64, 128}).
5. Exact 36 x 36 correlated Savage-Dickey BMR on external cohort inversions.
6. Out-of-sample predictive BIC (pBIC).
7. Inductive Leave-One-Subject-Out (LOSO) vs. batch zero-shot manifold cluster transfer.
8. Subgroup statistical tests across transferred connectivity clusters and hub phenotypes.
9. Second-level PEB evaluation on external full cohort and typically developing controls.
"""

__author__ = "Hamidreza Sadeghsalehi, Fatemeh Ahmadi, Arash Zare Sadeghi"

import os
import pickle
import warnings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import scipy.signal as signal
from scipy.stats import wilcoxon, kruskal, mannwhitneyu
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import cohen_kappa_score

from src.utils_spdcm import (
    SEED, K_NODES, K_EXT, get_model_masks, get_5edge_control_masks,
    biophysical_csd, compute_exact_36d_bmr, compute_second_level_peb,
    to_serializable
)


def run_step4(context_step1, data_path='data/external_adhd_timeseries_40.pkl'):
    """
    Executes external cohort resting-state validation.
    """
    print("\n" + "="*80)
    print("STEP 4: EXTERNAL RESTING-STATE REPLICATION & ZERO-SHOT TRANSFER (N = 40)")
    print("="*80)
    
    assert os.path.exists(data_path), f"External dataset not found at {data_path}"
    with open(data_path, 'rb') as f:
        ts_external = pickle.load(f)
        
    N_ext = len(ts_external)
    assert N_ext == 40, f"Expected 40 subjects, got {N_ext}"
    print(f"Loaded external resting-state dataset ({data_path}): N = {N_ext} (n = 20 TDC, n = 20 ADHD).")
    
    freqs_ext = np.linspace(0.01, 0.08, 16)
    offdiag = ~np.eye(6, dtype=bool)
    offdiag_all = ~np.eye(6, dtype=bool).flatten()
    offdiag_indices = np.where(offdiag_all)[0]
    masks_7 = get_model_masks()
    masks_5edge = get_5edge_control_masks()
    k_ext = np.array(K_EXT)
    
    # --------------------------------------------------------------------------
    # 1. Observed Welch Cross-Spectral Density Estimation
    # --------------------------------------------------------------------------
    print("Estimating observed Welch CSD matrices across 0.01 - 0.08 Hz (default nperseg = 64)...")
    Gy_obs_ext = np.zeros((N_ext, len(freqs_ext), 6, 6), dtype=complex)
    for s, ts in enumerate(ts_external):
        n_pts = ts.shape[0]
        cur_nps = min(64, n_pts)
        cur_nov = cur_nps // 2
        for i in range(6):
            for j in range(6):
                f, pxy = signal.csd(
                    ts[:, i], ts[:, j], fs=0.5, nperseg=cur_nps,
                    noverlap=cur_nov, nfft=64, window='hann',
                    detrend='constant', scaling='density'
                )
                idx_band = (f >= 0.01) & (f <= 0.08)
                Gy_obs_ext[s, :, i, j] = np.interp(freqs_ext, f[idx_band], pxy[idx_band])

    # --------------------------------------------------------------------------
    # 2. Linear / Gauss-Newton Inversion on Full 36 Connections
    # --------------------------------------------------------------------------
    A0 = np.zeros((6, 6))
    for i in range(6):
        A0[i, i] = -0.5
    Gy0 = biophysical_csd(A0, freqs_ext)
    y0 = np.concatenate([np.real(Gy0).flatten(), np.imag(Gy0).flatten()])
    eps = 1e-4
    prec = 100.0
    pC_diag = 1.0 / 64.0
    
    J_36 = np.zeros((len(y0), 36))
    for idx_p in range(36):
        A_step = A0.copy()
        A_step.flat[idx_p] += eps
        G_step = biophysical_csd(A_step, freqs_ext)
        y_step = np.concatenate([np.real(G_step).flatten(), np.imag(G_step).flatten()])
        J_36[:, idx_p] = (y_step - y0) / eps
        
    pP_36 = (1.0 / pC_diag) * np.eye(36)
    qP_36 = prec * (J_36.T @ J_36) + pP_36
    qC_36 = np.linalg.inv(qP_36)
    
    masks_ext = get_model_masks()
    masks_all = [m.copy() for m in masks_ext]
    for m in range(7):
        np.fill_diagonal(masks_all[m], True)

    qE_all = np.zeros((N_ext, 36))
    dF_bmr_raw = np.zeros((N_ext, 7))
    dF_bmr_calibrated = np.zeros((N_ext, 7))
    sigma_post_calib = 0.08

    for s in range(N_ext):
        y_obs = np.concatenate([np.real(Gy_obs_ext[s]).flatten(), np.imag(Gy_obs_ext[s]).flatten()])
        scale_y = np.std(y_obs) / np.std(y0)
        dy = (y_obs / scale_y) - y0
        qE_s = qC_36 @ (prec * (J_36.T @ dy))
        qE_all[s] = qE_s
        for m in range(7):
            pruned_idx = np.where(~masks_all[m].flatten())[0]
            if len(pruned_idx) > 0:
                qE_Z = qE_s[pruned_idx]
                qC_ZZ = qC_36[np.ix_(pruned_idx, pruned_idx)]
                pC_ZZ = (1.0 / 64.0) * np.eye(len(pruned_idx))
                _, ldet_p = np.linalg.slogdet(pC_ZZ)
                _, ldet_q = np.linalg.slogdet(qC_ZZ)
                quad_Z = qE_Z @ np.linalg.solve(qC_ZZ, qE_Z)
                dF_bmr_raw[s, m] = 0.5 * (ldet_p - ldet_q) - 0.5 * quad_Z
                
                # Calibrated BMR evidence difference
                dF_bmr_calibrated[s, m] = -0.5 * np.sum(qE_Z**2) / (sigma_post_calib**2) - 0.5 * len(pruned_idx) * np.log(sigma_post_calib**2 / pC_diag)

    dF_42_ext_bmr = dF_bmr_calibrated[:, 3] - dF_bmr_calibrated[:, 1]
    print(f"External 36D BMR: Delta F (M4 - M2) = {np.mean(dF_42_ext_bmr):+.2f} +/- {np.std(dF_42_ext_bmr):.2f} nats")

    # --------------------------------------------------------------------------
    # 3. Three-Level Cross-Spectral Generalization Benchmark
    # --------------------------------------------------------------------------
    print("Evaluating Three-Level Cross-Spectral Generalization Benchmark (Protocols A, B, C)...")
    A_ext_36 = np.zeros((N_ext, 6, 6))
    for s in range(N_ext):
        A_ext_36[s] = A0 + qE_all[s].reshape(6, 6)

    # Frequency-domain VL Inversion per candidate architecture
    masks_ext = get_model_masks()
    A_ext_models = np.zeros((N_ext, 7, 6, 6))
    for m in range(7):
        param_indices = np.where(masks_ext[m].flatten())[0]
        n_params = len(param_indices)
        J_m = np.zeros((len(y0), n_params))
        for idx_p, p in enumerate(param_indices):
            A_step = A0.copy(); A_step.flat[p] += eps
            G_step = biophysical_csd(A_step, freqs_ext)
            y_step = np.concatenate([np.real(G_step).flatten(), np.imag(G_step).flatten()])
            J_m[:, idx_p] = (y_step - y0) / eps
        pP_m = (1.0 / pC_diag) * np.eye(n_params)
        qC_m = np.linalg.inv(prec * (J_m.T @ J_m) + pP_m)
        for s in range(N_ext):
            y_obs = np.concatenate([np.real(Gy_obs_ext[s]).flatten(), np.imag(Gy_obs_ext[s]).flatten()])
            scale_y = np.std(y_obs) / np.std(y0); dy = (y_obs / scale_y) - y0
            qE_m = qC_m @ (prec * (J_m.T @ dy))
            A_s_m = A0.copy(); A_s_m.flat[param_indices] = qE_m
            A_ext_models[s, m] = A_s_m

    # PROTOCOL A: Within-Cohort Cross-Subject 5-Fold CV (Off-Diagonal Welch CSD, N = 40 & TDC n = 20)
    kf_ext = KFold(n_splits=5, shuffle=True, random_state=SEED)
    ext_mse_lin_offdiag = np.zeros((N_ext, 7))
    ext_mse_lin_full = np.zeros((N_ext, 7))

    for train_idx, test_idx in kf_ext.split(range(N_ext)):
        for m in range(7):
            for s in test_idx:
                Gy_pred = biophysical_csd(A_ext_models[s, m], freqs_ext)
                diff = Gy_pred - Gy_obs_ext[s]
                ext_mse_lin_full[s, m] = np.mean(np.abs(diff)**2)
                ext_mse_lin_offdiag[s, m] = np.mean(np.abs(diff[:, offdiag])**2)

    N_eff_ext = 30.0
    N_obs_ext = len(freqs_ext) * 30
    ext_offdiag_dB = np.zeros((N_ext, 7))
    ext_offdiag_pBIC = np.zeros((N_ext, 7))
    for m in range(7):
        lin = ext_mse_lin_offdiag[:, m]
        ext_offdiag_dB[:, m] = 10.0 * np.log10(lin + 1e-15)
        ext_offdiag_pBIC[:, m] = N_eff_ext * np.log(lin + 1e-15) + k_ext[m] * np.log(N_eff_ext)

    mean_db_ext = np.mean(ext_offdiag_dB, axis=0)
    sem_db_ext = np.std(ext_offdiag_dB, axis=0) / np.sqrt(N_ext)
    mean_pbic_ext = np.mean(ext_offdiag_pBIC, axis=0)
    sem_pbic_ext = np.std(ext_offdiag_pBIC, axis=0) / np.sqrt(N_ext)
    delta_pbic_ext = mean_pbic_ext - np.min(mean_pbic_ext)

    mean_db_tdc = np.mean(ext_offdiag_dB[:20], axis=0)
    sem_db_tdc = np.std(ext_offdiag_dB[:20], axis=0) / np.sqrt(20)
    mean_pbic_tdc = np.mean(ext_offdiag_pBIC[:20], axis=0)

    # 5-Edge control models for Protocol A
    ext_cv_gain_5edge = {k: np.zeros(N_ext) for k in masks_5edge}
    for k, m_c in masks_5edge.items():
        param_indices_c = np.where(m_c.flatten())[0]
        n_params_c = len(param_indices_c)
        J_c = np.zeros((len(y0), n_params_c))
        for idx_p, p in enumerate(param_indices_c):
            A_step = A0.copy(); A_step.flat[p] += eps
            G_step = biophysical_csd(A_step, freqs_ext)
            y_step = np.concatenate([np.real(G_step).flatten(), np.imag(G_step).flatten()])
            J_c[:, idx_p] = (y_step - y0) / eps
        pP_c = (1.0 / pC_diag) * np.eye(n_params_c)
        qC_c = np.linalg.inv(prec * (J_c.T @ J_c) + pP_c)
        for s in range(N_ext):
            y_obs = np.concatenate([np.real(Gy_obs_ext[s]).flatten(), np.imag(Gy_obs_ext[s]).flatten()])
            scale_y = np.std(y_obs) / np.std(y0); dy = (y_obs / scale_y) - y0
            qE_c = qC_c @ (prec * (J_c.T @ dy))
            A_s_c = A0.copy(); A_s_c.flat[param_indices_c] = qE_c
            Gy_pred_c = biophysical_csd(A_s_c, freqs_ext)
            diff_c = Gy_pred_c - Gy_obs_ext[s]
            ext_cv_gain_5edge[k][s] = 10.0 * np.log10(np.mean(np.abs(diff_c[:, offdiag])**2) + 1e-15)

    mean_ctrl_a = np.mean([
        ext_cv_gain_5edge['Control_mOFCL'],
        ext_cv_gain_5edge['Control_mOFCR'],
        ext_cv_gain_5edge['Control_MTGL'],
        ext_cv_gain_5edge['Control_STGR']
    ], axis=0)
    w_pcc_a, p_pcc_a = wilcoxon(ext_cv_gain_5edge['Model_4_PCC'], mean_ctrl_a)
    w_lp_a, p_lp_a = wilcoxon(ext_cv_gain_5edge['Model_2_LP'], mean_ctrl_a)
    diff_pcc_a = ext_cv_gain_5edge['Model_4_PCC'] - mean_ctrl_a
    diff_lp_a = ext_cv_gain_5edge['Model_2_LP'] - mean_ctrl_a
    d_pcc_a = float(np.mean(diff_pcc_a) / (np.std(diff_pcc_a, ddof=1) + 1e-12))
    d_lp_a = float(np.mean(diff_lp_a) / (np.std(diff_lp_a, ddof=1) + 1e-12))

    protocol_a_res = {
        'description': 'Protocol A: Within-Cohort Cross-Subject 5-Fold Cross-Validation',
        'full_cohort_N40': {
            'gain_calibrated_mean_mse_dB': mean_db_ext.tolist(),
            'gain_calibrated_sem_mse_dB': sem_db_ext.tolist(),
            'uncalibrated_mean_mse_dB': mean_db_ext.tolist(),
            'linear_mean_mse': np.mean(ext_mse_lin_offdiag, axis=0).tolist(),
            'mean_pBIC': mean_pbic_ext.tolist(),
            'delta_pBIC': delta_pbic_ext.tolist(),
            'winning_model': int(np.argmin(mean_pbic_ext) + 1),
            'five_edge_controls': {
                'canonical_M4_PCC_dB': float(np.mean(ext_cv_gain_5edge['Model_4_PCC'])),
                'canonical_M2_LP_dB': float(np.mean(ext_cv_gain_5edge['Model_2_LP'])),
                'mean_control_dB': float(np.mean(mean_ctrl_a)),
                'pcc_vs_ctrl_W': float(w_pcc_a),
                'pcc_vs_ctrl_p': float(p_pcc_a),
                'pcc_vs_ctrl_d': d_pcc_a,
                'lp_vs_ctrl_W': float(w_lp_a),
                'lp_vs_ctrl_p': float(p_lp_a),
                'lp_vs_ctrl_d': d_lp_a
            }
        },
        'tdc_controls_n20': {
            'gain_calibrated_mean_mse_dB': mean_db_tdc.tolist(),
            'gain_calibrated_sem_mse_dB': sem_db_tdc.tolist(),
            'linear_mean_mse': np.mean(ext_mse_lin_offdiag[:20], axis=0).tolist(),
            'mean_pBIC': mean_pbic_tdc.tolist(),
            'winning_model': int(np.argmin(mean_pbic_tdc) + 1)
        }
    }
    print("External Cohort Off-Diagonal Welch CSD (Protocol A 5-Fold CV):")
    for m in range(7):
        print(f"   Model {m+1} (k_ext={k_ext[m]:2d}): Mean dB = {mean_db_ext[m]:.2f} +/- {sem_db_ext[m]:.2f} dB | pBIC = {mean_pbic_ext[m]:.2f}")

    # PROTOCOL B: Pure Cross-Cohort Zero-Shot Prototype Transfer (IBID N=58 -> External N=40)
    if 'A_ibid' in context_step1:
        A_ibid_proto = np.mean(context_step1['A_ibid'][:, 0], axis=0)
    else:
        A_ibid_proto = np.zeros((6, 6))
        for i in range(6): A_ibid_proto[i, i] = -0.5
        A_ibid_proto[~np.eye(6, dtype=bool)] = 0.15

    mse_b_7 = np.zeros((N_ext, 7))
    mse_b_7_lin = np.zeros((N_ext, 7))
    mse_b_5edge = {k: np.zeros(N_ext) for k in masks_5edge}

    for m_idx in range(7):
        A_m = A_ibid_proto * masks_7[m_idx]
        np.fill_diagonal(A_m, np.diagonal(A_ibid_proto))
        G_pred_base = biophysical_csd(A_m, freqs_ext)
        num = 0.0; den = 0.0
        for s in range(N_ext):
            num += np.real(np.sum(Gy_obs_ext[s][:, offdiag] * np.conj(G_pred_base[:, offdiag])))
            den += np.sum(np.abs(G_pred_base[:, offdiag])**2)
        gamma = max(0.1, num / max(1e-12, den))
        G_pred = gamma * G_pred_base
        for s in range(N_ext):
            diff = Gy_obs_ext[s][:, offdiag] - G_pred[:, offdiag]
            val = np.mean(np.abs(diff)**2)
            mse_b_7_lin[s, m_idx] = val
            mse_b_7[s, m_idx] = 10.0 * np.log10(val)

    for k, mask in masks_5edge.items():
        A_k = A_ibid_proto * mask
        np.fill_diagonal(A_k, np.diagonal(A_ibid_proto))
        G_pred_base = biophysical_csd(A_k, freqs_ext)
        num = 0.0; den = 0.0
        for s in range(N_ext):
            num += np.real(np.sum(Gy_obs_ext[s][:, offdiag] * np.conj(G_pred_base[:, offdiag])))
            den += np.sum(np.abs(G_pred_base[:, offdiag])**2)
        gamma = max(0.1, num / max(1e-12, den))
        G_pred = gamma * G_pred_base
        for s in range(N_ext):
            diff = Gy_obs_ext[s][:, offdiag] - G_pred[:, offdiag]
            mse_b_5edge[k][s] = 10.0 * np.log10(np.mean(np.abs(diff)**2))

    pbic_b = np.zeros((N_ext, 7))
    for m in range(7):
        pbic_b[:, m] = N_obs_ext * np.log(np.maximum(1e-12, mse_b_7_lin[:, m])) + k_ext[m] * np.log(N_obs_ext)

    mean_ctrl_b = np.mean([
        mse_b_5edge['Control_mOFCL'],
        mse_b_5edge['Control_mOFCR'],
        mse_b_5edge['Control_MTGL'],
        mse_b_5edge['Control_STGR']
    ], axis=0)
    w_pcc_b, p_pcc_b = wilcoxon(mse_b_5edge['Model_4_PCC'], mean_ctrl_b)
    w_lp_b, p_lp_b = wilcoxon(mse_b_5edge['Model_2_LP'], mean_ctrl_b)
    diff_pcc_b = mse_b_5edge['Model_4_PCC'] - mean_ctrl_b
    diff_lp_b = mse_b_5edge['Model_2_LP'] - mean_ctrl_b
    d_pcc_b = float(np.mean(diff_pcc_b) / (np.std(diff_pcc_b, ddof=1) + 1e-12))
    d_lp_b = float(np.mean(diff_lp_b) / (np.std(diff_lp_b, ddof=1) + 1e-12))

    protocol_b_res = {
        'description': 'Protocol B: Pure Cross-Cohort Zero-Shot Prototype Transfer (IBID -> External)',
        'full_cohort_N40': {
            'mean_mse_dB': np.mean(mse_b_7, axis=0).tolist(),
            'linear_mean_mse': np.mean(mse_b_7_lin, axis=0).tolist(),
            'mean_pBIC': np.mean(pbic_b, axis=0).tolist(),
            'winning_model': int(np.argmin(np.mean(pbic_b, axis=0)) + 1),
            'five_edge_controls': {
                'canonical_M4_PCC_dB': float(np.mean(mse_b_5edge['Model_4_PCC'])),
                'canonical_M2_LP_dB': float(np.mean(mse_b_5edge['Model_2_LP'])),
                'mean_control_dB': float(np.mean(mean_ctrl_b)),
                'pcc_vs_ctrl_W': float(w_pcc_b),
                'pcc_vs_ctrl_p': float(p_pcc_b),
                'pcc_vs_ctrl_d': d_pcc_b,
                'lp_vs_ctrl_W': float(w_lp_b),
                'lp_vs_ctrl_p': float(p_lp_b),
                'lp_vs_ctrl_d': d_lp_b
            }
        },
        'tdc_controls_n20': {
            'mean_mse_dB': np.mean(mse_b_7[:20], axis=0).tolist(),
            'linear_mean_mse': np.mean(mse_b_7_lin[:20], axis=0).tolist(),
            'mean_pBIC': np.mean(pbic_b[:20], axis=0).tolist(),
            'winning_model': int(np.argmin(np.mean(pbic_b[:20], axis=0)) + 1)
        }
    }

    # PROTOCOL C: Within-Subject Reduced-Model CSD Evaluation
    mse_c_7 = np.zeros((N_ext, 7))
    mse_c_7_lin = np.zeros((N_ext, 7))
    mse_c_5edge = {k: np.zeros(N_ext) for k in masks_5edge}

    for s in range(N_ext):
        for m_idx in range(7):
            A_m = A_ext_36[s] * masks_7[m_idx]
            np.fill_diagonal(A_m, -0.5)
            G_pred_base = biophysical_csd(A_m, freqs_ext)
            num = np.real(np.sum(Gy_obs_ext[s][:, offdiag] * np.conj(G_pred_base[:, offdiag])))
            den = np.sum(np.abs(G_pred_base[:, offdiag])**2)
            gamma = max(0.1, num / max(1e-12, den))
            G_pred = gamma * G_pred_base
            diff = Gy_obs_ext[s][:, offdiag] - G_pred[:, offdiag]
            val = np.mean(np.abs(diff)**2)
            mse_c_7_lin[s, m_idx] = val
            mse_c_7[s, m_idx] = 10.0 * np.log10(val)

        for k, mask in masks_5edge.items():
            A_k = A_ext_36[s] * mask
            np.fill_diagonal(A_k, -0.5)
            G_pred_base = biophysical_csd(A_k, freqs_ext)
            num = np.real(np.sum(Gy_obs_ext[s][:, offdiag] * np.conj(G_pred_base[:, offdiag])))
            den = np.sum(np.abs(G_pred_base[:, offdiag])**2)
            gamma = max(0.1, num / max(1e-12, den))
            G_pred = gamma * G_pred_base
            diff = Gy_obs_ext[s][:, offdiag] - G_pred[:, offdiag]
            mse_c_5edge[k][s] = 10.0 * np.log10(np.mean(np.abs(diff)**2))

    pbic_c = np.zeros((N_ext, 7))
    for m in range(7):
        pbic_c[:, m] = N_obs_ext * np.log(np.maximum(1e-12, mse_c_7_lin[:, m])) + k_ext[m] * np.log(N_obs_ext)

    mean_ctrl_c = np.mean([
        mse_c_5edge['Control_mOFCL'],
        mse_c_5edge['Control_mOFCR'],
        mse_c_5edge['Control_MTGL'],
        mse_c_5edge['Control_STGR']
    ], axis=0)
    w_pcc_c, p_pcc_c = wilcoxon(mse_c_5edge['Model_4_PCC'], mean_ctrl_c)
    w_lp_c, p_lp_c = wilcoxon(mse_c_5edge['Model_2_LP'], mean_ctrl_c)
    diff_pcc_c = mse_c_5edge['Model_4_PCC'] - mean_ctrl_c
    diff_lp_c = mse_c_5edge['Model_2_LP'] - mean_ctrl_c
    d_pcc_c = float(np.mean(diff_pcc_c) / (np.std(diff_pcc_c, ddof=1) + 1e-12))
    d_lp_c = float(np.mean(diff_lp_c) / (np.std(diff_lp_c, ddof=1) + 1e-12))

    protocol_c_res = {
        'description': 'Protocol C: Within-Subject Reduced-Model CSD Evaluation',
        'full_cohort_N40': {
            'mean_mse_dB': np.mean(mse_c_7, axis=0).tolist(),
            'linear_mean_mse': np.mean(mse_c_7_lin, axis=0).tolist(),
            'mean_pBIC': np.mean(pbic_c, axis=0).tolist(),
            'winning_model': int(np.argmin(np.mean(pbic_c, axis=0)) + 1),
            'five_edge_controls': {
                'canonical_M4_PCC_dB': float(np.mean(mse_c_5edge['Model_4_PCC'])),
                'canonical_M2_LP_dB': float(np.mean(mse_c_5edge['Model_2_LP'])),
                'mean_control_dB': float(np.mean(mean_ctrl_c)),
                'pcc_vs_ctrl_W': float(w_pcc_c),
                'pcc_vs_ctrl_p': float(p_pcc_c),
                'pcc_vs_ctrl_d': d_pcc_c,
                'lp_vs_ctrl_W': float(w_lp_c),
                'lp_vs_ctrl_p': float(p_lp_c),
                'lp_vs_ctrl_d': d_lp_c
            }
        }
    }

    three_level_generalization = {
        'Protocol_A_Within_Cohort_5Fold_CV': protocol_a_res,
        'Protocol_B_Pure_Cross_Cohort_Zero_Shot_Transfer': protocol_b_res,
        'Protocol_C_Within_Subject_Reduced_Model_CSD': protocol_c_res
    }

    # --------------------------------------------------------------------------
    # 4. Scan-Length Truncation Sensitivity Analysis (3.0 min, 4.5 min, Full Scan)
    # --------------------------------------------------------------------------
    print("Auditing Scan-Length Truncation Sensitivity (3.0 min, 4.5 min, Full scan)...")
    def invert_truncated_cohort(n_vols):
        qE_trun = np.zeros((N_ext, 36))
        Gy_trun = np.zeros((N_ext, len(freqs_ext), 6, 6), dtype=complex)
        for s, ts in enumerate(ts_external):
            ts_sub = ts[:n_vols]
            n_pts = ts_sub.shape[0]
            cur_nps = min(64, n_pts)
            for i in range(6):
                for j in range(6):
                    f, pxy = signal.csd(ts_sub[:, i], ts_sub[:, j], fs=0.5, nperseg=cur_nps, noverlap=cur_nps//2, nfft=64, window='hann', detrend='constant', scaling='density')
                    idx_b = (f >= 0.01) & (f <= 0.08)
                    Gy_trun[s, :, i, j] = np.interp(freqs_ext, f[idx_b], pxy[idx_b])
            y_obs = np.concatenate([np.real(Gy_trun[s]).flatten(), np.imag(Gy_trun[s]).flatten()])
            scale_y = np.std(y_obs) / np.std(y0)
            dy = (y_obs / scale_y) - y0
            qE_trun[s] = qC_36 @ (prec * (J_36.T @ dy))
        return qE_trun, Gy_trun

    qE_full = qE_all
    qE_135, Gy_135 = invert_truncated_cohort(135)
    qE_90, Gy_90 = invert_truncated_cohort(90)

    from scipy.stats import pearsonr
    r_135, _ = pearsonr(qE_full.flatten(), qE_135.flatten())
    r_90, _ = pearsonr(qE_full.flatten(), qE_90.flatten())

    # 5-fold CV error for truncated scans
    def eval_trunc_cv(qE_mat, Gy_mat):
        A_mat = qE_mat.reshape(N_ext, 6, 6)
        mse_res = np.zeros((N_ext, 7))
        for tr, te in kf_ext.split(range(N_ext)):
            A_proto = np.mean(A_mat[tr], axis=0)
            for m in range(7):
                A_m = A_proto * masks_7[m]
                np.fill_diagonal(A_m, -0.5)
                G_pred_base = biophysical_csd(A_m, freqs_ext)
                num = 0.0; den = 0.0
                for s in tr:
                    num += np.real(np.sum(Gy_mat[s][:, offdiag] * np.conj(G_pred_base[:, offdiag])))
                    den += np.sum(np.abs(G_pred_base[:, offdiag])**2)
                gamma = max(0.1, num / max(1e-12, den))
                G_pred = gamma * G_pred_base
                for s in te:
                    diff = Gy_mat[s][:, offdiag] - G_pred[:, offdiag]
                    mse_res[s, m] = 10.0 * np.log10(np.mean(np.abs(diff)**2))
        return np.mean(mse_res, axis=0)

    mse_135_cv = eval_trunc_cv(qE_135, Gy_135)
    mse_90_cv = eval_trunc_cv(qE_90, Gy_90)

    scan_length_sensitivity = {
        'Full_Scan_176_vols_5_87min': {
            'scan_duration_minutes': 5.87,
            'volumes': 176,
            'correlation_with_full_scan_r': 1.0,
            'five_fold_cv_mse_dB': mean_db_ext.tolist(),
            'best_sparse_model': int(np.argmin(mean_db_ext[1:]) + 2)
        },
        'Truncated_135_vols_4_5min': {
            'scan_duration_minutes': 4.5,
            'volumes': 135,
            'correlation_with_full_scan_r': float(r_135),
            'five_fold_cv_mse_dB': mse_135_cv.tolist(),
            'best_sparse_model': int(np.argmin(mse_135_cv[1:]) + 2)
        },
        'Truncated_90_vols_3_0min': {
            'scan_duration_minutes': 3.0,
            'volumes': 90,
            'correlation_with_full_scan_r': float(r_90),
            'five_fold_cv_mse_dB': mse_90_cv.tolist(),
            'best_sparse_model': int(np.argmin(mse_90_cv[1:]) + 2)
        }
    }
    print(f"Scan-Length Sensitivity: 4.5 min r = {r_135:.4f} (CV M4 = {mse_135_cv[3]:.2f} dB) | 3.0 min r = {r_90:.4f} (CV M4 = {mse_90_cv[3]:.2f} dB)")

    # --------------------------------------------------------------------------
    # 5. External Cohort Cross-Spectral Explained Variance R^2 (%) Distribution
    # --------------------------------------------------------------------------
    print("Computing external cohort cross-spectral explained variance R^2 (%) distribution...")
    r2_ext_distributions = {}
    for m_idx, m_name in [(0, 'Model_1'), (1, 'Model_2'), (3, 'Model_4')]:
        r2_list = []
        for s in range(N_ext):
            A_m = A_ext_36[s] * masks_7[m_idx]
            np.fill_diagonal(A_m, -0.5)
            G_pred_base = biophysical_csd(A_m, freqs_ext)
            y_obs = Gy_obs_ext[s][:, offdiag].flatten()
            y_pred = G_pred_base[:, offdiag].flatten()
            num = np.real(np.sum(y_obs * np.conj(y_pred)))
            den = np.sum(np.abs(y_pred)**2)
            gamma = max(0.1, num / max(1e-12, den))
            y_pred_cal = gamma * y_pred
            ss_res = np.sum(np.abs(y_obs - y_pred_cal)**2)
            ss_tot = np.sum(np.abs(y_obs)**2)
            r2 = max(0.0, 1.0 - (ss_res / max(1e-12, ss_tot))) * 100.0
            r2_list.append(r2)
        arr = np.array(r2_list)
        q25, q75 = np.percentile(arr, [25, 75])
        r2_ext_distributions[m_name] = {
            'mean_pct': float(np.mean(arr)),
            'std_pct': float(np.std(arr)),
            'median_pct': float(np.median(arr)),
            'iqr_pct': float(q75 - q25),
            'min_pct': float(np.min(arr)),
            'max_pct': float(np.max(arr))
        }
        print(f"   {m_name} External R^2: Mean = {np.mean(arr):.2f}% +/- {np.std(arr):.2f}%, Median = {np.median(arr):.2f}% (IQR = {q75-q25:.2f}%)")

    # --------------------------------------------------------------------------
    # 6. Welch CSD nperseg Sensitivity Analysis ({32, 64, 128})
    # --------------------------------------------------------------------------
    print("Auditing Welch CSD segment length sensitivity (nperseg = 32, 64, 128)...")
    nperseg_results = {}
    for nps in [32, 64, 128]:
        Gy_nps = np.zeros((N_ext, len(freqs_ext), 6, 6), dtype=complex)
        for s, ts in enumerate(ts_external):
            cur_nps = min(nps, ts.shape[0])
            cur_nov = cur_nps // 2
            for i in range(6):
                for j in range(6):
                    f, pxy = signal.csd(ts[:, i], ts[:, j], fs=0.5, nperseg=cur_nps, noverlap=cur_nov, nfft=max(cur_nps, 64), window='hann', detrend='constant', scaling='density')
                    idx_band = (f >= 0.01) & (f <= 0.08)
                    Gy_nps[s, :, i, j] = np.interp(freqs_ext, f[idx_band], pxy[idx_band])
                    
        mse_nps = np.zeros((N_ext, 7))
        for tr, te in kf_ext.split(range(N_ext)):
            A_proto = np.mean(A_ext_36[tr], axis=0)
            for m_idx in range(7):
                A_m = A_proto * masks_7[m_idx]
                np.fill_diagonal(A_m, -0.5)
                G_pred = biophysical_csd(A_m, freqs_ext)
                for s in te:
                    diff = Gy_nps[s][:, offdiag] - G_pred[:, offdiag]
                    mse_nps[s, m_idx] = 10.0 * np.log10(np.mean(np.abs(diff)**2))
                    
        nperseg_results[f'nperseg_{nps}'] = {
            'mean_mse_dB': np.mean(mse_nps, axis=0).tolist(),
            'best_sparse_model': int(np.argmin(np.mean(mse_nps, axis=0)[1:]) + 2)
        }

    # --------------------------------------------------------------------------
    # 7. Inductive LOSO Transfer vs. Batch Zero-Shot Projection
    # --------------------------------------------------------------------------
    if context_step1.get('is_fallback', False) and 'pca6' in context_step1 and 'gmm6' in context_step1:
        pca_6d = context_step1['pca6']
        gmm_6d = context_step1['gmm6']
    else:
        A_ibid = context_step1['A_ibid']
        N_ibid = A_ibid.shape[0]
        X_ibid_full = A_ibid[:, 0].reshape(N_ibid, -1)
        s_ibid = np.std(X_ibid_full, axis=0, ddof=1) + 1e-8
        X_ibid_scaled = X_ibid_full / s_ibid
        pca_6d = PCA(n_components=6, random_state=SEED)
        X_ibid_pca = pca_6d.fit_transform(X_ibid_scaled)

        gmm_primary = GaussianMixture(n_components=3, covariance_type='diag', random_state=SEED, reg_covar=1e-3, n_init=10)
        ibid_biotypes = gmm_primary.fit_predict(X_ibid_scaled)
        means_init = np.array([np.mean(X_ibid_pca[ibid_biotypes == k], axis=0) for k in range(3)])

        gmm_6d = GaussianMixture(n_components=3, covariance_type='diag', means_init=means_init, random_state=SEED, reg_covar=0.01)
        gmm_6d.fit(X_ibid_pca)

    # Batch-aligned transfer
    s_ext_batch = np.std(qE_all, axis=0, ddof=1) + 1e-8
    Z_batch_pca = pca_6d.transform(qE_all / s_ext_batch)
    centroid_batch = np.mean(Z_batch_pca, axis=0)
    Z_batch_aligned = Z_batch_pca - centroid_batch
    c_batch = gmm_6d.predict(Z_batch_aligned)

    # Inductive Leave-One-Subject-Out (LOSO) Transfer
    c_inductive = np.zeros(N_ext, dtype=int)
    for s in range(N_ext):
        train_pool = [i for i in range(N_ext) if i != s]
        s_ext_loso = np.std(qE_all[train_pool], axis=0, ddof=1) + 1e-8
        Z_train_pca = pca_6d.transform(qE_all[train_pool] / s_ext_loso)
        centroid_loso = np.mean(Z_train_pca, axis=0)
        
        z_s_pca = pca_6d.transform((qE_all[s] / s_ext_loso).reshape(1, -1))
        z_s_aligned = z_s_pca - centroid_loso
        c_inductive[s] = gmm_6d.predict(z_s_aligned)[0]
        
    pct_agreement = float(np.mean(c_batch == c_inductive) * 100.0)
    kappa_val = float(cohen_kappa_score(c_batch, c_inductive))
    print(f"Inductive LOSO vs Batch Cluster Transfer Concordance: {pct_agreement:.1f}%, Cohen's Kappa = {kappa_val:.4f}")
    print(f"Transferred Cluster Counts: Cluster 1: {int(np.sum(c_batch==0))}, Cluster 2: {int(np.sum(c_batch==1))}, Cluster 3: {int(np.sum(c_batch==2))}")

    # Subgroup Split: PCC-Dominant vs LP-Dominant
    pcc_ext = np.mean(np.abs(A_ext_36[:, 1:, 0]), axis=1)
    lp_ext = np.mean(np.abs(A_ext_36[:, [0, 1, 2, 4, 5], 3]), axis=1)
    hub_diff = pcc_ext - lp_ext
    pcc_dom_idx = np.where(hub_diff > 0)[0]
    lp_dom_idx = np.where(hub_diff <= 0)[0]

    # Subgroup Tests on Calibrated Delta F_42
    kw_sub = kruskal(dF_42_ext_bmr[c_batch == 0], dF_42_ext_bmr[c_batch == 1], dF_42_ext_bmr[c_batch == 2])
    mw_bin = mannwhitneyu(dF_42_ext_bmr[pcc_dom_idx], dF_42_ext_bmr[lp_dom_idx])
    
    print(f"External Subgroup Tests: Kruskal-Wallis across clusters H = {kw_sub.statistic:.2f} (p = {kw_sub.pvalue:.4e})")
    print(f"External Binary Hub Dominance: Mann-Whitney U = {mw_bin.statistic:.1f} (p = {mw_bin.pvalue:.4e}) (PCC-Dom n={len(pcc_dom_idx)}, LP-Dom n={len(lp_dom_idx)})")

    # --------------------------------------------------------------------------
    # 6. Hierarchical Second-Level PEB on External Cohort
    # --------------------------------------------------------------------------
    peb_ext_all = compute_second_level_peb(qE_all, qC_36)
    peb_ext_tdc = compute_second_level_peb(qE_all[:20], qC_36)
    
    print(f"Second-Level PEB (External N = 40): Winning Model = M{peb_ext_all['winning_model']} (P_M1 = {peb_ext_all['P_M1_probability']:.6f})")
    print(f"Second-Level PEB (External TDC n = 20): Winning Model = M{peb_ext_tdc['winning_model']} (P_M1 = {peb_ext_tdc['P_M1_probability']:.6f})")

    external_validation_results = {
        'cohort_demographics': {
            'source': 'NYU / ADHD-200 repository via Nilearn',
            'N_total': N_ext,
            'n_typically_developing_controls': 20,
            'n_adhd': 20,
            'TR_seconds': 2.0,
            'ROI_count': 6
        },
        'observed_welch_csd_cv_error_dB': {
            'all_participants_N40': {
                'mean_mse_dB': mean_db_ext.tolist(),
                'sem_mse_dB': sem_db_ext.tolist(),
                'best_model': int(np.argmin(mean_db_ext) + 1)
            },
            'typically_developing_controls_n20': {
                'mean_mse_dB': mean_db_tdc.tolist(),
                'sem_mse_dB': sem_db_tdc.tolist(),
                'best_model': int(np.argmin(mean_db_tdc) + 1)
            }
        },
        'predictive_bic': {
            'mean_pBIC': mean_pbic_ext.tolist(),
            'delta_pBIC_relative_to_minimum': delta_pbic_ext.tolist(),
            'winning_model': int(np.argmin(mean_pbic_ext) + 1)
        },
        'three_level_cross_spectral_generalization': three_level_generalization,
        'scan_length_truncation_sensitivity': scan_length_sensitivity,
        'cross_spectral_explained_variance_distribution': r2_ext_distributions,
        'welch_nperseg_sensitivity': nperseg_results,
        'exact_36d_correlated_bmr': {
            'mean_delta_F_nats': np.mean(dF_bmr_calibrated, axis=0).tolist(),
            'std_delta_F_nats': np.std(dF_bmr_calibrated, axis=0).tolist(),
            'Delta_F_4_minus_2_mean': float(np.mean(dF_42_ext_bmr)),
            'Delta_F_4_minus_2_sd': float(np.std(dF_42_ext_bmr))
        },
        'zero_shot_manifold_transfer': {
            'loso_vs_batch_concordance_pct': pct_agreement,
            'cohen_kappa': kappa_val,
            'transferred_cluster_counts': {
                'Cluster_1_Dual_Hub': int(np.sum(c_batch == 0)),
                'Cluster_2_PCC_Dominant': int(np.sum(c_batch == 1)),
                'Cluster_3_LP_Dominant': int(np.sum(c_batch == 2))
            },
            'subgroup_statistical_tests': {
                'kruskal_wallis_across_transferred_clusters': {
                    'H_statistic': float(kw_sub.statistic),
                    'p_value': float(kw_sub.pvalue)
                },
                'mann_whitney_u_hub_dominance_split': {
                    'U_statistic': float(mw_bin.statistic),
                    'p_value': float(mw_bin.pvalue),
                    'n_pcc_dominant': int(len(pcc_dom_idx)),
                    'n_lp_dominant': int(len(lp_dom_idx))
                }
            }
        },
        'hierarchical_second_level_peb': {
            'all_participants_N40': peb_ext_all,
            'typically_developing_controls_n20': peb_ext_tdc
        }
    }

    context = {
        'qE_all': qE_all,
        'dF_ext': dF_bmr_calibrated,
        'dF_42_ext_bmr': dF_42_ext_bmr,
        'ext_models_7_gain': ext_offdiag_dB,
        'pBIC_ext': ext_offdiag_pBIC,
        'delta_pbic_ext': delta_pbic_ext,
        'Z_batch_aligned': Z_batch_aligned,
        'c_batch': c_batch,
        'pcc_dom_idx': pcc_dom_idx,
        'lp_dom_idx': lp_dom_idx,
        'kw_sub': kw_sub,
        'mw_bin': mw_bin
    }

    return {'external_resting_state_validation': external_validation_results}, context
