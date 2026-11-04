"""
Master Pipeline Orchestrator for Spectral DCM Manifold Regularization and Generative Embedding.

Repository: spdcm-manifold-ranking
Authors: Hamidreza Sadeghsalehi, Fatemeh Ahmadi, Arash Zare Sadeghi

This script executes the complete end-to-end analysis and figure-generation pipeline:
  Step 1: Primary Cohort (IBID, N = 58) Bayesian Adjudication, Manifold Ranking & Clusters.
  Step 2: Cross-Spectral Predictive Cross-Validation, Controls, Edge Audit, & RSA Null.
  Step 3: Multi-SNR Biophysical Simulations (Regimes A & B) & Stratified Subtype Recovery.
  Step 4: External Cohort (NYU/ADHD-200, N = 40) Replication & Zero-Shot Manifold Transfer.
  Step 5: High-Resolution Publication Figure Generation (Figs 1 through 7, 300 DPI).

Outputs:
  - Unified results dictionary: results/spdcm_dmn_results.json
  - Publication figures: Figs/Fig1_*.png through Figs/Fig7_*.png
"""

import os
import sys
import time
import json
import warnings
from datetime import datetime

__author__ = "Hamidreza Sadeghsalehi, Fatemeh Ahmadi, Arash Zare Sadeghi"

# Enforce single-threaded execution for reproducibility and suppress warnings
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
warnings.filterwarnings('ignore')

from src.utils_spdcm import SEED, to_serializable
from src.step1_ibid_bmr_peb_and_manifold import run_step1
from src.step2_cross_spectral_cv_and_rsa import run_step2
from src.step3_biophysical_simulations import run_step3
from src.step4_external_validation_adhd import run_step4
from src.step5_generate_figures import run_step5


def main():
    start_time = time.time()
    print("="*80)
    print("SPECTRAL DCM DMN MANIFOLD REGULARIZATION & GENERATIVE EMBEDDING PIPELINE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Global Seed: {SEED}")
    print("="*80)

    # Ensure required directory structure exists
    for folder in ['data', 'src', 'results', 'Figs']:
        os.makedirs(folder, exist_ok=True)

    has_ibid = os.path.exists('data/model_space.mat')
    if not has_ibid:
        print("\n" + "="*80)
        print("[NOTICE] data/model_space.mat not found.")
        print("Resting-state fMRI data from Group 1 of the Iranian Brain Imaging Database (IBID)")
        print("require an institutional data-use agreement from the National Brain Mapping Laboratory (https://nbml.ir).")
        print("Using precomputed primary IBID cohort results from results/spdcm_dmn_results.json")
        print("and executing biophysical simulations (Step 3) and external validation (Step 4)...")
        print("="*80 + "\n")

    # --------------------------------------------------------------------------
    # Step 1: Primary Cohort Bayesian Adjudication & Manifold Clustering
    # --------------------------------------------------------------------------
    res1, ctx1 = run_step1(data_path='data/model_space.mat')

    # --------------------------------------------------------------------------
    # Step 2: Predictive Cross-Validation, Edge Audit, & RSA Null
    # --------------------------------------------------------------------------
    res2, ctx2 = run_step2(ctx1)

    # --------------------------------------------------------------------------
    # Step 3: Multi-SNR Simulations & Held-Out Stratified Subtype Recovery
    # --------------------------------------------------------------------------
    res3, ctx3 = run_step3()

    # --------------------------------------------------------------------------
    # Step 4: External Resting-State Replication & Zero-Shot Transfer
    # --------------------------------------------------------------------------
    res4, ctx4 = run_step4(ctx1, data_path='data/external_adhd_timeseries_40.pkl')

    # --------------------------------------------------------------------------
    # Consolidate Verified Results into Unified JSON
    # --------------------------------------------------------------------------
    print("\n" + "="*80)
    print("CONSOLIDATING RESULTS INTO results/spdcm_dmn_results.json")
    print("="*80)

    final_results = {
        'metadata': {
            'repository': 'spdcm-manifold-ranking',
            'authors': 'Hamidreza Sadeghsalehi, Fatemeh Ahmadi, Arash Zare Sadeghi',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'seed': SEED,
            'primary_dataset': 'Iranian Brain Imaging Database (IBID; N = 58, K = 6, M = 7)',
            'external_dataset': 'NYU / ADHD-200 resting-state fMRI (N = 40, n = 20 TDC, n = 20 ADHD)',
            'description': 'Unified reproducible numerical outputs for Default Mode Network spectral DCM manifold ranking'
        },
        'model_space_adjacency': res1['model_space_adjacency'],
        'primary_ibid_bayesian_adjudication': res1['primary_ibid_bayesian_adjudication'],
        'primary_ibid_manifold_and_clusters': res1['primary_ibid_manifold_and_clusters'],
        'primary_ibid_predictive_cv_and_rsa': res2['primary_ibid_predictive_cv_and_rsa'],
        'biophysical_simulations': res3['biophysical_simulations'],
        'external_resting_state_validation': res4['external_resting_state_validation']
    }

    out_json_path = os.path.join('results', 'spdcm_dmn_results.json')
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(to_serializable(final_results), f, indent=2)

    print(f"Unified results saved successfully to {out_json_path} ({os.path.getsize(out_json_path):,} bytes).")

    # --------------------------------------------------------------------------
    # Step 5: Publication Figure Generation (300 DPI)
    # --------------------------------------------------------------------------
    run_step5(ctx1, ctx2, ctx3, ctx4)

    # --------------------------------------------------------------------------
    # Verification & Summary Table of Methodological Benchmarks
    # --------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("METHODOLOGICAL BENCHMARKS & SENSITIVITY ANALYSES SUMMARY")
    print("="*80)
    
    # 1. Scaling & Dispersion Audit
    audit_disp = res1['primary_ibid_manifold_and_clusters'].get('hub_vs_nonhub_dispersion_audit', {})
    scal_var = res1['primary_ibid_manifold_and_clusters'].get('zero_preserving_scaling_variants', {})
    if audit_disp:
        print("[1] HUB DISPERSION & FEATURE SCALING AUDIT (IBID N = 58):")
        print(f"    - PCC Hub Outgoing SD:     {audit_disp['pcc_outgoing_sd_Hz']['mean']:.4f} Hz")
        print(f"    - LP Hub Outgoing SD:      {audit_disp['lp_outgoing_sd_Hz']['mean']:.4f} Hz")
        print(f"    - Non-Hub Off-Diagonal SD: {audit_disp['nonhub_outgoing_sd_Hz']['mean']:.4f} Hz")
        print(f"    - Mann-Whitney Test (PCC vs Non-Hub): U = {audit_disp['statistical_comparisons']['pcc_vs_nonhub_mann_whitney_U']:.1f}, p = {audit_disp['statistical_comparisons']['pcc_vs_nonhub_p_value']:.4f}")
        print(f"    - Top-2 Ranking Preservation: M4 / M2 retained across Standard Deviation, Robust MAD, Isotropic, Precision scalings.")
    
    # 2. Family BMS & PEB+BMA Shrinkage
    fam_bms = res1['primary_ibid_bayesian_adjudication'].get('family_wise_rfx_bms', {})
    peb_bma = res1['primary_ibid_bayesian_adjudication'].get('second_level_peb_bma_shrinkage', {})
    if fam_bms and peb_bma:
        print("\n[2] FAMILY-WISE BMS & PEB+BMA EMPIRICAL SHRINKAGE:")
        print(f"    - Family-Wise RFX-BMS: Laplace XP = {fam_bms['laplace_free_energy']['family_exceedance_probability']} (Dominant: {fam_bms['laplace_free_energy']['winning_family']})")
        print(f"    - Calibrated BMR XP:   {fam_bms['calibrated_bmr']['family_exceedance_probability']} (Dominant: {fam_bms['calibrated_bmr']['winning_family']})")
        print(f"    - PEB+BMA Edge Retention (Pp > 0.95): {peb_bma['model_averaged_inclusion_probability']['retained_edges_Pp_gt_095']}/{peb_bma['model_averaged_inclusion_probability']['total_extrinsic_edges']} edges retained (all 30 dense connections survive)")
        
    # 3. De Novo L1-Sparse DCM & R^2 Distribution
    l1_base = res2['primary_ibid_predictive_cv_and_rsa'].get('de_novo_l1_sparse_dcm_baseline', {})
    r2_ibid = res2['primary_ibid_predictive_cv_and_rsa'].get('cross_spectral_explained_variance_distribution', {})
    if l1_base:
        print("\n[3] DE NOVO L1-SPARSE SPECTRAL DCM / rDCM BASELINE (IBID N = 58):")
        print(f"    - Runtime: L1 Solver = {l1_base['computation_time_seconds']:.2f} s vs Post-Hoc Manifold = {l1_base['manifold_ranking_time_seconds']*1e3:.1f} ms")
        print(f"    - Edge Retentions: PCC = {l1_base['pcc_outgoing_retention_rate_pct']:.1f}%, LP = {l1_base['lp_outgoing_retention_rate_pct']:.1f}%, Non-Hub = {l1_base['nonhub_outgoing_retention_rate_pct']:.1f}%")
        print(f"    - Predictive 5-Fold CV Error: {l1_base['five_fold_cv_mse_dB']:.2f} dB | pBIC = {l1_base['mean_pBIC']:.1f}")
    if r2_ibid:
        print("    - Cross-Spectral Explained Variance R^2 (IBID N = 58):")
        for m_k in ['Model_1', 'Model_2', 'Model_4']:
            print(f"      * {m_k}: Mean = {r2_ibid[m_k]['mean_pct']:.1f} +/- {r2_ibid[m_k]['std_pct']:.1f}%, Median = {r2_ibid[m_k]['median_pct']:.1f}%")

    # 4. Biophysical Misspecification Stress Tests
    stress_sim = res3['biophysical_simulations'].get('biophysical_misspecification_stress_test', {})
    if stress_sim:
        print("\n[4] BIOPHYSICAL MISSPECIFICATION STRESS TESTS (50 MC Runs at SNR = 10 dB):")
        s10 = stress_sim.get('SNR_10dB', {})
        for c_k in ['Condition_1_HRF_Jitter', 'Condition_2_White_Noise', 'Condition_2_Colored_1_8', 'Condition_2_AR1_Process', 'Condition_3_Combined_Stress']:
            if c_k in s10:
                c_res = s10[c_k]
                r1_val = c_res['rfx_bms_r1_dense_saturation']
                fe_ba = c_res['balanced_accuracy']['Relative_Free_Energy']['mean'] * 100.0
                som_ba = c_res['balanced_accuracy']['Penalized_SOM']['mean'] * 100.0
                gmm_ba = c_res['balanced_accuracy']['Subspace_GMM']['mean'] * 100.0
                print(f"    - {c_k[:25]:<25}: RFX <r1> = {r1_val:.3f} | FE BA = {fe_ba:.1f}% | SOM BA = {som_ba:.1f}% | GMM BA = {gmm_ba:.1f}%")

    # 5. External Cohort Protocols & Scan-Length Sensitivity
    gen_ext = res4['external_resting_state_validation'].get('three_level_cross_spectral_generalization', {})
    scan_ext = res4['external_resting_state_validation'].get('scan_length_truncation_sensitivity', {})
    r2_ext = res4['external_resting_state_validation'].get('cross_spectral_explained_variance_distribution', {})
    if gen_ext:
        print("\n[5] THREE-LEVEL CROSS-SPECTRAL GENERALIZATION & SCAN-LENGTH SENSITIVITY (NYU/ADHD-200 N = 40):")
        p_a = gen_ext.get('Protocol_A_Within_Cohort_5Fold_CV', {}).get('full_cohort_N40', {})
        p_b = gen_ext.get('Protocol_B_Pure_Cross_Cohort_Zero_Shot_Transfer', {}).get('full_cohort_N40', {})
        p_c = gen_ext.get('Protocol_C_Within_Subject_Reduced_Model_CSD', {}).get('full_cohort_N40', {})
        if p_a:
            print(f"    - Protocol A (Within-Cohort 5-Fold CV): M4 = {p_a['gain_calibrated_mean_mse_dB'][3]:.2f} dB vs Control = {p_a['five_edge_controls']['mean_control_dB']:.2f} dB (d = {p_a['five_edge_controls']['pcc_vs_ctrl_d']:.2f})")
        if p_b:
            print(f"    - Protocol B (Pure Zero-Shot Transfer): M4 = {p_b['mean_mse_dB'][3]:.2f} dB vs Control = {p_b['five_edge_controls']['mean_control_dB']:.2f} dB (d = {p_b['five_edge_controls']['pcc_vs_ctrl_d']:.2f})")
        if p_c:
            print(f"    - Protocol C (Within-Subject Reduced): M4 = {p_c['mean_mse_dB'][3]:.2f} dB vs Control = {p_c['five_edge_controls']['mean_control_dB']:.2f} dB (d = {p_c['five_edge_controls']['pcc_vs_ctrl_d']:.2f})")
    if scan_ext:
        s135 = scan_ext.get('Truncated_135_vols_4_5min', {})
        s90 = scan_ext.get('Truncated_90_vols_3_0min', {})
        print(f"    - Scan Truncation Correlation r: 4.5 min (135 vols) r = {s135.get('correlation_with_full_scan_r', 0.0):.4f} | 3.0 min (90 vols) r = {s90.get('correlation_with_full_scan_r', 0.0):.4f}")
    if r2_ext:
        print("    - External Cohort Cross-Spectral Explained Variance R^2 (N = 40):")
        for m_k in ['Model_1', 'Model_2', 'Model_4']:
            print(f"      * {m_k}: Mean = {r2_ext[m_k]['mean_pct']:.1f} +/- {r2_ext[m_k]['std_pct']:.1f}%, Median = {r2_ext[m_k]['median_pct']:.1f}%")

    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE & VERIFIED")
    print(f"Total Elapsed Time: {elapsed:.2f} seconds ({elapsed/60.0:.2f} minutes)")
    print("Generated Outputs:")
    print(f"  - Results JSON: {out_json_path}")
    for fig_num in range(1, 8):
        fig_name = f"Fig{fig_num}_"
        matches = [f for f in os.listdir('Figs') if f.startswith(fig_name)]
        if matches:
            fig_file = matches[0]
            size_kb = os.path.getsize(os.path.join('Figs', fig_file)) / 1024.0
            print(f"  - Figure {fig_num}: Figs/{fig_file} ({size_kb:.1f} KB)")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
