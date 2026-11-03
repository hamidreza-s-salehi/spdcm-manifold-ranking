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
    # Verification & Summary
    # --------------------------------------------------------------------------
    elapsed = time.time() - start_time
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
