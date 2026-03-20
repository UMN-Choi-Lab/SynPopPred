#!/usr/bin/env python3
"""Run all 4 synthetic population methods and compare results.

This script demonstrates the full pipeline:
    1. Load census microdata (Daejeon, South Korea, 1990-2020)
    2. Split into training (1990-2010) and target (2015) sets
    3. Run 4 methods: IPF, CTGAN, DistilGPT-2 (PopLLM), Llama 8B (PopLLM)
    4. Evaluate each with SRMSE and JSD metrics
    5. Print a comparison table

Usage:
    python run_all.py                    # Run all methods
    python run_all.py --methods ipf      # Run only IPF
    python run_all.py --methods ipf llm  # Run IPF + DistilGPT-2
    python run_all.py --n-samples 1000   # Generate fewer samples (faster)

Data setup:
    Copy the harmonized Parquet files into the data/ subdirectory:
        data/census_harmonized_all.parquet   (all years combined)
    OR individual year files:
        data/census_harmonized_1990.parquet
        ...
        data/census_harmonized_2020.parquet
"""

import argparse
import time

import pandas as pd

from config import CENSUS_ATTRIBUTES, CATEGORICAL_ATTRIBUTES
from evaluate import evaluate, ipf_reweight
from ipf_synth import IPFSynthesizer


def load_data(data_dir: str = "data"):
    """Load and split census data into train and target sets.

    Training: 1990-2010 (5 census waves)
    Target:   2015 (the year we're trying to synthesize)
    """
    try:
        df = pd.read_parquet(f"{data_dir}/census_harmonized_all.parquet")
    except FileNotFoundError:
        # Try loading individual year files
        dfs = []
        for year in [1990, 1995, 2000, 2005, 2010, 2015, 2020]:
            try:
                ydf = pd.read_parquet(f"{data_dir}/census_harmonized_{year}.parquet")
                ydf["year"] = year
                dfs.append(ydf)
            except FileNotFoundError:
                continue
        if not dfs:
            raise FileNotFoundError(
                "No data found. Copy census_harmonized_*.parquet files to data/")
        df = pd.concat(dfs, ignore_index=True)

    train_years = [1990, 1995, 2000, 2005, 2010]
    target_year = 2015

    df["year"] = df["year"].astype(int)
    train_df = df[df["year"].isin(train_years)].reset_index(drop=True)
    target_df = df[df["year"] == target_year].reset_index(drop=True)

    print(f"Training data: {len(train_df):,} records ({train_years})")
    print(f"Target data:   {len(target_df):,} records (year {target_year})")
    return train_df, target_df, target_year


def run_ipf(train_df, target_df, n_samples):
    """Run IPF baseline."""
    print("\n" + "=" * 60)
    print("METHOD 1: IPF (Iterative Proportional Fitting)")
    print("=" * 60)

    t0 = time.time()
    ipf = IPFSynthesizer()
    ipf.fit(train_df)

    # Compute target marginals from the actual target year
    target_marginals = IPFSynthesizer.compute_marginals(target_df)
    syn_df = ipf.generate(n_samples, target_marginals=target_marginals)
    elapsed = time.time() - t0

    results = evaluate(syn_df, target_df)
    results["time_sec"] = elapsed
    results["method"] = "IPF"
    print(f"  Time: {elapsed:.1f}s | SRMSE: {results['srmse_avg']:.4f} | "
          f"JSD: {results['jsd_full_joint']:.4f}")
    return results, syn_df


def run_ctgan(train_df, target_df, target_year, n_samples):
    """Run CTGAN baseline."""
    print("\n" + "=" * 60)
    print("METHOD 2: CTGAN (Conditional Tabular GAN)")
    print("=" * 60)

    from ctgan_synth import CTGANWrapper

    t0 = time.time()
    ctgan = CTGANWrapper(epochs=300, batch_size=500)
    ctgan.fit(train_df)
    syn_df = ctgan.generate(n_samples, target_year=target_year)
    elapsed = time.time() - t0

    # Also evaluate with IPF reweighting
    target_marginals = IPFSynthesizer.compute_marginals(target_df)
    syn_ipf = ipf_reweight(syn_df, target_marginals)

    results_raw = evaluate(syn_df, target_df)
    results_ipf = evaluate(syn_ipf, target_df)

    results = {
        "method": "CTGAN",
        "time_sec": elapsed,
        "srmse_avg": results_raw["srmse_avg"],
        "jsd_full_joint": results_raw["jsd_full_joint"],
        "srmse_avg_ipf": results_ipf["srmse_avg"],
        "jsd_full_joint_ipf": results_ipf["jsd_full_joint"],
    }
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Raw:  SRMSE={results_raw['srmse_avg']:.4f} | JSD={results_raw['jsd_full_joint']:.4f}")
    print(f"  +IPF: SRMSE={results_ipf['srmse_avg']:.4f} | JSD={results_ipf['jsd_full_joint']:.4f}")
    return results, syn_df


def run_llm(train_df, target_df, target_year, n_samples,
            model_name="distilgpt2", label="DistilGPT-2"):
    """Run PopLLM (LLM fine-tuning)."""
    print("\n" + "=" * 60)
    print(f"METHOD: PopLLM ({label})")
    print("=" * 60)

    from popllm import PopLLMSynthesizer

    # Configure LoRA based on model size
    if "gpt2" in model_name.lower():
        lora_rank, lora_alpha = 16, 32
        lora_targets = ["c_attn"]
        batch_size, epochs = 32, 5
        temp = 0.8
    else:
        # Llama 8B: use optimized config from autoresearch
        lora_rank, lora_alpha = 32, 32
        lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        batch_size, epochs = 16, 2
        temp = 0.95

    t0 = time.time()
    model = PopLLMSynthesizer(
        model_name=model_name,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target_modules=lora_targets,
    )

    # Subsample training data for speed (optional)
    if len(train_df) > 20000 and "gpt2" in model_name.lower():
        train_sub = train_df.sample(n=20000, random_state=42)
    elif len(train_df) > 80000:
        train_sub = train_df.sample(n=80000, random_state=42)
    else:
        train_sub = train_df

    model.fit(train_sub, epochs=epochs, batch_size=batch_size)
    syn_df = model.generate(n_samples, year=target_year, temperature=temp)
    elapsed = time.time() - t0

    # Evaluate raw and with IPF reweighting
    target_marginals = IPFSynthesizer.compute_marginals(target_df)
    syn_ipf = ipf_reweight(syn_df, target_marginals)

    results_raw = evaluate(syn_df, target_df)
    results_ipf = evaluate(syn_ipf, target_df)

    results = {
        "method": label,
        "time_sec": elapsed,
        "n_generated": len(syn_df),
        "parse_rate": len(syn_df) / max(n_samples, 1),
        "srmse_avg": results_raw["srmse_avg"],
        "jsd_full_joint": results_raw["jsd_full_joint"],
        "srmse_avg_ipf": results_ipf["srmse_avg"],
        "jsd_full_joint_ipf": results_ipf["jsd_full_joint"],
    }
    print(f"  Time: {elapsed:.1f}s | Parse rate: {results['parse_rate']:.1%}")
    print(f"  Raw:  SRMSE={results_raw['srmse_avg']:.4f} | JSD={results_raw['jsd_full_joint']:.4f}")
    print(f"  +IPF: SRMSE={results_ipf['srmse_avg']:.4f} | JSD={results_ipf['jsd_full_joint']:.4f}")
    return results, syn_df


def main():
    parser = argparse.ArgumentParser(description="Synthetic Population Tutorial")
    parser.add_argument("--methods", nargs="+",
                        default=["ipf", "ctgan", "llm", "llama"],
                        choices=["ipf", "ctgan", "llm", "llama"],
                        help="Methods to run")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Number of synthetic records to generate")
    parser.add_argument("--data-dir", default="data",
                        help="Path to data directory")
    parser.add_argument("--llama-model", default="unsloth/Meta-Llama-3.1-8B",
                        help="HuggingFace model ID for Llama")
    args = parser.parse_args()

    train_df, target_df, target_year = load_data(args.data_dir)

    all_results = []

    if "ipf" in args.methods:
        r, _ = run_ipf(train_df, target_df, args.n_samples)
        all_results.append(r)

    if "ctgan" in args.methods:
        r, _ = run_ctgan(train_df, target_df, target_year, args.n_samples)
        all_results.append(r)

    if "llm" in args.methods:
        r, _ = run_llm(train_df, target_df, target_year, args.n_samples,
                       model_name="distilgpt2", label="DistilGPT-2")
        all_results.append(r)

    if "llama" in args.methods:
        r, _ = run_llm(train_df, target_df, target_year, args.n_samples,
                       model_name=args.llama_model, label="Llama-3.1-8B")
        all_results.append(r)

    # Print comparison table
    if all_results:
        print("\n" + "=" * 72)
        print("COMPARISON TABLE")
        print("=" * 72)
        print(f"{'Method':<18} {'Time':>8} {'SRMSE':>8} {'JSD':>8} "
              f"{'SRMSE+IPF':>10} {'JSD+IPF':>10}")
        print("-" * 72)
        for r in all_results:
            srmse_ipf = r.get("srmse_avg_ipf", r["srmse_avg"])
            jsd_ipf = r.get("jsd_full_joint_ipf", r["jsd_full_joint"])
            print(f"{r['method']:<18} {r['time_sec']:>7.1f}s "
                  f"{r['srmse_avg']:>8.4f} {r['jsd_full_joint']:>8.4f} "
                  f"{srmse_ipf:>10.4f} {jsd_ipf:>10.4f}")
        print("-" * 72)
        print("Lower = better for all metrics")


if __name__ == "__main__":
    main()
