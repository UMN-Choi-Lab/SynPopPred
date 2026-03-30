# Synthetic Population Generation Tutorial

Compare four methods for generating synthetic census populations:

| Method | Approach | Key Library |
|--------|----------|-------------|
| **IPF** | Traditional statistical fitting | `ipfn` |
| **CTGAN** | Conditional Tabular GAN | `sdv` |
| **DistilGPT-2** | LLM fine-tuning (82M params) | `transformers` + `peft` |
| **Llama 3.1 8B** | LLM fine-tuning (8B params) | `transformers` + `peft` |

## Setup

```bash
pip install -r requirements.txt
```

### Data

Copy the harmonized Parquet files into `data/`:

```
data/census_harmonized_all.parquet    # All years combined (preferred)
# OR individual files:
data/census_harmonized_1990.parquet
data/census_harmonized_1995.parquet
...
data/census_harmonized_2020.parquet
```

These files contain Daejeon (South Korea) census microdata with 5 attributes:
- `sex`, `age_group`, `edu`, `occupation`, `marital_status`
- 7 census years: 1990, 1995, 2000, 2005, 2010, 2015, 2020

### For Llama 3.1 8B

Requires a GPU with >= 24 GB VRAM. Uses the ungated mirror `unsloth/Meta-Llama-3.1-8B` by default (no HuggingFace license needed).

## Quick Start

```bash
# Run all methods (requires GPU for LLM methods)
python run_all.py

# Run only IPF (no GPU needed, ~5 seconds)
python run_all.py --methods ipf

# Run IPF + CTGAN (no GPU needed, ~10 minutes for CTGAN training)
python run_all.py --methods ipf ctgan

# Run IPF + DistilGPT-2 (small GPU OK, ~5 minutes)
python run_all.py --methods ipf llm

# Generate fewer samples for faster testing
python run_all.py --n-samples 1000

# Use a specific Llama model
python run_all.py --methods llama --llama-model meta-llama/Llama-3.1-8B
```

### Best Configuration (Llama 8B, JSD=0.043)

To reproduce the best result from 74 hyperparameter experiments:

```bash
# Full best config: Llama 8B + 8x oversampling
# Requires 48 GB VRAM (RTX 6000 Ada), ~65 min training, ~160 min generation
python run_all.py --methods llama --n-samples 25000 --oversample 8

# Faster version: 2x oversampling (~40 min generation)
python run_all.py --methods llama --n-samples 25000 --oversample 2

# Quick test: small sample, no oversampling
python run_all.py --methods llama --n-samples 1000
```

The optimized Llama 8B config is automatically applied when `--methods llama`:
- **LoRA**: rank=32, alpha=32, targets=q/k/v/o_proj, rsLoRA enabled
- **Training**: lr=1.5e-4, 2 epochs, cosine LR scheduler, weight_decay=0
- **Generation**: temp=0.95, top_k=50, top_p=0.95
- **Data**: Uses full training data (no subsampling) — older census waves help at lower lr
- **Oversampling**: `--oversample N` generates N times more records for IPF pool

Key insight: generating many more records than needed and letting IPF reweight
the full pool is the single biggest quality improvement (JSD 0.062 → 0.043).

## Files

| File | Description |
|------|-------------|
| `config.py` | Census attribute schema and structural zero rules |
| `serializer.py` | Record ↔ text conversion for LLM training |
| `ipf_synth.py` | IPF baseline (traditional method) |
| `ctgan_synth.py` | CTGAN baseline (GAN-based) |
| `popllm.py` | PopLLM (LLM fine-tuning with LoRA) |
| `evaluate.py` | SRMSE and JSD evaluation metrics + IPF reweighting |
| `run_all.py` | Main script — trains, generates, evaluates, compares |

## Evaluation Metrics

- **SRMSE** (Standardized RMSE): How well marginal distributions match.
  Compares per-category frequencies (e.g., % Male, % Female).
- **JSD** (Jensen-Shannon Divergence): How well the joint distribution matches.
  Compares the full 5-way contingency table (8,064 cells).
- **+IPF** suffix: Results after applying IPF reweighting post-processing
  to improve marginal consistency.

Both metrics: **lower = better**, 0 = perfect match.

## Method Summary

### IPF (Iterative Proportional Fitting)
The traditional method. Builds a contingency table from training data,
then adjusts cell counts to match target marginals. Guarantees marginal
consistency by design, but assumes conditional independence.

### CTGAN
A conditional GAN that learns the joint distribution implicitly through
adversarial training. Can be conditioned on year. No marginal guarantees.

### PopLLM (DistilGPT-2 / Llama 8B)
Fine-tunes a pretrained language model on serialized census records using
LoRA. Each record becomes a "sentence" like:
> In 2010, a resident of Daejeon was recorded: Sex is Female, Age Group is 30-34, ...

The LLM learns P(attributes | year) through next-token prediction.
At inference, we generate text completions and parse them back to records.

## Hyperparameter Findings (from 74 automated experiments)

| Parameter | Optimal | Impact |
|-----------|---------|--------|
| Learning rate | **1.5e-4** | Critical — broke the 0.069 JSD plateau (extremely sharp optimum) |
| rsLoRA | **True** | Improves raw JSD from 0.120 → 0.088 |
| Cosine LR | **Yes** | Additive benefit with rsLoRA |
| Weight decay | **0.0** | Only helps with full data + lr=1.5e-4 |
| LoRA rank | **32** | Sweet spot (16 too small, 48/64 no benefit) |
| LoRA dropout | **0.05** | Critical — dropout=0 causes catastrophic collapse |
| Temperature | **0.95** | Sharp optimum: 0.90 and 1.0 both worse |
| Training data | **Full 124K** | Only helps at lr=1.5e-4 (at lr=2e-4, old data hurts) |
| Epochs | **2** | 3 epochs overfits |
| Oversampling | **8x** | 25K→0.062, 50K→0.053, 100K→0.046, 200K→0.043 |

### What doesn't work
- EOS token in training → 0% parse rate
- Bayesian Network attribute ordering → breaks parsing
- Repetition penalty → distorts distributions
- Era context narratives → doubles sequence length, underfits at max_length=256
  (may work with max_length=512, untested)
- NEFTune noise → degrades joint distribution
- Label smoothing → hurts categorical accuracy
