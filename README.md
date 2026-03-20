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
