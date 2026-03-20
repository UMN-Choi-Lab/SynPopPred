"""PopLLM: LLM-based synthetic population generator.

Fine-tunes a causal language model (DistilGPT-2 or Llama 3.1 8B) on
serialized census records using LoRA (Low-Rank Adaptation), then
generates new records via auto-regressive text generation.

Key idea: Treat each census record as a "sentence" and train the LLM
to predict the next token. The LLM implicitly learns the joint
distribution P(sex, age, education, occupation, marital_status | year).

This is based on the GReaT (Generation of Realistic Tabular data) approach:
    Borisov et al. (2023). "Language Models are Realistic Tabular Data
    Generators." ICLR 2023.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from config import CATEGORICAL_ATTRIBUTES, CENSUS_ATTRIBUTES, RANDOM_SEED
from serializer import build_generation_prefix, parse_record, serialize_dataframe
from era_context import ERA_NARRATIVES, build_era_prefix

logger = logging.getLogger(__name__)


def check_record_feasibility(record: dict) -> list[str]:
    """Check a record for structural zero violations.

    Returns list of violated rule IDs (empty = valid).
    """
    violations = []
    age = record.get("age_group")
    age_groups = CATEGORICAL_ATTRIBUTES["age_group"]
    if age not in age_groups:
        return violations
    idx = age_groups.index(age)

    edu = record.get("edu")
    marital = record.get("marital_status")
    occ = record.get("occupation")

    # Z1-Z3: Education constraints for young children
    if idx == 0 and edu and edu != "No schooling":
        violations.append("Z1")
    if idx == 1 and edu and edu not in ("No schooling", "Elementary school"):
        violations.append("Z2")
    if idx == 2 and edu and edu not in ("No schooling", "Elementary school", "Middle school"):
        violations.append("Z3")
    # Z4-Z5: Children must be unmarried and not working
    if idx <= 2:
        if marital and marital != "Never married":
            violations.append("Z4")
        if occ and occ != "Not economically active":
            violations.append("Z5")

    return violations


class PopLLMSynthesizer:
    """LLM-based synthetic population generator using LoRA fine-tuning.

    Supports two model sizes:
        - "distilgpt2": 82M params, ~5 min training, good for prototyping
        - "meta-llama/Llama-3.1-8B": 8B params, ~25 min training, best quality
          (or use ungated mirror "unsloth/Meta-Llama-3.1-8B")

    Usage:
        model = PopLLMSynthesizer(model_name="distilgpt2")
        model.fit(train_df)
        synthetic = model.generate(n_samples=5000, year=2015)
    """

    def __init__(
        self,
        model_name: str = "distilgpt2",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: list[str] | None = None,
        max_length: int = 256,
        seed: int = RANDOM_SEED,
    ):
        """Initialize PopLLM.

        Args:
            model_name: HuggingFace model ID.
            lora_rank: LoRA rank (higher = more capacity, slower training).
            lora_alpha: LoRA scaling factor (typically 2x rank).
            lora_target_modules: Which attention layers to adapt.
                Default: ["c_attn"] for GPT-2, ["q_proj", "v_proj"] for Llama.
            max_length: Max token sequence length.
            seed: Random seed for reproducibility.
        """
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.max_length = max_length
        self.seed = seed

        # Auto-select LoRA target modules based on model architecture
        if lora_target_modules is not None:
            self.lora_target_modules = lora_target_modules
        elif "gpt2" in model_name.lower():
            self.lora_target_modules = ["c_attn"]
        else:
            # Llama, Mistral, etc. use separate Q/K/V/O projection layers
            self.lora_target_modules = ["q_proj", "v_proj"]

        self._model = None
        self._tokenizer = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        epochs: int = 5,
        batch_size: int = 32,
        learning_rate: float = 2e-4,
        output_dir: str = "checkpoints/popllm",
        use_era_context: bool = False,
    ) -> "PopLLMSynthesizer":
        """Fine-tune the LLM on serialized census records.

        Pipeline:
            1. serialize_dataframe() converts each row to natural language
            2. Tokenize the text sequences
            3. Apply LoRA adapters to the pretrained model
            4. Train with causal LM objective (next-token prediction)

        Args:
            use_era_context: If True, prepend era demographic narratives
                (TFR, aging %, education trends) to each training record.
                This provides temporal context for the LLM.
        """
        self._use_era_context = use_era_context
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Load pretrained model and tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Use fp16 for large models to fit in GPU memory
        load_kwargs = {}
        if torch.cuda.is_available() and "gpt2" not in self.model_name.lower():
            load_kwargs["dtype"] = torch.float16

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs)

        # Apply LoRA — only fine-tunes a small number of parameters
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.05,
            target_modules=self.lora_target_modules,
        )
        self._model = get_peft_model(base_model, lora_config)

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        print(f"LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

        # Serialize census records to natural language text
        # If use_era_context=True, each record gets a demographic context prefix
        train_texts = serialize_dataframe(
            train_df, permute=True, seed=self.seed,
            use_era_context=use_era_context)
        train_dataset = self._tokenize_texts(train_texts)

        val_dataset = None
        if val_df is not None:
            val_texts = serialize_dataframe(
                val_df, permute=False, seed=self.seed,
                use_era_context=use_era_context)
            val_dataset = self._tokenize_texts(val_texts)

        # Train with HuggingFace Trainer
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=val_dataset is not None,
            seed=self.seed,
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        if val_dataset is not None:
            training_args.eval_strategy = "epoch"

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer, mlm=False)

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        result = trainer.train()
        print(f"Training complete. Final loss: {result.training_loss:.4f}")
        return self

    def generate(
        self,
        n_samples: int,
        year: int,
        region: str = "Daejeon",
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        batch_size: int = 64,
        use_era_context: bool = False,
    ) -> pd.DataFrame:
        """Generate synthetic population records via auto-regressive sampling.

        Pipeline:
            1. Build a generation prefix: "In {year}, a resident of {region}..."
               (optionally with era context: "[Context: TFR=0.84, elderly=15.7%...]")
            2. Feed prefix to the LLM, sample continuations
            3. Parse generated text back to structured records
            4. Filter out malformed or structurally invalid records
        """
        if self._model is None:
            raise RuntimeError("Must call fit() first.")

        # Merge LoRA into base model for faster inference
        try:
            self._model = self._model.merge_and_unload()
        except Exception:
            pass

        self._model.train(False)
        self._model.to(self._device)

        if hasattr(self._model, "generation_config"):
            self._model.generation_config.max_length = None

        # Build generation prefix — with era context if enabled
        if use_era_context:
            prefix = build_era_prefix(year=year, region=region)
        else:
            prefix = build_generation_prefix(year=year, region=region)

        # Allow up to 3x oversampling to account for parse failures
        max_batches = max(int(np.ceil(n_samples / batch_size)) * 3, 10)

        all_records = []
        n_failed = 0

        for batch_i in range(max_batches):
            if len(all_records) >= n_samples:
                break

            records, failed = self._generate_batch(
                prefix, batch_size, temperature, top_k, top_p)
            all_records.extend(records)
            n_failed += failed

            if (batch_i + 1) % 5 == 0:
                total = len(all_records) + n_failed
                rate = len(all_records) / max(total, 1) * 100
                print(f"  Batch {batch_i+1}: {len(all_records)}/{n_samples} "
                      f"records ({rate:.0f}% parse rate)")

        all_records = all_records[:n_samples]
        total = len(all_records) + n_failed
        print(f"Generated {len(all_records)}/{n_samples} records. "
              f"Parse rate: {len(all_records)/max(total,1)*100:.1f}%")

        if not all_records:
            return pd.DataFrame(columns=list(CENSUS_ATTRIBUTES))

        df = pd.DataFrame(all_records)
        out_cols = [c for c in CENSUS_ATTRIBUTES if c in df.columns]
        return df[out_cols].reset_index(drop=True)

    def _generate_batch(self, prefix, batch_size, temperature, top_k, top_p):
        """Generate one batch of text and parse to records."""
        inputs = self._tokenizer(
            [prefix] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length // 2,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        texts = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

        valid_records = []
        n_failed = 0

        for text in texts:
            record = parse_record(text)

            # Check all 5 attributes are present
            if not all(a in record for a in CENSUS_ATTRIBUTES):
                n_failed += 1
                continue

            # Validate attribute values are in allowed set
            valid = True
            for attr in CENSUS_ATTRIBUTES:
                if record[attr] not in CATEGORICAL_ATTRIBUTES[attr]:
                    valid = False
                    break
            if not valid:
                n_failed += 1
                continue

            # Check structural zero constraints
            if check_record_feasibility(record):
                n_failed += 1
                continue

            valid_records.append({a: record[a] for a in CENSUS_ATTRIBUTES})

        return valid_records, n_failed

    def _tokenize_texts(self, texts):
        """Tokenize text sequences into a HuggingFace Dataset."""
        dataset = Dataset.from_dict({"text": texts})

        def tokenize_fn(examples):
            return self._tokenizer(
                examples["text"], truncation=True, max_length=self.max_length)

        return dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    def save(self, path: str):
        """Save LoRA adapter weights and tokenizer."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)

    def load(self, path: str) -> "PopLLMSynthesizer":
        """Load saved LoRA adapter weights."""
        from peft import PeftModel
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model = PeftModel.from_pretrained(base, path)
        self._model.to(self._device)
        return self
