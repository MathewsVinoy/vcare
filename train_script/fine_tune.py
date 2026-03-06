import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from transformers.utils import is_bitsandbytes_available
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer

# -----------------------
# Load dataset
# -----------------------
def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(_stringify(item) for item in value if item is not None).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _extract_input_output(record: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if "input" in record and "output" in record:
        inp = _stringify(record.get("input"))
        out = _stringify(record.get("output"))
        return inp, out

    instruction = _stringify(record.get("instruction") or record.get("prompt"))
    context = _stringify(record.get("context") or record.get("article") or record.get("text"))
    question = _stringify(record.get("question") or record.get("query"))
    raw_input = _stringify(record.get("input"))

    input_parts = [part for part in [instruction, raw_input, question, context] if part]
    inp = "\n\n".join(input_parts).strip()

    out = _stringify(
        record.get("output")
        or record.get("answer")
        or record.get("response")
        or record.get("completion")
        or record.get("label")
    )

    if not inp or out is None:
        return None
    return inp, out


def _iter_records(raw: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield item
    elif isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], list):
            for item in raw["data"]:
                if isinstance(item, dict):
                    yield item
        else:
            yield raw


def load_dataset(path: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                extracted = _extract_input_output(record)
                if extracted:
                    inp, out = extracted
                    records.append({"input": inp, "output": out})
        return records

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    for record in _iter_records(raw):
        extracted = _extract_input_output(record)
        if extracted:
            inp, out = extracted
            records.append({"input": inp, "output": out})

    return records


# Fixed dataset path - it's in a subdirectory
dataset_path = "data/medical_meadow_wikidoc/medical_meadow_wikidoc.json"

if not Path(dataset_path).exists():
    raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

print(f"Loading dataset from: {dataset_path}")
training_data = load_dataset(dataset_path)

if not training_data:
    raise ValueError("No training samples found. Check dataset_path and schema.")

print(f"Loaded {len(training_data)} training samples")
print(f"Sample record: {training_data[0]}")


# -----------------------
# Training function
# -----------------------
def run_training(use_gpu: bool) -> None:
    label = "GPU" if use_gpu else "CPU"
    print(f"\n{'='*50}")
    print(f"Starting {label} training run")
    print(f"{'='*50}\n")

    cuda_bf16 = use_gpu and torch.cuda.is_bf16_supported()
    device_map = "auto" if use_gpu else {"": "cpu"}

    print(f"Using device  : {label}")
    print(f"GPU name      : {torch.cuda.get_device_name(0) if use_gpu else 'N/A'}")
    print(f"Using BF16    : {cuda_bf16}\n")

    dataset_num_proc = 2 if use_gpu else 1
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 16 if use_gpu else 8
    optim = "adamw_8bit" if use_gpu else "adamw_torch"

    # ── Model & tokenizer ────────────────────────────────────
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    max_seq_length = 1024

    use_4bit = False
    bnb_config = None

    if use_gpu and is_bitsandbytes_available():
        use_4bit = True
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cuda_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization")

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=(torch.bfloat16 if cuda_bf16 else torch.float16) if use_gpu else torch.float32,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    if use_gpu and use_4bit:
        model = prepare_model_for_kbit_training(model)
        print("Model prepared for k-bit training")

    # ── Prompt formatting ────────────────────────────────────
    def format_prompt(example: dict) -> str:
        return (
            f"### Input:\n{example['input']}\n\n"
            f"### Output:\n{example['output']}{tokenizer.eos_token}"
        )

    print("Formatting dataset...")
    formatted_data = [format_prompt(item) for item in training_data]
    hf_dataset = Dataset.from_dict({"text": formatted_data})
    print(f"Dataset ready with {len(hf_dataset)} examples")

    # ── LoRA ─────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Configuring model for training...")
    model.config.use_cache = False

    # Enable gradient checkpointing for memory efficiency (also helpful on CPU)
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")

    # ── Training arguments ───────────────────────────────────
    output_dir = f"outputs_{label.lower()}"
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=use_gpu and not cuda_bf16,
        bf16=cuda_bf16,
        logging_steps=25,
        optim=optim,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_pin_memory=use_gpu,
        report_to="none",
        no_cuda=not use_gpu,
    )

    # ── Trainer ──────────────────────────────────────────────
    print("Initializing trainer...")
    trainer = None
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=dataset_num_proc,
            packing=False,
            args=training_args,
        )
        print("Trainer initialized with tokenizer parameter")
    except TypeError as e:
        print(f"First attempt failed: {e}")
        try:
            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=hf_dataset,
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                dataset_num_proc=dataset_num_proc,
                packing=False,
                args=training_args,
            )
            print("Trainer initialized with processing_class parameter")
        except TypeError as e2:
            print(f"Second attempt failed: {e2}")
            trainer = SFTTrainer(
                model=model,
                train_dataset=hf_dataset,
                args=training_args,
                formatting_func=lambda x: x["text"],
            )
            print("Trainer initialized with formatting_func")

    if trainer is None:
        raise RuntimeError("Failed to initialize trainer")

    print(f"\nStarting {label} training...")
    trainer.train()
    print(f"\n{label} training completed!")

    # ── Save ─────────────────────────────────────────────────
    save_dir = f"mistral_lora_model_{label.lower()}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to: {save_dir}\n")

    # Free memory before the next run
    del model, tokenizer, trainer
    if use_gpu:
        torch.cuda.empty_cache()


# -----------------------
# Run on both devices
# -----------------------
if torch.cuda.is_available():
    run_training(use_gpu=True)
else:
    print("No GPU detected — skipping GPU training run.")

run_training(use_gpu=False)