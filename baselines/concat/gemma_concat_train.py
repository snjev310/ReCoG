import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import argparse

parser = argparse.ArgumentParser(description="Concat (MRL || HRL) to ELRL training (Gemma)")
parser.add_argument('--model_id', type=str, required=True)
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--sheet_name', type=str, required=True)
parser.add_argument('--src_col1', type=str, required=True, help="Column name for source language 1 (MRL)")
parser.add_argument('--src_col2', type=str, required=True, help="Column name for source language 2 (HRL)")
parser.add_argument('--src_lang1', type=str, required=True, help="Language name for source 1 (MRL)")
parser.add_argument('--src_lang2', type=str, required=True, help="Language name for source 2 (HRL)")
parser.add_argument('--target_column', type=str, required=True)
parser.add_argument('--tgt_lang', type=str, required=True, help="Target language name")
parser.add_argument('--output_dir', type=str, required=True)

args = parser.parse_args()

MODEL_ID = args.model_id
OUTPUT_DIR = args.output_dir
FILE_PATH = args.input_file
SHEET_NAME = args.sheet_name
SRC_COL1 = args.src_col1
SRC_COL2 = args.src_col2
SRC_LANG1 = args.src_lang1
SRC_LANG2 = args.src_lang2
TARGET_COL = args.target_column
TGT_LANG = args.tgt_lang

# Prompt format matches Appendix A.2 (Listing 1) of the paper exactly.
# Training: user turn + assistant turn (target text).
TRAIN_PROMPT_TEMPLATE = (
    "Translate the following {src_lang1} and {src_lang2} text to {tgt_lang}.\n"
    "Input: {src_text}\n"
    "Respond with only the translation.\n"
    "{tgt_text}"
)

def load_and_format_data(file_path, sheet_name):
    print(f"Loading data from: {file_path}")
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    df.dropna(subset=[SRC_COL1, SRC_COL2, TARGET_COL], inplace=True)

    # Concatenate the two source columns with a separator
    df["combined_input"] = df[SRC_COL1].astype(str) + " || " + df[SRC_COL2].astype(str)

    hf_dataset = Dataset.from_pandas(df[["combined_input", TARGET_COL]])

    dataset_dict = hf_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Total training samples: {len(dataset_dict['train'])}")
    print(f"Total validation samples: {len(dataset_dict['test'])}")
    return dataset_dict

def formatting_func(example):
    prompt = TRAIN_PROMPT_TEMPLATE.format(
        src_lang1=SRC_LANG1,
        src_lang2=SRC_LANG2,
        tgt_lang=TGT_LANG,
        src_text=example["combined_input"],
        tgt_text=example[TARGET_COL]
    )
    return {"text": prompt}

dataset_dict = load_and_format_data(FILE_PATH, SHEET_NAME)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj"]
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    logging_steps=10,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="none",
)

# Apply formatting
dataset_dict['train'] = dataset_dict['train'].map(formatting_func, remove_columns=["combined_input", TARGET_COL])
dataset_dict['test'] = dataset_dict['test'].map(formatting_func, remove_columns=["combined_input", TARGET_COL])

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    peft_config=peft_config,
)

print("Starting LoRA Fine-Tuning...")
trainer.train()

trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"LoRA adapters and tokenizer saved to {OUTPUT_DIR}")