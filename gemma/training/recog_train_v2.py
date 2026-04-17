import json
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from torch.utils.data import DataLoader, Dataset
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Trainer, TrainingArguments
from peft import PeftModelForCausalLM, get_peft_model, LoraConfig, TaskType, PeftModel
from tqdm import tqdm # For tracking progress
from sklearn.model_selection import train_test_split
# from datasets import load_metric
from transformers.trainer_utils import EvalLoopOutput

# --- ARGUMENT PARSING (Unchanged) ---
parser = argparse.ArgumentParser()
parser.add_argument("--excel_path", type=str, required=True, help="Path to the Excel file containing translation data.")
parser.add_argument("--sheet_name", type=str, required=True, help="Sheet name in the Excel file.")
parser.add_argument("--train_json", type=str, default="train.jsonl")
parser.add_argument("--val_json", type=str, default="val.jsonl")
parser.add_argument("--output_gating_model", type=str, default="gating_model.pt", help="File path to save the trained gating network weights.")
parser.add_argument("--output_tokenizer_dir", type=str, default="tokenizer", help="Directory to save the tokenizer.")
parser.add_argument("--src1_col", type=str, default="Bengali", help="Column name for source language 1 (e.g., Hindi).")
parser.add_argument("--src2_col", type=str, default="English", help="Column name for source language 2 (e.g., English).")
parser.add_argument("--tgt_col", type=str, default="Magahi", help="Column name for target language (e.g., Angika).")
parser.add_argument("--src1_lang", type=str, default="Bengali", help="Language name for source 1.")
parser.add_argument("--src2_lang", type=str, default="English", help="Language name for source 2.")
parser.add_argument("--tgt_lang", type=str, default="Magahi", help="Target language name.")
parser.add_argument("--base_model", type=str, required=True, help="Base Gemma model ID (e.g., google/gemma-7b).")
parser.add_argument("--lora_hi", type=str, required=True, help="Path to the Hindi LoRA adapter weights.")
parser.add_argument("--lora_en", type=str, required=True, help="Path to the English LoRA adapter weights.")
parser.add_argument("--gating_log_path", type=str, default="gating_values_train.jsonl", help="Path to save gating values.")
args, unknown = parser.parse_known_args() # Use parse_known_args to ignore notebook arguments

# --- Data Preparation (Unchanged) ---
def prepare_triplet_jsonl():
    """Reads Excel, splits data, and saves to JSONL files."""
    df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name)
    df = df[[args.src1_col, args.src2_col, args.tgt_col]].dropna()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    def save_jsonl(df, path):
        with open(path, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    save_jsonl(train_df, args.train_json)
    save_jsonl(val_df, args.val_json)

class combinedGatingDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Prompt format matches Appendix A.2 (Listing 1) of the paper exactly.
        # The user turn ends with "Respond with only the translation."
        # The assistant turn contains only the target text.
        self.train_prompt_template = (
            "Translate the following {src_lang} text to {tgt_lang}.\n"
            "Input: {src_text}\n"
            "Respond with only the translation.\n"
            "{tgt_text}"
        )

    def clean_text(self, x):
        if not isinstance(x, str):
            x = str(x)
        x = x.strip()
        x = x.replace("\u200b", "")
        x = x.replace("\ufeff", "")
        return x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        src1 = self.clean_text(ex[args.src1_col])
        src2 = self.clean_text(ex[args.src2_col])
        tgt  = self.clean_text(ex[args.tgt_col])

        prompt_hi = self.train_prompt_template.format(
            src_lang=args.src1_lang,
            tgt_lang=args.tgt_lang,
            src_text=src1,
            tgt_text=tgt
        )

        prompt_en = self.train_prompt_template.format(
            src_lang=args.src2_lang,
            tgt_lang=args.tgt_lang,
            src_text=src2,
            tgt_text=tgt
        )

        tokenized_hi = self.tokenizer(
            prompt_hi,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        tokenized_en = self.tokenizer(
            prompt_en,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        target_tokens = self.tokenizer(
            tgt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        ).input_ids.squeeze()

        target_tokens[target_tokens == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids_hi': tokenized_hi.input_ids.squeeze(),
            'attention_mask_hi': tokenized_hi.attention_mask.squeeze(),
            'input_ids_en': tokenized_en.input_ids.squeeze(),
            'attention_mask_en': tokenized_en.attention_mask.squeeze(),
            'labels': target_tokens
        }

# --- Gating Network (Unchanged) ---
class GatingNetwork(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),  # 3 layers from hi + 3 from en
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h1, h2):
        x = torch.cat([h1, h2], dim=-1)
        g = self.ffn(x).squeeze(-1)
        return g

class CombinedGatingTrainer(Trainer):
    def __init__(self, model_hi, model_en, gating_model, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.model_hi = model_hi
        self.model_en = model_en
        self.gating_model = gating_model
        self.tokenizer = tokenizer
        self.device1 = next(model_hi.parameters()).device
        self.device2 = next(model_en.parameters()).device
        self.gating_device = next(gating_model.parameters()).device
        self.pad_id = tokenizer.pad_token_id
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        input_ids_hi = inputs['input_ids_hi'].to(self.device1)
        attn_hi = inputs['attention_mask_hi'].to(self.device1)

        input_ids_en = inputs['input_ids_en'].to(self.device2)
        attn_en = inputs['attention_mask_en'].to(self.device2)

        labels = inputs['labels'].to(self.gating_device)

        # -------------------------
        # HARD SAFETY CHECK + CLAMP
        # -------------------------
        vocab_hi = self.model_hi.config.vocab_size
        vocab_en = self.model_en.config.vocab_size

        input_ids_hi = input_ids_hi.clamp(0, vocab_hi - 1)
        input_ids_en = input_ids_en.clamp(0, vocab_en - 1)

        # -------------------------
        # Forward Experts
        # -------------------------
        with torch.no_grad():
            out_hi = self.model_hi(
                input_ids=input_ids_hi,
                attention_mask=attn_hi,
                output_hidden_states=True,
                return_dict=True
            )

            out_en = self.model_en(
                input_ids=input_ids_en,
                attention_mask=attn_en,
                output_hidden_states=True,
                return_dict=True
            )

        # -------------------------
        # Last valid token
        # -------------------------
        seq_len_hi = (attn_hi.sum(dim=1).long() - 1).clamp(min=0)
        seq_len_en = (attn_en.sum(dim=1).long() - 1).clamp(min=0)

        batch_idx_hi = torch.arange(seq_len_hi.size(0), device=self.device1)
        batch_idx_en = torch.arange(seq_len_en.size(0), device=self.device2)

        h_hi = torch.cat([
            out_hi.hidden_states[-1][batch_idx_hi, seq_len_hi],
            out_hi.hidden_states[-2][batch_idx_hi, seq_len_hi],
            out_hi.hidden_states[-3][batch_idx_hi, seq_len_hi]
        ], dim=-1).to(self.gating_device)

        h_en = torch.cat([
            out_en.hidden_states[-1][batch_idx_en, seq_len_en],
            out_en.hidden_states[-2][batch_idx_en, seq_len_en],
            out_en.hidden_states[-3][batch_idx_en, seq_len_en]
        ], dim=-1).to(self.gating_device)

        # -------------------------
        # Gating
        # -------------------------
        g_values = self.gating_model(h_hi, h_en)
        g = g_values.unsqueeze(-1).unsqueeze(1)

        logits_hi = out_hi.logits.to(self.gating_device)
        logits_en = out_en.logits.to(self.gating_device)

        log_probs_hi = F.log_softmax(logits_hi, dim=-1)
        log_probs_en = F.log_softmax(logits_en, dim=-1)

        probs_hi = log_probs_hi.exp()
        probs_en = log_probs_en.exp()

        p_combined = g * probs_hi + (1 - g) * probs_en
        log_p = torch.log(p_combined + 1e-8)

        # -------------------------
        # CLM shift
        # -------------------------
        log_p = log_p[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        loss_fct = nn.NLLLoss(ignore_index=-100)

        loss = loss_fct(
            log_p.view(-1, log_p.size(-1)),
            shifted_labels.view(-1)
        )

        return (loss, log_p) if return_outputs else loss

# --- Main Execution ---

print(" Preparing data...")
prepare_triplet_jsonl()

print("\n Loading tokenizer and models...")

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.base_model,
    use_fast=True
)
# Gemma tokenizer requires a pad token for batching
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Load Base Model for PeftModel wrapper (MUST be CLM)
# Use the base model ID, but load the LoRA weights on top via PeftModel
# base_model_raw = AutoModelForCausalLM.from_pretrained(
#     args.base_model,
#     torch_dtype=torch.bfloat16,
#     # Load base model on CPU to prepare for PeftModel application, 
#     # as PeftModel will place the model on the specified device later.
#     device_map=None 
# )

# 3. Load PeftModels (Experts)
# The experts are placed on separate GPUs for parallelism and memory reasons (cuda:0 and cuda:1)
# model_hi = PeftModel.from_pretrained(base_model_raw, args.lora_hi).to('cuda:0')
# model_en = PeftModel.from_pretrained(base_model_raw, args.lora_en).to('cuda:1')
# model_hi.eval()
# model_en.eval()

# 4. Initialize Gating Network
# hidden_dim is the size of the final hidden state of the CLM (Gemma)
# hidden_dim = model_hi.config.hidden_size # Gemma uses hidden_size, not d_model
# Gating network is placed on cuda:1 with the English expert
# gating_model = GatingNetwork(hidden_dim=hidden_dim).to(torch.bfloat16).to('cuda:1')

# Load TWO independent base models

base_model_hi = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.bfloat16
).to("cuda:0")

base_model_en = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.bfloat16
).to("cuda:1")

# Apply LoRA separately

model_hi = PeftModel.from_pretrained(base_model_hi, args.lora_hi).to("cuda:0")
model_en = PeftModel.from_pretrained(base_model_en, args.lora_en).to("cuda:1")



hidden_dim = model_hi.config.hidden_size
gating_model = GatingNetwork(hidden_dim=hidden_dim).to(torch.bfloat16).to('cuda:1')
# 5. Freeze LoRA Adapters (only train the gating network)
# Note: LoRA layers in CLM models are usually in all linear layers, not just 'decoder.block.i'.
# We freeze all parameters *except* those in the gating_model.
# for name, param in model_hi.named_parameters():
#     param.requires_grad = False
# for name, param in model_en.named_parameters():
#     param.requires_grad = False


# Freeze everything by default
for name, param in model_hi.named_parameters():
    param.requires_grad = False
for name, param in model_en.named_parameters():
    param.requires_grad = False

# Unfreeze last 3 decoder layers of model_hi and model_en
def unfreeze_last_n_decoder_layers(model, n=3):
    for name, param in model.named_parameters():
        if any(f"model.decoder.layers.{i}." in name for i in range(model.config.num_hidden_layers - n, model.config.num_hidden_layers)):
            param.requires_grad = True

unfreeze_last_n_decoder_layers(model_hi)
unfreeze_last_n_decoder_layers(model_en)

# 6. Load Datasets
with open(args.train_json) as f:
    train_data = [json.loads(line) for line in f]
with open(args.val_json) as f:
    val_data = [json.loads(line) for line in f]

# We need to manually set a max_length here, which should be the same as the
# sequence length used during the LORA fine-tuning of the base models (e.g., 1024)
train_dataset = combinedGatingDataset(train_data, tokenizer, max_length=512)
val_dataset = combinedGatingDataset(val_data, tokenizer, max_length=512)

args_hf = TrainingArguments(
    output_dir="training_logs",
    per_device_train_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    learning_rate=3e-5,
    save_total_limit=2,
    logging_steps=10,
    load_best_model_at_end=False,
    greater_is_better=False,
    remove_unused_columns=False,
    fp16=False,
    bf16=True, # Use bfloat16 to match model dtype
)

trainer = CombinedGatingTrainer(
    # The 'model' argument for Trainer is the model whose weights are being updated: GatingNetwork
    model=gating_model, 
    model_hi=model_hi,
    model_en=model_en,
    gating_model=gating_model,
    tokenizer=tokenizer,
    args=args_hf,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print("\n Starting training...")
# Only the GatingNetwork weights will be updated.
trainer.train()

print("\n Saving model and tokenizer...")
torch.save(gating_model.state_dict(), args.output_gating_model)
tokenizer.save_pretrained(args.output_tokenizer_dir)
print(" Done!")
        



 
