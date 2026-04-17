import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import argparse
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from peft import PeftModel
from sklearn.model_selection import train_test_split
from transformers.trainer_utils import EvalLoopOutput

parser = argparse.ArgumentParser()
parser.add_argument("--excel_path", type=str, required=True)
parser.add_argument("--sheet_name", type=str, required=True)
parser.add_argument("--train_json", type=str, default="train.jsonl")
parser.add_argument("--val_json", type=str, default="val.jsonl")
parser.add_argument("--output_gating_model", type=str, default="gating_model.pt")
parser.add_argument("--output_tokenizer_dir", type=str, default="tokenizer")
parser.add_argument("--src1_col", type=str, default="Hindi")
parser.add_argument("--src2_col", type=str, default="English")
parser.add_argument("--tgt_col", type=str, default="Angika")
parser.add_argument("--src1_lang", type=str, default="Hindi")
parser.add_argument("--src2_lang", type=str, default="English")
parser.add_argument("--tgt_lang", type=str, default="Angika")
parser.add_argument("--base_model", type=str, required=True)
parser.add_argument("--lora_hi", type=str, required=True)
parser.add_argument("--lora_en", type=str, required=True)
args = parser.parse_args()

def prepare_triplet_jsonl():
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
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        inputs_hi = self.tokenizer(
            f"translate {args.src1_lang} to {args.tgt_lang}: {ex[args.src1_col]}",
            padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        inputs_en = self.tokenizer(
            f"translate {args.src2_lang} to {args.tgt_lang}: {ex[args.src2_col]}",
            padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        target = self.tokenizer(
            ex[args.tgt_col],
            padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            'input_ids_hi': inputs_hi.input_ids.squeeze(),
            'attention_mask_hi': inputs_hi.attention_mask.squeeze(),
            'input_ids_en': inputs_en.input_ids.squeeze(),
            'attention_mask_en': inputs_en.attention_mask.squeeze(),
            'labels': target.input_ids.squeeze(),
            'decoder_attention_mask': target.attention_mask.squeeze()
        }

class GatingNetwork(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
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

        labels_hi = inputs['labels'].to(self.device1)
        labels_en = inputs['labels'].to(self.device2)
        decoder_attn_hi = inputs['decoder_attention_mask'].to(self.device1)
        decoder_attn_en = inputs['decoder_attention_mask'].to(self.device2)

        with torch.no_grad():
            enc_hi = self.model_hi.get_encoder()(input_ids=input_ids_hi, attention_mask=attn_hi)
            enc_en = self.model_en.get_encoder()(input_ids=input_ids_en, attention_mask=attn_en)

        out_hi = self.model_hi(input_ids=input_ids_hi, attention_mask=attn_hi,
                               decoder_input_ids=labels_hi, decoder_attention_mask=decoder_attn_hi,
                               encoder_outputs=enc_hi, output_hidden_states=True, return_dict=True)

        out_en = self.model_en(input_ids=input_ids_en, attention_mask=attn_en,
                               decoder_input_ids=labels_en, decoder_attention_mask=decoder_attn_en,
                               encoder_outputs=enc_en, output_hidden_states=True, return_dict=True)

        h_hi = torch.cat(out_hi.decoder_hidden_states[-3:], dim=-1).to(self.gating_device)
        h_en = torch.cat(out_en.decoder_hidden_states[-3:], dim=-1).to(self.gating_device)
        g = self.gating_model(h_hi, h_en).unsqueeze(-1)

        probs_hi = F.softmax(out_hi.logits.to(self.gating_device), dim=-1)
        probs_en = F.softmax(out_en.logits.to(self.gating_device), dim=-1)
        p_combined = g * probs_hi + (1 - g) * probs_en

        log_p = torch.log(p_combined + 1e-8)
        loss_fct = nn.NLLLoss(ignore_index=self.pad_id)
        loss = loss_fct(log_p.view(-1, log_p.size(-1)), labels_hi.to(self.gating_device).view(-1))
        return (loss, log_p) if return_outputs else loss

    def prediction_step(self, *args, **kwargs):
        print("Skipping prediction_step for GatingNetwork.")
        return (None, None, None)

    def evaluation_loop(self, *args, **kwargs):
        print("Skipping HuggingFace evaluation loop for GatingNetwork.")
        return EvalLoopOutput(predictions=None, label_ids=None, metrics={"eval_loss": 0.0}, num_samples=0)

def freeze_all_except_last_3_lora(model):
    for name, param in model.named_parameters():
        param.requires_grad = False
        for i in [-1, -2, -3]:
            if f"decoder.block.{i}" in name and "lora" in name:
                param.requires_grad = True

print(" Preparing data...")
prepare_triplet_jsonl()

print("\n Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
model_hi = PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained(args.base_model), args.lora_hi).cuda(0)
model_en = PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained(args.base_model), args.lora_en).cuda(1)
gating_model = GatingNetwork(hidden_dim=3 * model_hi.base_model.config.d_model).cuda(1)

freeze_all_except_last_3_lora(model_hi)
freeze_all_except_last_3_lora(model_en)

with open(args.train_json) as f:
    train_data = [json.loads(line) for line in f]
with open(args.val_json) as f:
    val_data = [json.loads(line) for line in f]

train_dataset = combinedGatingDataset(train_data, tokenizer)
val_dataset = combinedGatingDataset(val_data, tokenizer)

args_hf = TrainingArguments(
    output_dir="training_logs",
    per_device_train_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    learning_rate=3e-5,
    save_total_limit=2,
    logging_steps=10,
    load_best_model_at_end=False,
    greater_is_better=False,
    remove_unused_columns=False
)

trainer = CombinedGatingTrainer(
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
trainer.train()

print("\n Saving model and tokenizer...")
torch.save(gating_model.state_dict(), args.output_gating_model)
tokenizer.save_pretrained(args.output_tokenizer_dir)
print(" Done!")
