import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)

parser.add_argument("--source_col_hi", type=str, default="Hindi")
parser.add_argument("--source_col_en", type=str, default="English")
parser.add_argument("--target_col", type=str, default="Angika")
parser.add_argument("--sheet", type=str, default=None)

parser.add_argument("--adapter_path_hi", type=str, required=True)
parser.add_argument("--adapter_path_en", type=str, required=True)

parser.add_argument("--device1", type=str, default="cuda:0")
parser.add_argument("--device2", type=str, default="cuda:1")

parser.add_argument("--max_len", type=int, default=128)

args = parser.parse_args()

MODEL_NAME = "/home/IITB/cfilt/sanjeev8.kumar/.cache/huggingface/hub/models--CohereLabs--aya-101/snapshots/e7dad472b9de8a30a00cf08a05c19003bf59028d"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_m1 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(args.device1)
base_m2 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(args.device2)

model1 = PeftModel.from_pretrained(base_m1, args.adapter_path_hi).to(args.device1)
model2 = PeftModel.from_pretrained(base_m2, args.adapter_path_en).to(args.device2)

model1.eval()
model2.eval()

@torch.no_grad()
def get_next_logits(
    model,
    encoder_outputs,
    attention_mask,
    decoder_input_ids,
    past_key_values=None,
):
    outputs = model(
        encoder_outputs=encoder_outputs,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids[:, -1:] if past_key_values else decoder_input_ids,
        use_cache=True,
        past_key_values=past_key_values,
        return_dict=True,
    )
    logits = outputs.logits[:, -1, :]  # (1, vocab)
    return logits, outputs.past_key_values

# =========================
# Cooperative MAX-prob decoding
# =========================
@torch.no_grad()
def generate_combined_maxprob(
    sentence_hi,
    sentence_en,
):
    # ---- Encode Hindi (M1) ----
    enc_hi = tokenizer(
        f"translate Hindi to {args.target_col}: {sentence_hi}",
        return_tensors="pt"
    ).to(args.device1)

    # ---- Encode English (M2) ----
    enc_en = tokenizer(
        f"translate English to {args.target_col}: {sentence_en}",
        return_tensors="pt"
    ).to(args.device2)

    enc_out_m1 = model1.get_encoder()(**enc_hi)
    enc_out_m2 = model2.get_encoder()(**enc_en)

    # ---- Decoder init ----
    start_id = model1.config.decoder_start_token_id or tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    combined_tokens = torch.tensor([[start_id]], device=args.device1)

    past_m1 = None
    past_m2 = None

    # ---- Decoding loop ----
    for _ in range(args.max_len):

        logits_m1, past_m1 = get_next_logits(
            model1,
            enc_out_m1,
            enc_hi.attention_mask,
            combined_tokens,
            past_m1
        )

        logits_m2, past_m2 = get_next_logits(
            model2,
            enc_out_m2,
            enc_en.attention_mask,
            combined_tokens.to(args.device2),
            past_m2
        )

        probs_m1 = F.softmax(logits_m1, dim=-1)
        probs_m2 = F.softmax(logits_m2, dim=-1).to(args.device1)

        # ---- MAX fusion ----
        fused_probs = torch.maximum(probs_m1, probs_m2)

        next_token = torch.argmax(fused_probs, dim=-1).item()

        combined_tokens = torch.cat(
            [combined_tokens,
             torch.tensor([[next_token]], device=args.device1)],
            dim=-1
        )

        if next_token == eos_id:
            break

    return tokenizer.decode(combined_tokens[0], skip_special_tokens=True)

# =========================
# Load data
# =========================
if args.sheet:
    df = pd.read_excel(args.data_path, sheet_name=args.sheet)
else:
    df = pd.read_excel(args.data_path)
    df = df[0:5]

# =========================
# Run decoding
# =========================
results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Combined Cooperative Decoding"):
    hi = str(row[args.source_col_hi])
    en = str(row[args.source_col_en])
    tgt = row.get(args.target_col, "")

    try:
        combined_out = generate_combined_maxprob(hi, en)
    except Exception as e:
        print(f"[ERROR] Row {idx}: {e}")
        combined_out = ""

    results.append({
        args.source_col_hi: hi,
        args.source_col_en: en,
        f"Original_{args.target_col}": tgt,
        "Combined_Generated": combined_out
    })

# =========================
# Save results
out_df = pd.DataFrame(results)
out_df.to_csv(args.output_path, index=False)

print(f"\n✅ Results saved to: {args.output_path}")


# python combined_heuristic_new.py \
#   --data_path ../MT-2/Angika_retranslate_1000_6_10.xlsx \
#   --output_path combined_heuristic_ang.csv \
#   --source_col_hi Hindi \
#   --source_col_en English \
#   --target_col Angika \
#   --adapter_path_hi ../MT-2/aya-peft-hi-ang-27-8/ \
#   --adapter_path_en ../MT-2/aya-peft-en-ang-27-8/ \
#   --device1 cuda:0 \
#   --device2 cuda:1