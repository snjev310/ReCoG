import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =====================================================
# Reproducibility
# =====================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =====================================================
# Arguments
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)

parser.add_argument("--source_col_hi", type=str, default="Hindi")
parser.add_argument("--source_col_en", type=str, default="English")
parser.add_argument("--target_col", type=str, default="Magahi")
parser.add_argument("--sheet", type=str, default=None)

parser.add_argument("--adapter_path_hi", type=str, required=True)
parser.add_argument("--adapter_path_en", type=str, required=True)

parser.add_argument("--device_hi", type=str, default="cuda:0")
parser.add_argument("--device_en", type=str, default="cuda:1")

parser.add_argument("--max_new_tokens", type=int, default=128)

args = parser.parse_args()

# =====================================================
# Load Gemma + tokenizer
# =====================================================
MODEL_NAME = "google/gemma-7b"   # or gemma-2b / gemma-7b-it

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token   # REQUIRED for Gemma

base_hi = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
).to(args.device_hi)

base_en = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
).to(args.device_en)

model_hi = PeftModel.from_pretrained(base_hi, args.adapter_path_hi).to(args.device_hi)
model_en = PeftModel.from_pretrained(base_en, args.adapter_path_en).to(args.device_en)

model_hi.eval()
model_en.eval()

# =====================================================
# Cooperative decoding (PROMPT-ALIGNED)
# =====================================================
@torch.no_grad()
def generate_combined_maxprob(
    sentence_hi,
    sentence_en,
):
    # Inference prompt format matches Appendix A.2 (Listing 1) of the paper exactly.
    # No target text is appended; generation begins after "Respond with only the translation."
    eval_prompt_template = (
        "Translate the following {src_lang} text to {tgt_lang}.\n"
        "Input: {src_text}\n"
        "Respond with only the translation.\n"
    )

    prompt_hi = eval_prompt_template.format(
        src_lang=args.source_col_hi,
        tgt_lang=args.target_col,
        src_text=sentence_hi
    )

    prompt_en = eval_prompt_template.format(
        src_lang=args.source_col_en,
        tgt_lang=args.target_col,
        src_text=sentence_en
    )

    # --------------------------------------------------
    # Tokenize prompts
    # --------------------------------------------------
    hi_inputs = tokenizer(prompt_hi, return_tensors="pt").to(args.device_hi)
    en_inputs = tokenizer(prompt_en, return_tensors="pt").to(args.device_en)

    # --------------------------------------------------
    # Shared decoding history (one per device)
    # --------------------------------------------------
    tokens_hi = hi_inputs.input_ids.clone()
    tokens_en = en_inputs.input_ids.clone()

    past_hi = None
    past_en = None

    eos_id = tokenizer.eos_token_id

    # --------------------------------------------------
    # Autoregressive cooperative decoding
    # --------------------------------------------------
    for _ in range(args.max_new_tokens):

        # ---- Hindi expert ----
        out_hi = model_hi(
            input_ids=tokens_hi[:, -1:] if past_hi else tokens_hi,
            past_key_values=past_hi,
            use_cache=True,
            return_dict=True,
        )
        logits_hi = out_hi.logits[:, -1, :]
        past_hi = out_hi.past_key_values

        # ---- English expert ----
        out_en = model_en(
            input_ids=tokens_en[:, -1:] if past_en else tokens_en,
            past_key_values=past_en,
            use_cache=True,
            return_dict=True,
        )
        logits_en = out_en.logits[:, -1, :]
        past_en = out_en.past_key_values

        # ---- Convert to probabilities ----
        probs_hi = F.softmax(logits_hi, dim=-1)
        probs_en = F.softmax(logits_en, dim=-1).to(args.device_hi)

        # ---- MAX-prob cooperative fusion ----
        fused_probs = torch.maximum(probs_hi, probs_en)

        next_token = torch.argmax(fused_probs, dim=-1).item()

        # ---- Append token to BOTH histories ----
        next_hi = torch.tensor([[next_token]], device=args.device_hi)
        next_en = torch.tensor([[next_token]], device=args.device_en)

        tokens_hi = torch.cat([tokens_hi, next_hi], dim=-1)
        tokens_en = torch.cat([tokens_en, next_en], dim=-1)

        if next_token == eos_id:
            break

    # --------------------------------------------------
    # Decode ONLY the generated response
    # --------------------------------------------------
    gen_tokens = tokens_hi[0][hi_inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)

# =====================================================
# Load data
# =====================================================
if args.sheet:
    df = pd.read_excel(args.data_path, sheet_name=args.sheet)
else:
    df = pd.read_excel(args.data_path)

# =====================================================
# Run decoding
# =====================================================
results = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Gemma Cooperative Decoding"):
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

# =====================================================
# Save output
# =====================================================
out_df = pd.DataFrame(results)
out_df.to_csv(args.output_path, index=False)

print(f"\nResults saved to {args.output_path}")


# CUDA_VISIBLE_DEVICES=0,3 python gemma3_heuristic.py \
#   --data_path NLLB_data_for_gating.xlsx \
#   --sheet flores-devtest \
#   --output_path gemma_combined_heuristic_Friulian.csv \
#   --source_col_hi Italian \
#   --source_col_en English \
#   --target_col Friulian \
#   --adapter_path_hi gemma-7b-lora-it_Friulian/ \
#   --adapter_path_en gemma-7b-lora-en_Friulian/ \
#   --device_hi cuda:0 \
#   --device_en cuda:1 \
#   --max_new_tokens 128