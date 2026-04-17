import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import os
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--excel_path", type=str, required=True, help="Path to the Excel file containing test data.")
parser.add_argument("--sheet_name", type=str, required=True, help="Sheet name in the Excel file.")
parser.add_argument("--output_file", type=str, default="combined_gating_predictions_last3.xlsx", help="Output Excel file path.")
parser.add_argument("--gating_model_path", type=str, default="gating_model.pt", help="Path to the trained GatingNetwork weights.")
parser.add_argument("--base_model", type=str, default="google/gemma-7b", help="Base Gemma model ID.")
parser.add_argument("--lora_hi", type=str, required=True, help="Path to the Hindi LoRA adapter weights.")
parser.add_argument("--lora_en", type=str, required=True, help="Path to the English LoRA adapter weights.")
parser.add_argument("--src1_lang", type=str, default="Hindi", help="Source language 1 name.")
parser.add_argument("--src2_lang", type=str, default="English", help="Source language 2 name.")
parser.add_argument("--tgt_lang", type=str, default="Angika", help="Target language name.")
# We load the hardcoded constants from the original training script
parser.add_argument("--gating_hidden_dim", type=int, default=3072, help="Hidden dimension size for the GatingNetwork.")
args, unknown = parser.parse_known_args() 

class GatingNetwork(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        # Note: hidden_dim is the size of the final hidden state representation
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, h1, h2):
        x = torch.cat([h1, h2], dim=-1)
        g = self.ffn(x).squeeze(-1)
        g = torch.clamp(g, 1e-4, 1 - 1e-4)
        return g

@torch.no_grad()
def generate_combined_gating_clm(
    model_hi, model_en, tokenizer, gating_model,
    sentence_hi, sentence_en,
    max_new_tokens=128,
    device_hi='cuda:0', device_en='cuda:1', gating_device='cuda:1'
):
    # Eval prompt format matches Appendix A.2 (Listing 1) of the paper exactly.
    # No target text is appended; generation begins after "Respond with only the translation."
    eval_prompt_template = (
        "Translate the following {src_lang} text to {tgt_lang}.\n"
        "Input: {src_text}\n"
        "Respond with only the translation.\n"
    )
    prompt_hi = eval_prompt_template.format(src_lang=args.src1_lang, tgt_lang=args.tgt_lang, src_text=sentence_hi)
    prompt_en = eval_prompt_template.format(src_lang=args.src2_lang, tgt_lang=args.tgt_lang, src_text=sentence_en)

    enc_hi = tokenizer(prompt_hi, return_tensors='pt')
    for k in enc_hi:
        enc_hi[k] = enc_hi[k].to(device_hi, non_blocking=True)

    enc_en = tokenizer(prompt_en, return_tensors='pt')
    for k in enc_en:
        enc_en[k] = enc_en[k].to(device_en, non_blocking=True)

    input_len = enc_hi.input_ids.shape[-1]

    current_sequence_hi = enc_hi.input_ids
    attention_mask_hi = enc_hi.attention_mask
    current_sequence_en = enc_en.input_ids
    attention_mask_en = enc_en.attention_mask

    output_tokens = []
    g_values = []
    past_key_values_hi = None
    past_key_values_en = None
    
    
    def debug_check(name, tensor):
        if torch.isnan(tensor).any():
            print(f"[DEBUG] NaN detected in {name}")
        elif torch.isinf(tensor).any():
            print(f"[DEBUG] INF detected in {name}")
        else:
            # print(tensor)
            print(f"[DEBUG] {name} OK")
    
    for step in range(max_new_tokens):
         ########### NEW ADDED ############
        if step == 0:
            input_ids_hi = current_sequence_hi
            input_ids_en = current_sequence_en
        else:
            input_ids_hi = next_token_hi  # Only the newly generated token
            input_ids_en = next_token_en

        out_hi = model_hi(
            input_ids=input_ids_hi,
            attention_mask=attention_mask_hi,
            past_key_values=past_key_values_hi,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True
        )

        out_en = model_en(
            input_ids=input_ids_en,
            attention_mask=attention_mask_en,
            past_key_values=past_key_values_en,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True                                        
        )
            ############ NEW ADDED ############
        
        
        # out_hi = model_hi(
        #     input_ids=current_sequence_hi,
        #     attention_mask=attention_mask_hi,
        #     past_key_values=past_key_values_hi,
        #     use_cache=True,
        #     output_hidden_states=True,
        #     return_dict=True
        # )

        # out_en = model_en(
        #     input_ids=current_sequence_en,
        #     attention_mask=attention_mask_en,
        #     past_key_values=past_key_values_en,
        #     use_cache=True,
        #     output_hidden_states=True,
        #     return_dict=True
        # )

        # Concatenate last 3 layers' final token hidden states
        h_hi = torch.cat([layer[:, -1, :] for layer in out_hi.hidden_states[-3:]], dim=-1).to(gating_device)
        h_en = torch.cat([layer[:, -1, :] for layer in out_en.hidden_states[-3:]], dim=-1).to(gating_device)

        g = gating_model(h_hi, h_en).unsqueeze(-1)
        g_values.append(g.item())

        logits_hi = out_hi.logits[:, -1, :].to(torch.float32).to(gating_device)
        logits_en = out_en.logits[:, -1, :].to(torch.float32).to(gating_device)
        
        # logits_hi = torch.clamp(logits_hi, min=-30, max=30)
        # logits_en = torch.clamp(logits_en, min=-30, max=30)

        probs_hi = F.softmax(logits_hi, dim=-1)
        probs_en = F.softmax(logits_en, dim=-1)
        
        logits_hi = torch.nan_to_num(logits_hi, nan=0.0, posinf=50.0, neginf=-50.0)
        logits_en = torch.nan_to_num(logits_en, nan=0.0, posinf=50.0, neginf=-50.0)

        
        
        # debug_check("logits_hi", logits_hi)
        # debug_check("logits_en", logits_en)
        # debug_check("g", g)
        
                
        
        
        # log_probs_hi = F.log_softmax(logits_hi, dim=-1)
        # log_probs_en = F.log_softmax(logits_en, dim=-1)

        # # Combine in log-space then exponentiate
        # combined_log_probs = torch.logaddexp(g.log() + log_probs_hi, (1 - g).log() + log_probs_en)
        # combined_probs = torch.exp(combined_log_probs)
        
        
        # Step 3: Sanity check - logits must match vocab size
        assert logits_hi.shape[-1] == logits_en.shape[-1], (
            f"Shape mismatch: logits_hi {logits_hi.shape}, logits_en {logits_en.shape}"
        )
        vocab_size = logits_hi.shape[-1]  # or use tokenizer.vocab_size
        assert logits_en.shape[-1] == vocab_size, (
            f"logits_en shape is {logits_en.shape}, expected vocab size {vocab_size}"
        )

        # # Optional: Print for first few examples
        # if step < 5:
        #     print("[DEBUG] logits_hi OK", logits_hi)
        #     print("[DEBUG] logits_en OK", logits_en)
        
        # print("[DEBUG] g", g)
        # print("g min:", g.min().item(), "g max:", g.max().item())
        
        
        combined_logits = g * logits_hi + (1 - g) * logits_en
        
        # combined_logits = g * logits_hi + (1 - g) * logits_en
        # combined_probs = F.softmax(combined_logits, dim=-1)
        combined_logits = torch.nan_to_num(combined_logits, nan=0.0, posinf=50.0, neginf=-50.0)
        combined_probs = F.softmax(combined_logits, dim=-1)
        
        # debug_check("combined_logits", combined_logits)
        # debug_check("combined_probs", combined_probs)
        
        # combined_probs = g * probs_hi + (1 - g) * probs_en
        next_token_id = torch.argmax(combined_probs, dim=-1).item()
        output_tokens.append(next_token_id)

        next_token_hi = torch.tensor([[next_token_id]], device=device_hi)
        next_token_en = torch.tensor([[next_token_id]], device=device_en)

        current_sequence_hi = torch.cat([current_sequence_hi, next_token_hi], dim=1)
        attention_mask_hi = torch.cat([attention_mask_hi, torch.tensor([[1]], device=device_hi)], dim=1)

        current_sequence_en = torch.cat([current_sequence_en, next_token_en], dim=1)
        attention_mask_en = torch.cat([attention_mask_en, torch.tensor([[1]], device=device_en)], dim=1)

        past_key_values_hi = out_hi.past_key_values
        past_key_values_en = out_en.past_key_values

        if next_token_id == tokenizer.eos_token_id:
            break

    full_sequence = current_sequence_hi.squeeze().tolist()
    generated_ids = full_sequence[input_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True), g_values

def main():
    print(f"Loading resources for inference...")
    torch.manual_seed(42)
    # Load tokenizer
    # We use the base model path as the tokenizer was saved there by your training script
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine GPU devices
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("Error: Two CUDA devices (cuda:0 and cuda:1) are required for this model setup.")
        sys.exit(1)
        
    device_hi = torch.device('cuda:0')
    device_en = torch.device('cuda:1')
    gating_device = torch.device('cuda:1')

    # Load Base Model for PeftModel wrapper
    # base_model_raw = AutoModelForCausalLM.from_pretrained(
    #     args.base_model,
    #     torch_dtype=torch.bfloat16,
    #     device_map=None # Load base on CPU
    # )
    
    base_model_hi = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.bfloat16,
    device_map=None)

    base_model_en = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.bfloat16,
    device_map=None)

    model_hi = PeftModel.from_pretrained(base_model_hi, args.lora_hi).to(device_hi)
    model_en = PeftModel.from_pretrained(base_model_en, args.lora_en).to(device_en)

    # # Load LoRA Experts
    # print(f"Loading Hi Expert from {args.lora_hi} on {device_hi}...")
    # model_hi = PeftModel.from_pretrained(base_model_raw, args.lora_hi).to(device_hi)
    # print(f"Loading En Expert from {args.lora_en} on {device_en}...")
    # model_en = PeftModel.from_pretrained(base_model_raw, args.lora_en).to(device_en)

    model_hi.eval()
    model_en.eval()

    # Load trained gating model
    # We must ensure the correct hidden dimension is used
    if 'gemma' in args.base_model.lower():
        hidden_dim = model_hi.config.hidden_size
    else:
        # Fallback to the hardcoded value if model identification fails
        hidden_dim = args.gating_hidden_dim 
    
    # Ensure Gating Network DTYPE matches the model's output DTYPE
    gating_model = GatingNetwork(hidden_dim=hidden_dim).to(torch.bfloat16).to(gating_device)
    
    gating_model.load_state_dict(torch.load(args.gating_model_path))
    gating_model.eval()
    print(f"Gating Model loaded successfully.")

    # --- Load Excel file for inference ---
    print(f"Loading inference data from {args.excel_path}...")
    try:
        df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name)
        df.dropna(subset=[args.src1_lang, args.src2_lang], inplace=True)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        sys.exit(1)

    # Create a new column to store generated outputs
    generated_outputs = []
    g_values_all = []

    # Iterate through each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating {args.tgt_lang} Translations"):
        sentence_hi = str(row[args.src1_lang])
        sentence_en = str(row[args.src2_lang])

        try:
            translation, g_values = generate_combined_gating_clm(
                model_hi, model_en, tokenizer, gating_model,
                sentence_hi=sentence_hi,
                sentence_en=sentence_en,
                device_hi=device_hi,
                device_en=device_en,
                gating_device=gating_device
            )
        except Exception as e:
            print(f"Error at row {idx} ({sentence_hi} / {sentence_en}): {e}")
            translation = "ERROR"

        generated_outputs.append(translation)
        g_values_all.append(g_values)

    df[f"Generated_{args.tgt_lang}"] = generated_outputs

    with open("gating_values.json", "w") as f:
        json.dump(g_values_all, f, indent=2)

    df.to_excel(args.output_file, index=False)
    print("=" * 50)
    print(f"Inference complete. Output saved to '{args.output_file}'")
    print("=" * 50)
    
if __name__ == "__main__":
    main()
    
