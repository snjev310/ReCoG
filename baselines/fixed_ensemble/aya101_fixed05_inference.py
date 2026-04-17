import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import json
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from peft import PeftModel
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm


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


@torch.no_grad()
def generate_combined_gating(model_hi, model_en, tokenizer,
                              sentence_hi, sentence_en,
                              prompt_hi, prompt_en,
                              max_length=128,
                              device_hi='cuda:0', device_en='cuda:1',
                              start_token_id=None):
    enc_hi = tokenizer(prompt_hi + sentence_hi, return_tensors='pt').to(device_hi)
    enc_en = tokenizer(prompt_en + sentence_en, return_tensors='pt').to(device_en)

    encoder_outputs_hi = model_hi.get_encoder()(enc_hi.input_ids)
    encoder_outputs_en = model_en.get_encoder()(enc_en.input_ids)

    if start_token_id is None:
        start_token_id = model_hi.config.decoder_start_token_id or tokenizer.pad_token_id

    decoder_input_ids = torch.tensor([[start_token_id]]).to(device_hi)
    output_tokens = []

    for step in range(max_length):
        dec_in_hi = decoder_input_ids.to(device_hi)
        dec_in_en = decoder_input_ids.to(device_en)

        out_hi = model_hi(encoder_outputs=encoder_outputs_hi, decoder_input_ids=dec_in_hi,
                          return_dict=True)
        out_en = model_en(encoder_outputs=encoder_outputs_en, decoder_input_ids=dec_in_en,
                          return_dict=True)

        logits_hi = out_hi.logits[:, -1, :]
        logits_en = out_en.logits[:, -1, :]
        probs_hi = F.softmax(logits_hi, dim=-1)
        probs_en = F.softmax(logits_en, dim=-1)

        # g = gating_model(hid_hi.to(gating_device), hid_en.to(gating_device))  # scalar
        # g = 0.5  # For equal weighting, you can set g to 0.5. 
        # combined_probs = g * probs_hi.to(gating_device) + (1 - g) * probs_en.to(gating_device)
        combined_probs = 0.5 * probs_hi + 0.5 * probs_en.to(device_hi)
        next_token_id = torch.argmax(combined_probs, dim=-1).item()
        output_tokens.append(next_token_id)

        decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([[next_token_id]]).to(device_hi)], dim=1)
        if next_token_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(output_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_hi", type=str, required=True)
    parser.add_argument("--lora_en", type=str, required=True)
    # parser.add_argument("--gating_model_path", type=str, required=True)
    parser.add_argument("--excel_path", type=str, required=True)
    parser.add_argument("--sheet_name", type=str, required=True)
    parser.add_argument("--src1_lang", type=str, default="Hindi")
    parser.add_argument("--src2_lang", type=str, default="English")
    parser.add_argument("--tgt_lang", type=str, default="Angika")
    parser.add_argument("--output_excel", type=str, default="combined_gating_predictions.xlsx")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model_hi = PeftModel.from_pretrained(
        AutoModelForSeq2SeqLM.from_pretrained(args.base_model), args.lora_hi).to("cuda:0").eval()
    model_en = PeftModel.from_pretrained(
        AutoModelForSeq2SeqLM.from_pretrained(args.base_model), args.lora_en).to("cuda:1").eval()

    model_hi.eval()
    model_en.eval()
    
    # gating_model = GatingNetwork(hidden_dim=12288).to("cuda:1")
    # gating_model.load_state_dict(torch.load(args.gating_model_path))
    # gating_model.eval()

    df = pd.read_excel(args.excel_path,args.sheet_name)
    # df = df[0:5]
    output_col = []

    prompt_hi = f"translate {args.src1_lang} to {args.tgt_lang}: "
    prompt_en = f"translate {args.src2_lang} to {args.tgt_lang}: "
    
    output = generate_combined_gating(
    model_hi, model_en, tokenizer,
    sentence_hi="आपका नाम क्या है?",
    sentence_en="What is your name?",
    prompt_hi=prompt_hi,
    prompt_en=prompt_en
)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        try:
            sentence_hi = row[args.src1_lang]
            sentence_en = row[args.src2_lang]

            output = generate_combined_gating(
                model_hi, model_en, tokenizer,
                sentence_hi, sentence_en, prompt_hi, prompt_en
            )
        except Exception as e:
            print(f"Error at row {idx}: {e}")
            output = "ERROR"
        output_col.append(output)

    df[f"Generated {args.tgt_lang}"] = output_col
    df.to_excel(args.output_excel, index=False)
    print(f"✅ Inference completed. Saved to: {args.output_excel}")


if __name__ == "__main__":
    main()