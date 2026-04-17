import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModelForCausalLM
from tqdm import tqdm
import argparse
from sacrebleu.metrics import BLEU, CHRF

# --- Arguments ---
parser = argparse.ArgumentParser(description="Concat (MRL || HRL) to ELRL inference using Gemma")
parser.add_argument('--model_id', type=str, required=True)
parser.add_argument('--lora_id', type=str, required=True)
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--sheet_name', type=str, required=True)
parser.add_argument('--src_col1', type=str, required=True, help="Column name for source language 1 (MRL)")
parser.add_argument('--src_col2', type=str, required=True, help="Column name for source language 2 (HRL)")
parser.add_argument('--src_lang1', type=str, required=True, help="Language name for source 1 (MRL)")
parser.add_argument('--src_lang2', type=str, required=True, help="Language name for source 2 (HRL)")
parser.add_argument('--target_column', type=str, required=True)
parser.add_argument('--tgt_lang', type=str, required=True, help="Target language name")
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=4)

args = parser.parse_args()

# Inference prompt format matches Appendix A.2 (Listing 1) of the paper exactly.
# No target text is appended; generation begins after "Respond with only the translation."
EVAL_PROMPT_TEMPLATE = (
    "Translate the following {src_lang1} and {src_lang2} text to {tgt_lang}.\n"
    "Input: {src_text}\n"
    "Respond with only the translation.\n"
)

# --- Model Loading ---
print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(args.lora_id)
base_model = AutoModelForCausalLM.from_pretrained(
    args.model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

print("Merging LoRA adapter...")
model = PeftModelForCausalLM.from_pretrained(base_model, args.lora_id)
model = model.merge_and_unload()

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16
)

# --- Load Input File ---
print(f"Loading Excel: {args.input_file} (sheet: {args.sheet_name})")
df = pd.read_excel(args.input_file, sheet_name=args.sheet_name)
df.dropna(subset=[args.src_col1, args.src_col2, args.target_column], inplace=True)

# Create concat source: MRL || HRL
df["combined_input"] = df[args.src_col1].astype(str) + " || " + df[args.src_col2].astype(str)
src_texts = df["combined_input"].tolist()
num_samples = len(src_texts)

print(f"Samples loaded: {num_samples}")

# --- Batch Inference Function ---
def format_and_generate(texts):
    prompts = [
        EVAL_PROMPT_TEMPLATE.format(
            src_lang1=args.src_lang1,
            src_lang2=args.src_lang2,
            tgt_lang=args.tgt_lang,
            src_text=txt
        )
        for txt in texts
    ]

    outputs = generator(
        prompts,
        max_new_tokens=100,
        do_sample=False,
        temperature=0.1,
        return_full_text=False,
        batch_size=args.batch_size,
    )

    predictions = []
    for output in outputs:
        raw_translation = output[0]['generated_text'].strip()
        cleaned_translation = raw_translation.split('\n')[0].strip()
        predictions.append(cleaned_translation)
    return predictions

# --- Run Inference ---
all_predictions = []
for i in tqdm(range(0, num_samples, args.batch_size), desc="Translating..."):
    batch = src_texts[i:i + args.batch_size]
    batch_preds = format_and_generate(batch)
    all_predictions.extend(batch_preds)

# --- Save to Excel ---
df[f'Predicted_{args.target_column}'] = all_predictions
df.to_excel(args.output_file, index=False)
print(f"\nInference complete. Results saved to: {args.output_file}")

# --- Evaluation ---
ref = [str(r).strip() for r in df[args.target_column]]
sys_out = [str(s).strip() for s in df[f'Predicted_{args.target_column}']]
bleu = BLEU()
chrf = CHRF(word_order=2, beta=2)

print(f"Evaluation for target: {args.target_column}")
print(f"BLEU   : {bleu.corpus_score(sys_out, [ref]).score:.2f}")
print(f"ChrF++ : {chrf.corpus_score(sys_out, [ref]).score:.2f}")