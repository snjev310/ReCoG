import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    TopKLogitsWarper,
)
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Concat Source (2-column) LoRA Inference")

    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--sheet", type=str, default="Sheet1")
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--lora_adapter", type=str, required=True)

    parser.add_argument("--src_col1", type=str, required=True)
    parser.add_argument("--src_col2", type=str, required=True)

    parser.add_argument("--output_column", type=str, required=True)
    parser.add_argument("--src_lang1", type=str, required=True)
    parser.add_argument("--src_lang2", type=str, required=True)
    parser.add_argument("--tgt_lang", type=str, required=True)

    return parser.parse_args()


def manual_generate_with_past(
    model,
    tokenizer,
    input_text,
    device,
    max_length=128,
    min_length=5,
    temperature=1.0,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.0,
    no_repeat_ngram_size=3,
):
    model.eval()

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        encoder_outputs = model.get_encoder()(
            input_ids=input_ids, attention_mask=attention_mask
        )

    decoder_input_ids = torch.tensor(
        [[model.config.decoder_start_token_id]], device=device
    )
    generated = decoder_input_ids
    eos_token_id = model.config.eos_token_id

    logits_processors = LogitsProcessorList()
    if min_length > 0 and eos_token_id is not None:
        logits_processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
    if repetition_penalty != 1.0:
        logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if no_repeat_ngram_size > 0:
        logits_processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if temperature != 1.0:
        logits_processors.append(TemperatureLogitsWarper(temperature))
    if top_p < 1.0:
        logits_processors.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        logits_processors.append(TopKLogitsWarper(top_k))

    past_key_values = None

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=generated[:, -1:] if past_key_values else generated,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
            )

        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)

        generated = torch.cat([generated, next_token], dim=-1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    model_path = (
        "/home/speech-nlp-cse/sanjeev8.kumar/.cache/huggingface/hub/models--CohereForAI--aya-101/snapshots/231cff3a9729ccdaee18839b32aaabac5278a21c"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    model = PeftModel.from_pretrained(
        base_model,
        args.lora_adapter,
        adapter_name_or_path=f"{args.lora_adapter}/adapter_model.safetensors",
    ).to(device)

    df = pd.read_excel(args.input_file)#, sheet_name=args.sheet)

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Decoding"):
        src1 = str(row[args.src_col1]).strip()
        src2 = str(row[args.src_col2]).strip()
        gold = row.get(args.output_column, "")

        concat_src = f"{src1} ||| {src2}"
        prompt = (
            f"Translate {args.src_lang1} and {args.src_lang2} "
            f"to {args.tgt_lang}: {concat_src}"
        )

        try:
            generated = manual_generate_with_past(
                model, tokenizer, prompt, device
            )
        except Exception as e:
            print(f"[ERROR] Row {idx}: {e}")
            generated = ""

        results.append(
            {
                args.src_col1: src1,
                args.src_col2: src2,
                args.output_column: gold,
                "Generated": generated,
            }
        )

    pd.DataFrame(results).to_csv(args.output_csv, index=False)
    print(f"\n✅ Saved results to {args.output_csv}")


if __name__ == "__main__":
    main()
