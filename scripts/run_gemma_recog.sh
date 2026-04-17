#!/bin/bash
# ============================================================
# Example: RECOG with Gemma-3 / Qwen (decoder-only)
# Language pair: Hindi + English → Magahi
# Requires 2 GPUs (set via CUDA_VISIBLE_DEVICES)
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1

BASE_MODEL="google/gemma-7b"   # or Qwen/Qwen3-8B
LORA_MRL="/path/to/gemma-lora-hi-magahi"
LORA_HRL="/path/to/gemma-lora-en-magahi"
DATA_TRAIN="data/NLLB_seed_train.xlsx"
DATA_EVAL="data/flores_devtest.xlsx"
GATING_MODEL="gemma_gating_magahi.pt"
OUTPUT_DIR="outputs/gemma_magahi"

mkdir -p "$OUTPUT_DIR"

# ---- Step 1: Train Gating Network ----
echo "Training gating network (decoder-only)..."
python gemma/training/recog_train.py \
  --excel_path "$DATA_TRAIN" \
  --sheet_name deva-indian \
  --base_model "$BASE_MODEL" \
  --lora_hi "$LORA_MRL" \
  --lora_en "$LORA_HRL" \
  --src1_col Hindi \
  --src2_col English \
  --tgt_col Magahi \
  --src1_lang Hindi \
  --src2_lang English \
  --tgt_lang Magahi \
  --output_gating_model "$GATING_MODEL" \
  --output_tokenizer_dir "$OUTPUT_DIR/tokenizer"

# ---- Step 2: Run RECOG Inference ----
echo "Running RECOG inference (decoder-only)..."
python gemma/inference/recog_inference.py \
  --base_model "$BASE_MODEL" \
  --lora_hi "$LORA_MRL" \
  --lora_en "$LORA_HRL" \
  --gating_model_path "$GATING_MODEL" \
  --excel_path "$DATA_EVAL" \
  --sheet_name flores-devtest \
  --src1_lang Hindi \
  --src2_lang English \
  --tgt_lang Magahi \
  --output_file "$OUTPUT_DIR/recog_magahi_predictions.xlsx"

echo "Done! Results saved to $OUTPUT_DIR"
