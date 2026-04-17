#!/bin/bash
# ============================================================
# Example: RECOG with Aya-101 (encoder–decoder)
# Language pair: Hindi + English → Magahi
# ============================================================

BASE_MODEL="/path/to/aya-101"     # Local path or HuggingFace ID: CohereForAI/aya-101
LORA_MRL="/path/to/aya-peft-hi-magahi"
LORA_HRL="/path/to/aya-peft-en-magahi"
DATA_TRAIN="data/NLLB_seed_train.xlsx"
DATA_EVAL="data/flores_devtest.xlsx"
GATING_MODEL="gating_model_magahi.pt"
OUTPUT_DIR="outputs/aya101_magahi"

mkdir -p "$OUTPUT_DIR"

# ---- Step 1: Train Gating Network ----
echo "Training gating network..."
python aya101/training/recog_train.py \
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
echo "Running RECOG inference..."
python aya101/inference/recog_inference.py \
  --base_model "$BASE_MODEL" \
  --lora_hi "$LORA_MRL" \
  --lora_en "$LORA_HRL" \
  --gating_model_path "$GATING_MODEL" \
  --excel_path "$DATA_EVAL" \
  --sheet_name flores-devtest \
  --src1_lang Hindi \
  --src2_lang English \
  --tgt_lang Magahi \
  --output_excel "$OUTPUT_DIR/recog_magahi_predictions.xlsx"

echo "Done! Results saved to $OUTPUT_DIR"
