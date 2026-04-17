#!/bin/bash
# ============================================================
# Baselines: Concat, Heuristic, Fixed-0.5 Ensemble
# ============================================================

BASE_MODEL="/path/to/aya-101"
LORA_MRL="/path/to/aya-peft-hi-magahi"
LORA_HRL="/path/to/aya-peft-en-magahi"
DATA_EVAL="data/flores_devtest.xlsx"
OUTPUT_DIR="outputs/baselines"

mkdir -p "$OUTPUT_DIR"

# ---- 1. Heuristic (max-prob) baseline — Aya-101 ----
echo "Running max-prob heuristic baseline (Aya-101)..."
python baselines/heuristic/aya101_heuristic.py \
  --data_path "$DATA_EVAL" \
  --sheet flores-devtest \
  --output_path "$OUTPUT_DIR/heuristic_magahi.csv" \
  --source_col_hi Hindi \
  --source_col_en English \
  --target_col Magahi \
  --adapter_path_hi "$LORA_MRL" \
  --adapter_path_en "$LORA_HRL" \
  --device1 cuda:0 \
  --device2 cuda:1

# ---- 2. Fixed 0.5–0.5 ensemble baseline — Aya-101 ----
echo "Running fixed 0.5 ensemble baseline (Aya-101)..."
python baselines/fixed_ensemble/aya101_fixed05_inference.py \
  --base_model "$BASE_MODEL" \
  --lora_hi "$LORA_MRL" \
  --lora_en "$LORA_HRL" \
  --excel_path "$DATA_EVAL" \
  --sheet_name flores-devtest \
  --src1_lang Hindi \
  --src2_lang English \
  --tgt_lang Magahi \
  --output_excel "$OUTPUT_DIR/fixed05_magahi.xlsx"

# ---- 3. Concat training — Aya-101 ----
echo "Running concat training baseline (Aya-101)..."
python baselines/concat/aya101_concat_train.py \
  --source_language_1 English \
  --source_language_2 Hindi \
  --target_language Magahi \
  --output_dir aya_concat_magahi_lora \
  --peft_model_id aya_concat_magahi_peft \
  --save_model_path aya_concat_magahi_saved \
  --train_sheet_name deva-indian

# ---- 4. Concat inference — Aya-101 ----
echo "Running concat inference (Aya-101)..."
python baselines/concat/aya101_concat_inference.py \
  --input_file "$DATA_EVAL" \
  --output_csv "$OUTPUT_DIR/concat_magahi.csv" \
  --lora_adapter aya_concat_magahi_peft \
  --src_col1 English \
  --src_col2 Hindi \
  --output_column Magahi \
  --src_lang1 English \
  --src_lang2 Hindi \
  --tgt_lang Magahi \
  --gpu 0

# ---- 5. Heuristic baseline — Gemma ----
echo "Running max-prob heuristic baseline (Gemma)..."
CUDA_VISIBLE_DEVICES=0,1 python baselines/heuristic/gemma_heuristic.py \
  --data_path "$DATA_EVAL" \
  --sheet flores-devtest \
  --output_path "$OUTPUT_DIR/gemma_heuristic_magahi.csv" \
  --source_col_hi Hindi \
  --source_col_en English \
  --target_col Magahi \
  --adapter_path_hi /path/to/gemma-lora-hi-magahi \
  --adapter_path_en /path/to/gemma-lora-en-magahi \
  --device_hi cuda:0 \
  --device_en cuda:1

# ---- 6. Concat training — Gemma ----
echo "Running concat training baseline (Gemma)..."
CUDA_VISIBLE_DEVICES=0 python baselines/concat/gemma_concat_train.py \
  --model_id google/gemma-7b \
  --input_file data/NLLB_seed_train.xlsx \
  --sheet_name deva-indian \
  --src_col1 Hindi \
  --src_col2 English \
  --src_lang1 Hindi \
  --src_lang2 English \
  --target_column Magahi \
  --tgt_lang Magahi \
  --output_dir gemma_concat_magahi_lora

# ---- 7. Concat inference — Gemma ----
echo "Running concat inference (Gemma)..."
CUDA_VISIBLE_DEVICES=0 python baselines/concat/gemma_concat_inference.py \
  --model_id google/gemma-7b \
  --lora_id gemma_concat_magahi_lora \
  --input_file "$DATA_EVAL" \
  --sheet_name flores-devtest \
  --src_col1 Hindi \
  --src_col2 English \
  --src_lang1 Hindi \
  --src_lang2 English \
  --target_column Magahi \
  --tgt_lang Magahi \
  --output_file "$OUTPUT_DIR/gemma_concat_magahi.xlsx"

echo "All baselines done! Results in $OUTPUT_DIR"
