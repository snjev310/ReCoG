# Data

## Datasets Used

### Training
- **NLLB Seed Corpus**: ~6,192 sentence pairs per language direction.  
  Download: https://github.com/facebookresearch/flores (see NLLB Seed data)

### Evaluation
- **FLORES-200 devtest**: 1,012 samples per language.  
  Download: https://github.com/facebookresearch/flores

### Angika
- Dataset from Kumar et al., EACL 2026 (SrcMix paper).

---

## Expected Excel Format

Your training/evaluation Excel file should contain one row per sentence with the following columns (column names are configurable via CLI arguments):

| Hindi | English | Magahi |
|---|---|---|
| राम बाजार गया। | Ram went to the market. | राम बजार गेल। |
| ... | ... | ... |

- **Column for MRL source** (e.g., `Hindi`, `Bengali`, `Italian`, `French`)
- **Column for HRL source** (always `English` in our experiments)
- **Column for ELRL target** (e.g., `Magahi`, `Angika`, `Meitei`, `Friulian`)

The sheet name is passed via `--sheet_name` (e.g., `deva-indian` for training, `flores-devtest` for evaluation).

---

## Synthetic MRL Generation

To generate synthetic MRL from HRL (English) when human MRL–ELRL pairs are unavailable:

**Indo-Aryan languages (Hindi as MRL):**
```bash
# Using IndicTrans2
python -c "
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# Load IndicTrans2 en-indic model
# Translate English column → Hindi column
"
```

**Other language families:**
```bash
# Using Aya-101 with prompt: 'translate English to Hindi: <sentence>'
```

See §2.2 and §4 of the paper for details on synthetic MRL generation and its impact on gating behavior.

---

## Notes

- Rows with missing values in any required column are automatically dropped.
- The scripts split training data into train/val (90/10) internally using `sklearn.model_selection.train_test_split` with `random_state=42`.
- All seeds are fixed to 42 for reproducibility.
