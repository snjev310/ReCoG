import pandas as pd
import random
seed = 42
random.seed(42)
import os
os.environ["WANDB_DISABLED"] = "true"
import torch
import transformers 
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM,Trainer, TrainingArguments
# import wandb
# wandb.login(key='9900563417be7e1b20caf399fe25fe5f8b1d4643')
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
import warnings
warnings.filterwarnings('ignore')
import argparse
from sklearn.model_selection import train_test_split


#Initialize the parser
parser = argparse.ArgumentParser(description="Process multiple variables for a machine translation task.")

# Add arguments
parser.add_argument("--source_language_1", type=str, required=True, help="Specify the source language")
parser.add_argument("--source_language_2", type=str, required=True, help="Specify the source language")
parser.add_argument("--target_language", type=str, required=True, help="Specify the target language")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output results")
parser.add_argument("--peft_model_id", type=str, required=True, help="ID of the PEFT model to use")
parser.add_argument("--save_model_path", type=str, required=True, help="Path to save the fine-tuned model")
parser.add_argument("--train_sheet_name", type=str, required=True, help="Path to the input data file")

# Parse the arguments
args = parser.parse_args()


# python concat_mt_training.py --source_language_1 'English' --source_language_2 'Hindi' --target_language 'Magahi'  --output_dir "aya_en_hi_concat_mag_16_12" --peft_model_id "aya_peft_en_hi_concat_mag_16_12" --save_model_path 'aya-aya_en_hi_concat_mag_16_12' --train_sheet_name 'deva-indian'

# Access the values
source_language_1 = args.source_language_1
source_language_2 = args.source_language_2
target_language = args.target_language
# data_path = args.data_path
# model_name = args.model_name
model_name = "/home/speech-nlp-cse/sanjeev8.kumar/.cache/huggingface/hub/models--CohereForAI--aya-101/snapshots/231cff3a9729ccdaee18839b32aaabac5278a21c"
output_dir = args.output_dir
peft_model_id = args.peft_model_id
save_model_path = args.save_model_path
train_sheet_name = args.train_sheet_name


train_data = pd.read_excel('../MT-2/NLLB_seed_new_data_train.xlsx',sheet_name=train_sheet_name)

# train_data = pd.read_excel('../MT-2/NLLB_data_for_gating.xlsx',sheet_name=train_sheet_name)

train_data = train_data.dropna(subset=[source_language_1, source_language_2, target_language])  # Drop rows with NaN in source or target language columns

train_data['src'] = train_data[source_language_1].str.strip() + ' ||| ' + train_data[source_language_2].str.strip()
train_data['tgt'] = train_data[target_language].str.strip()

# train_data = train_data.dropna(subset=[source_language, target_language])  # Drop rows with NaN in source or target language columns
# test_data = pd.read_excel('NLLB_seed_new_data_train.xlsx',sheet_name='flores-devtest')


train_data, dev_data = train_test_split(train_data[['src', 'tgt']], test_size=0.02, random_state=42)

train_data = Dataset.from_pandas(train_data)
dev_data = Dataset.from_pandas(dev_data)

print("Test data: ",len(dev_data))
print("Train data: ",len(train_data))

model_id = model_name
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

from datasets import concatenate_datasets
import numpy as np
# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([train_data, dev_data]).map(lambda x: tokenizer(x['src'], truncation=True), batched=True, remove_columns=['src', 'tgt'])
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# take 85 percentile of max length for better utilization
max_source_length = int(np.percentile(input_lenghts, 50))
print(f"Max source length: {max_source_length}")
 
# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([train_data, dev_data]).map(lambda x: tokenizer(x['tgt'], truncation=True), batched=True, remove_columns=['src', 'tgt'])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# take 90 percentile of max length for better utilization
max_target_length = int(np.percentile(target_lenghts, 90))
print(f"Max target length: {max_target_length}")

def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = [f"Translate {source_language_1} and {source_language_2} to {target_language}: " + item for item in sample['src']]
    # print('inputs: ',inputs)
    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample['tgt'], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

tokenized_dataset_train = train_data.map(preprocess_function, batched=True, remove_columns=['src','tgt'])
tokenized_dataset_dev = dev_data.map(preprocess_function, batched=True, remove_columns=['src','tgt'])

print(f"Keys of tokenized train dataset: {list(tokenized_dataset_train.features)}")
print(f"Keys of tokenized test dataset: {list(tokenized_dataset_dev.features)}")

# tokenized_dataset_train.save_to_disk(f"{data_path}/train")
# tokenized_dataset_dev.save_to_disk(f"{data_path}/test")

# Define LoRA Config
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True)
model.config.use_cache = False

lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM,
 inference_mode=False
)
# prepare int-8 model for training
# model = prepare_model_for_int8_training(model)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


from transformers import DataCollatorForSeq2Seq
 
# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

 
output_dir = output_dir
 
# model.gradient_checkpointing_enable()
torch.cuda.empty_cache()
 
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,  # Try 2 or more since you now have 160GB total
    gradient_accumulation_steps=1,
    learning_rate=1e-3,
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
    report_to="none",
    bf16=True,  # Use bf16 if your GPUs support it
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    # fp16=False,  # Either bf16=True or fp16=True, not both
    # no_cuda=False,  # Optional: force GPU
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset_train,
)

# # train model
trainer.train()


# # Save our LoRA model & tokenizer results
peft_model_id = peft_model_id
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
trainer.save_model(save_model_path)
