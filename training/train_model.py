import torch
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

ckpt = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(ckpt, device_map="auto",cache_dir="./models",attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(ckpt,cache_dir="./models")

with open("../data/processed_corpse.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# df = pd.DataFrame(data)
# df['create_time'] = pd.to_datetime(df['create_time'], unit='s')
# df["last_modified_time"] = pd.to_datetime(df["last_modified_time"], unit='s')
# df.sort_values(by="last_modified_time", ascending=False, inplace=True)

df = pd.DataFrame(data)
df["text"] = df["text"].apply(lambda x: x.replace("<endofsection>", ""))

rng = np.random.default_rng()
indicies = np.arange(len(df))
rng.shuffle(indicies)

split_ratio = 0.85
train_indices = indicies[:int(split_ratio * len(indicies))]
eval_indices = indicies[int(split_ratio * len(indicies)):]

training_dataset = Dataset.from_pandas(df.iloc[train_indices])
eval_dataset = Dataset.from_pandas(df.iloc[eval_indices])

training_args = SFTConfig(
    per_device_train_batch_size=4,
    packing = True,
    max_length=1024,
    num_train_epochs=10,
    learning_rate=5e-4,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=5,
    optim="adamw_torch_fused",
    dataset_text_field="text",
    output_dir="outputs",
    lr_scheduler_type="linear",
    weight_decay=0.01,
    warmup_steps=5,
)
sft_trainer = SFTTrainer(
    model  = model,
    args = training_args,
    train_dataset = training_dataset,
    eval_dataset = eval_dataset,
    processing_class = tokenizer
)

sft_trainer.train()