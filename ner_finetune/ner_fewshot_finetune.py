#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ner.py

執行時可透過 --ner_types 指定多個 NER 類型，例如：
    python train_ner.py --ner_types PATIENT DOCTOR FAMILYNAME PERSONALNAME
"""

import os
import sys
import torch
import random
import argparse
import json
import math
from tqdm import tqdm

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from trl import SFTConfig, SFTTrainer

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune NER 模型，動態指定要訓練的 NER 類型")
    parser.add_argument(
        "--ner_types",
        nargs="+",
        required=True,
        help="要訓練的 NER 類型清單，例如：PATIENT DOCTOR FAMILYNAME"
    )
    parser.add_argument("--user1", required=True, help="範例一輸入")
    parser.add_argument("--asit1", required=True, help="範例一輸出")
    parser.add_argument("--user2", required=True, help="範例二輸入")
    parser.add_argument("--asit2", required=True, help="範例二輸出")
    parser.add_argument("--user3", required=True, help="範例三輸入")
    parser.add_argument("--asit3", required=True, help="範例三輸出")
    parser.add_argument("--user4", required=True, help="範例四輸入")
    parser.add_argument("--asit4", required=True, help="範例四輸出")
    parser.add_argument("--user5", required=True, help="範例五輸入")
    parser.add_argument("--asit5", required=True, help="範例五輸出")
    # 若有其他自訂參數，例如 batch_size、epoch 數量，也可以在此加入
    return parser.parse_args()

def load_jsonlines(file_name: str):
    with open(file_name, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def prepare_data(dataset_path: str, args, sft_tokenizer):
    trainset_fname = dataset_path
    trainset = load_jsonlines(trainset_fname)

    formatted_trainset = []
    max_token_len = 0
    
    print("Formatting training data...")
    for train_data in tqdm(trainset, total=len(trainset)):
        chats = [
            {"role": "system", "content": train_data['system']},
            {"role": "user", "content": args.user1},
            {"role": "assistant", "content": args.asit1},
            {"role": "user", "content": args.user2},
            {"role": "assistant", "content": args.asit2},
            {"role": "user", "content": args.user3},
            {"role": "assistant", "content": args.asit3},
            {"role": "user", "content": args.user4},
            {"role": "assistant", "content": args.asit4},
            {"role": "user", "content": args.user5},
            {"role": "assistant", "content": args.asit5},
            {"role": "user", "content": train_data['user']},
            {"role": "assistant", "content": train_data['assistant']},
        ]
        
        train_sample = sft_tokenizer.apply_chat_template(chats, tokenize=False)
        # (Optionally, if your template includes extra header information, remove it here.)
        header = """
Cutting Knowledge Date: December 2023
Today Date: 04 Jun 2025
"""
        train_sample = train_sample.replace(header, "")
            

        formatted_trainset.append({"text": train_sample})
        tokenized = sft_tokenizer(train_sample)
        max_token_len = max(max_token_len, len(tokenized["input_ids"]))
    
    print(f"max_token_len = {max_token_len}")
    dataset = Dataset.from_list(formatted_trainset)
    print("Sample text:\n", dataset[0]["text"])
    return dataset, max_token_len

def main():
    # 1. 解析命令列參數
    args = parse_args()
    specified_types_for_this_run = args.ner_types  # 由命令列帶入
    
    # 2. 設定 random seed、device
    set_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 3. 模型與 Tokenizer 設定
    sft_bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    sft_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="/tmp2/b10902031/LLMs/models/meta-llama/Llama-3.2-3B-Instruct",
        quantization_config=sft_bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    sft_tokenizer = AutoTokenizer.from_pretrained("/tmp2/b10902031/LLMs/models/meta-llama/Llama-3.2-3B-Instruct")
    
    # 4. PEFT / LoRA 設定
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["up_proj", "down_proj", "gate_proj", "k_proj", "q_proj", "v_proj", "o_proj"]
    )
    peft_model = get_peft_model(sft_model, peft_config)
    
    # 5. 訓練資料路徑（依據 NER 類型串成檔名）
    types_str = "_".join(specified_types_for_this_run)
    train_dataset, max_token_len = prepare_data(f"/tmp2/b10902031/AICUP/train/ner_finetune_trainset_{types_str}.jsonl", args, sft_tokenizer)
    valid_dataset, _ = prepare_data(f"/tmp2/b10902031/AICUP/valid/ner_finetune_trainset_{types_str}.jsonl", args, sft_tokenizer)

    # 6. Fine-tuning 配置
    output_dir = f"adapter_fewshot_{types_str}"
    training_arguments = SFTConfig(
        use_liger_kernel=True,
        seed=1126,
        data_seed=1126,
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=10,
        logging_strategy="epoch",
        logging_steps=1,
        eval_strategy="epoch",
        eval_steps=1,
        save_strategy="epoch",
        save_steps=1,
        lr_scheduler_type="linear",
        learning_rate=4e-5,
        weight_decay=1e-4,
        bf16=True,
        do_eval=True,
        group_by_length=True,
        packing=True,
        max_seq_length=max_token_len,
        dataset_text_field="text",
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        processing_class=sft_tokenizer,
        args=training_arguments,
    )
    
    # 7. 開始 Fine-tuning
    trainer.train()

if __name__ == "__main__":
    main()
