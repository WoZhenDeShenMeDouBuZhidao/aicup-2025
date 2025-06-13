import json
import re
import os
from tqdm import tqdm
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel

# 新增：datasets 套件
from datasets import Dataset

# ------------------------------------------------------------
# 1. 命令列參數（與原本相同）
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune NER 模型，動態指定要訓練的 NER 類型")
    parser.add_argument(
        "--ner_types",
        nargs="+",
        required=True,
        help="要訓練的 NER 類型清單，例如：PATIENT DOCTOR FAMILYNAME"
    )
    parser.add_argument("--adapter_step", required=True, help="參數更新步數")
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
    return parser.parse_args()

args = parse_args()
ner_group = args.ner_types

# ------------------------------------------------------------
# 2. 常數與黑名單（與原本相同）
# ------------------------------------------------------------
NER_TAG_MEANING = {
    'PATIENT': """Refers to a patient's name (full, first, last, with title; e.g., "Johnathan Doe", "Mary", "Mr. Smith", "王大明", "麗華", "陳先生").""",
    'DOCTOR': """Refers to a doctor's name (e.g., "Dr. Emily White", "Dr. Jones", "Chang", "王醫師", "張志明醫師"); often prefixed with "Dr." or a title like "醫師".""",
    'FAMILYNAME': """Refers to a person's surname in a non-patient/doctor context or as a general family reference (e.g., "Williams", "Peterson family", "Chen's", "張", "黃氏").""",
    'PERSONALNAME': """Refers to a person's given or full name not tagged as PATIENT/DOCTOR (e.g., "Michael", "Sarah", "James Anderson", "家豪", "小美", "陳健明").""",
    'PROFESSION':"""Refers to a person's job or profession (e.g., "journalist", "actor", "a lawyer", "教師", "工程師", "律師"); includes job titles or descriptive phrases.""",
    'ROOM': """Specific room identifiers within a building (e.g., "Room 402", "Ward 3 East", "ICU Bed 5", "402號房", "東三區病房"); alphanumeric codes or names associated with rooms/beds.""",
    'DEPARTMENT': """Specific department within an institution (e.g., "Cardiology Department", "Radiology", "Oncology Ward", "心臟科", "放射科"); often includes "Department", "Unit", "Ward", "Lab", "科", "部", "中心".""",
    'HOSPITAL': """Name of a hospital or major medical center (e.g., "Mercy General Hospital", "St. Luke's Medical Center", "仁愛綜合醫院", "聖路加醫學中心"); often includes "Hospital", "Medical Center", "Clinic", "醫院", "醫學中心".""",
    'ORGANIZATION': """Company, non-medical institution, or other organized body (e.g., "BlueCross Health", "Tech Solutions Inc.", "Stanford University", "國泰人壽", "宏碁公司"); includes names of businesses, foundations, universities not tagged as hospital/department.""",
    'STREET': """Street names and numbers (e.g., "123 Willow Creek Drive", "Main St", "中山北路一段25號", "忠孝東路"); includes number, name, and type (Road, Ave, St, 路, 街, 巷, 弄, 號).""",
    'CITY': """Names of cities or towns (e.g., "Springfield", "Denver", "London", "台北市", "高雄"); capitalized names, often followed by state/country.""",
    'DISTRICT': """Named sub-regions of a city or larger administrative area (e.g., "Bronx", "Westminster borough", "大安區", "板橋區"); may include "District", "Borough", "區".""",
    'COUNTY': """Named administrative division (e.g., "Dade County", "Shire of Broome", "宜蘭縣", "屏東縣"); often includes "County", "Shire", "縣".""",
    'STATE': """Major administrative division of a country (e.g., "Texas", "NY", "Victoria", "SA", "加州", "台灣省"); full names or abbreviations, "州", "省".""",
    'COUNTRY': """Names of sovereign nations (e.g., "Canada", "United Kingdom", "USA", "日本", "英國", "美國"); full names or common unambiguous abbreviations.""",
    'ZIP': """Postal codes for mail (e.g., "9010", "2067", "110", "30078"); 4-digit or 5-digit numeric sequences.""",
    'LOCATION-OTHER': """Geographical entities not fitting other categories (e.g., "Midwest", "Silicon Valley", "Grand Canyon", "南部地區", "陽明山國家公園"); named regions, landmarks, imprecise locations.""",
    'AGE': """Numerical representation of a person's age (e.g., "35-year-old", "10", "age 62", "40s", "37歲", "十歲"); numbers often with "years old", "y.o.", "歲", or stand with a person's name in context.""",
    'DATE': """Specific dates, months, years, days of the week, relative dates (e.g., "June 15, 2024", "03/05/2023", "now", "next Monday", "August 1988", "2024年6月15日", "民國112年3月5日", "昨天", "上週"); various formats and textual representations.""",
    'TIME': """Specific times of day or relative times (e.g., "8:00 AM", "14:30", "this morning", "noon", "last night around 10 p.m.", "上午8點", "14時30分", "今天早上", "中午"); clock times (HH:MM, AM/PM) and textual phrases.""",
    'DURATION': """Spans of time (e.g., "three weeks", "24 hours", "several years", "a few days", "past few years", "a long time", "a while", "whole weekend", "三個星期", "這幾天", "春天", "這周期間"); numbers with time units or phrases.""",
    'SET': """Recurring events or frequencies (e.g., "twice a day", "every day", "per month", "weekly", "every 15 minutes", "每日服藥兩次", "每隔一天", "每月一次"); describes how often an event occurs.""",
    'MEDICAL_RECORD_NUMBER': """Unique identifier for a patient's medical record (e.g., "405974.QBV", "7890123.MRN", "2805065.FMV"); alphanumeric, most of them satisfy the format {digit sequence}.{alphabet sequence}""",
    'ID_NUMBER': """Other identification numbers (lab, specimen, national ID, insurance, etc.) (e.g., "73C10671", "lab number 92M63178", "NHI ZYX987A", "Accession ID: L2023-5678", "身分證字號為A123456789", "檢體編號 S2023-001"); alphanumeric, often context-specific.""",
}

NER_TAG_BLACKLIST = {
    'PROFESSION': [
        "job", "Dr.", "surgery", "parent", "brother", "sister", 
        "work", "parent", "patient", "doctor", "profess", "Profess", 
        "sci-fi", "title", "psychiatrist", "physician", "baby",
        ""
    ],
    'SET': [' mm', 'semester'],
    'DATE': [' mm', 'semester'],
    'DURATION': [' mm', 'semester'],
    'ORGANIZATION': ['organizations']
}

BATCH_SIZE = 64  # 視 GPU VRAM 調整

# ------------------------------------------------------------
# 3. 載入模型與 Pipeline（與原本相同）
# ------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="/tmp2/b10902031/LLMs/models/meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "/tmp2/b10902031/LLMs/models/meta-llama/Llama-3.2-3B-Instruct",
    padding_side='left'
)

# 先建立一個 text-generation pipeline
pipe = pipeline(
    'text-generation',
    model=base_model,
    batch_size=BATCH_SIZE, 
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=256,
    do_sample=False,
    temperature=None,
    top_p=None,
    top_k=None,
)
# 再把 adapter 插入
adapter_path = f"/tmp2/b10902031/AICUP/ner_finetune/adapter_fewshot_{'_'.join(ner_group)}/{args.adapter_step}"
pipe.model = PeftModel.from_pretrained(base_model, adapter_path)
pipe.tokenizer.pad_token_id = tokenizer.eos_token_id

# ------------------------------------------------------------
# 4. 幫助函式：把一筆 transcription 轉成 prompt
#    （與原本相同，只是改為獨立成一個 function）
# ------------------------------------------------------------
def build_prompt_for_text(transcription_text: str) -> str:
    ner_types = ""
    for ner_type in ner_group:
        ner_types += f"- {ner_type}: {NER_TAG_MEANING[ner_type]}\n"
    SYSTEM_PROMPT = f"""You are an expert in Named Entity Recognition.
Your task is to identify and tag entities in the provided Electronic Health Records.
If there is no specified entity, output original content.
Entity Type you need to recognize:
{ner_types}"""

    examples = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{args.user1}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{args.asit1}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{args.user2}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{args.asit2}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{args.user3}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{args.asit3}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{args.user4}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{args.asit4}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{args.user5}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{args.asit5}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{transcription_text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return examples

# ------------------------------------------------------------
# 5. 利用 Hugging Face Dataset 批次化呼叫 Pipeline
# ------------------------------------------------------------
def generate_ner_predictions_with_dataset(transcriptions_json_path: str, batch_size: int = 8) -> list[str]:
    """
    使用 Hugging Face Dataset + manual batching 的方式，並加上 tqdm 顯示進度，
    將所有 prompt 分批送入 pipeline 做 inference，最後再接續原本的解析流程。
    """
    final_output_lines = []

    # 1. 讀入 JSON
    with open(transcriptions_json_path, 'r', encoding='utf-8') as f:
        data_id_to_info = json.load(f)

    # 2. 預先計算每個 word 在 full_text 中的 char offset（完全沿用原本邏輯）
    for data_id, info in data_id_to_info.items():
        full_text = info['text']
        char_search_start = 0
        for word_obj in info.get('words', []):
            word_text = word_obj.get('word').strip()
            actual_char_start = full_text.find(word_text, char_search_start)
            if actual_char_start == -1:
                print(f"Warning (ID: {data_id}): Word '{word_text}' not found sequentially in full text.")
                word_obj['char_start_in_text'] = -1
                word_obj['char_end_in_text'] = -1
                char_search_start += len(word_text)
            else:
                word_obj['char_start_in_text'] = actual_char_start
                word_obj['char_end_in_text'] = actual_char_start + len(word_text)
                char_search_start = word_obj['char_end_in_text']

    # 3. 把所有 (data_id_wav, info) 收成 list
    entries = []
    for data_id_wav, info in data_id_to_info.items():
        entries.append((data_id_wav, info))

    # 4. 針對每筆 entry 產生 prompt，存成 list
    prompt_list = []
    for data_id_wav, info in entries:
        prompt_list.append(build_prompt_for_text(info['text']))

    # 5. Dataset.from_dict
    hf_dataset = Dataset.from_dict({"text": prompt_list})

    # --------------------------------------------------------------------
    # 6. 手動分批呼叫 pipeline 並加上 tqdm 顯示每個 batch 的進度
    # --------------------------------------------------------------------
    total_prompts = len(hf_dataset)
    total_batches = (total_prompts + batch_size - 1) // batch_size  # ceil
    
    all_pipe_outputs = []
    for batch_idx in tqdm(range(total_batches), desc="Running pipeline (batches)"):
        # 計算本 batch 的範圍
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_prompts)
        
        # 取出這個 batch 的 prompt 字串
        sub_prompts = hf_dataset["text"][start_idx:end_idx]
        
        # 真正呼叫 pipeline，得到這個 batch 的所有 outputs (list of dict or list of list(dict))
        batch_outputs = pipe(sub_prompts)
        
        # 把結果累加到 all_pipe_outputs
        all_pipe_outputs.extend(batch_outputs)
    # --------------------------------------------------------------------
    # 現在 all_pipe_outputs 的長度應該 = total_prompts
    # --------------------------------------------------------------------

    # 7. 逐筆把 pipeline output 拆解、解析、對齊時間戳，生成 final_output_lines
    for idx, (data_id_wav, info) in enumerate(entries):
        out_item = all_pipe_outputs[idx]
        if isinstance(out_item, list) and len(out_item) > 0 and "generated_text" in out_item[0]:
            generated = out_item[0]["generated_text"]
        elif isinstance(out_item, dict) and "generated_text" in out_item:
            generated = out_item["generated_text"]
        else:
            raise ValueError(f"Unexpected output format from pipeline: {out_item}")

        # 切出 <|start_header_id|>assistant<|end_header_id|> 與 <|eot_id|> 之間的純標註內容
        tagged_text = generated.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()

        full_text = info["text"]
        data_id_for_output = os.path.splitext(data_id_wav)[0]
        print(f"\n\nID = {data_id_wav}\n### full_text: {full_text}\n### ner_text: {tagged_text}\n")

        # 驗證去掉所有標籤後是否與原文一致
        reconstructed_clean = re.sub(r"<[^>]+>", "", tagged_text)
        if full_text != reconstructed_clean:
            print(f"(ID: {data_id_for_output}): Skipping data with unmatched input/output.")
            continue

        # 和原本一樣的正則 match、黑名單過濾、計算 char offset、對齊 timestamp
        word_objects_with_char_spans = info["words"]
        for match_obj in re.finditer(r"<([A-Z_]+)>([\s\S]*?)</\1>", tagged_text):
            ner_type = match_obj.group(1)
            if ner_type not in ner_group:
                print(f"(ID: {data_id_for_output}): Unknown NER type {ner_type}")
                continue

            ner_content_str = match_obj.group(2)
            if not ner_content_str.strip():
                print(f"(ID: {data_id_for_output}): Skipping empty NER content for type {ner_type}.")
                continue
            if full_text.find(ner_content_str) == -1:
                print(f"(ID: {data_id_for_output}): Skipping hallucination NER content for type {ner_type}.")
                continue

            # NER CONTENT 子字串黑名單
            skip = False
            if ner_type in NER_TAG_BLACKLIST:
                for removed_ner_content in NER_TAG_BLACKLIST[ner_type]:
                    if ner_content_str.find(removed_ner_content) != -1:
                        skip = True
            if skip:
                continue

            # DATE 額外過濾
            if ner_type == "DATE" and (ner_content_str == "every day" or ner_content_str == "Every day"):
                ner_type = "SET"

            # TIME 額外過濾
            if ner_type == "TIME" and (ner_content_str == "every day" or ner_content_str == "Every day"):
                ner_type = "SET"

            # DURATION 額外過濾
            if ner_type == "DURATION" and (ner_content_str == "every day" or ner_content_str == "Every day"):
                ner_type = "SET"
            
            # ID_NUMBER 額外過濾
            if ner_type == "ID_NUMBER" and (ner_content_str.find(" ") != -1 or bool(re.fullmatch(r"\d{4}", ner_content_str))):
                continue

            # DEPARTMENT 額外過濾
            if ner_type == "DEPARTMENT" and (ner_content_str.find("Health") != -1 or ner_content_str.find("Hospital") != -1):
                ner_type = "HOSPITAL"
            
            # ORGANIZATION 額外過濾
            if ner_type == "ORGANIZATION":
                if ner_content_str.find("Health") != -1 or ner_content_str.find("Hospital") != -1:
                    ner_type = "HOSPITAL"
                if  ner_content_str.find("Pathology") != -1 or ner_content_str.find("Clinic") != -1:
                    ner_type = "DEPARTMENT"

            # CITY 額外過濾
            if ner_type == "CITY" and bool(re.fullmatch(r"\d{4}", ner_content_str)):
                ner_type = "ZIP"

            # STATE 額外過濾
            if ner_type == "STATE" and bool(re.fullmatch(r"\d{4}", ner_content_str)):
                ner_type = "ZIP"
            
            # COUNTY 額外過濾
            if ner_type == "COUNTY" and ner_content_str.find(" Territory") != -1:
                ner_type = "STATE"

            # COUNTRY 額外過濾
            if ner_type ==  "COUNTRY" and ner_content_str.find("country") != -1:
                continue

            # STREET 額外過濾
            if ner_type == "STREET" and (ner_content_str.find("Department") != -1 or ner_content_str.find("department") != -1 or ner_content_str.find("Hospital") != -1):
                continue
            
            # ZIP 額外過濾
            if ner_type == "ZIP" and not bool(re.fullmatch(r"\d{4}", ner_content_str)):
                continue

            # 計算 char_start, char_end
            text_before = tagged_text[: match_obj.start(2)]
            char_start_in_original = len(re.sub(r"<[^>]+>", "", text_before))
            char_end_in_original = char_start_in_original + len(ner_content_str)
            last_char_idx_of_entity = char_end_in_original - 1

            actual_entity_start_time = None
            actual_entity_end_time = None
            for word_obj in word_objects_with_char_spans:
                w_char_start = word_obj.get("char_start_in_text", -1)
                w_char_end = word_obj.get("char_end_in_text", -1)
                if w_char_start == -1 or w_char_end == -1:
                    continue
                word_start_time_val = word_obj.get("start")
                word_end_time_val = word_obj.get("end")
                if word_start_time_val is None or word_end_time_val is None:
                    continue

                if actual_entity_start_time is None and (w_char_start <= char_start_in_original < w_char_end):
                    actual_entity_start_time = word_start_time_val
                if w_char_start <= last_char_idx_of_entity < w_char_end:
                    actual_entity_end_time = word_end_time_val

            if actual_entity_start_time is not None and actual_entity_end_time is not None:
                output_line = f"{data_id_for_output}\t{ner_type}\t{actual_entity_start_time:.3f}\t{actual_entity_end_time:.3f}\t{ner_content_str}"
                final_output_lines.append(output_line)
                print(output_line)
            else:
                print(
                    f"(ID: {data_id_for_output}): Could not map entity '{ner_content_str}' "
                    f"(type {ner_type}) to timestamps. Char span: ({char_start_in_original}-{char_end_in_original})."
                )

    return final_output_lines


if __name__ == "__main__":
    json_file_path = 'transcriptions_corrected.json'

    predictions = generate_ner_predictions_with_dataset(json_file_path, batch_size=BATCH_SIZE)

    print("\n--- Generated NER Predictions (task2_answer.txt format) ---")
    if predictions:
        for line in predictions:
            print(line)
    else:
        print("No predictions were generated.")

    types_str = "_".join(ner_group)
    output_file_path = f'task2_answer_{types_str}.txt'
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for line in predictions:
                f.write(line + '\n')
        print(f"\nPredictions saved to {output_file_path}")
    except IOError:
        print(f"Error: Could not write predictions to {output_file_path}")
