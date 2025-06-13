# correct_with_nemo.py

import os
import json
from tqdm import tqdm
import nemo.collections.asr as nemo_asr
from nemo.utils import logging as nemo_logging
# 可選 CRITICAL, ERROR, WARNING, INFO, DEBUG
nemo_logging.set_verbosity(nemo_logging.ERROR)

# ========= 使用者設定 =========
wav_dir      = "/tmp2/b10902031/AICUP/test/audio"
json_in      = "transcriptions.json"
json_out     = "transcriptions_corrected.json"
summary_out  = "task1_answer.txt"
# NVIDIA NeMo ASR（英文專家）
# nemo_model = nemo_asr.models.ASRModel.from_pretrained(
#     model_name="nvidia/parakeet-tdt-0.6b-v2"
# )
nemo_model = nemo_asr.models.ASRModel.restore_from(
    restore_path="/tmp2/b10902031/AICUP/asr_finetune/nemo_experiments/Speech_To_Text_Finetuning/2025-05-31_00-23-07/checkpoints/Speech_To_Text_Finetuning.nemo"
)
# ==============================

# 1. 讀入先前的 JSON
with open(json_in, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 對所有判為 en 的檔，重新跑 NeMo ASR
for fname in tqdm(sorted(data.keys())):
    entry = data[fname]
    if entry["language"] != "en":
        continue  # 只修正英文

    wav_path = os.path.join(wav_dir, fname)
    nemo_out = nemo_model.transcribe([wav_path], timestamps=True)[0]
    # 重新組 full_text + word timestamps
    full_text = nemo_out.text.strip()
    words = [
        {"word": w["word"],
         "start": round(w["start"], 3),
         "end":   round(w["end"],   3)}
        for w in nemo_out.timestamp["word"]
    ]

    if len(full_text) > 0:
        # 更新 JSON
        entry["text"]     = full_text
        entry["words"]    = words
        entry["language"] = "en"  # 再次確認

# 3. 寫出修正後的 JSON
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print("✅ Done: generated", json_out)

# 4. 產生 summary.txt，格式：fname \t zh/en \t full_text
with open(summary_out, "w", encoding="utf-8") as f:
    for fname in sorted(data.keys()):
        lang = data[fname]["language"]
        text = data[fname]["text"]
        f.write(f"{fname.split('.')[0]}\t{text}\n")
print("✅ Done: generated", summary_out)
