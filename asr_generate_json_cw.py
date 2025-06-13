# generate_json_cw.py

import os
import json
from tqdm import tqdm
from faster_whisper import WhisperModel
from opencc import OpenCC
import re

def contains_traditional_chinese(text):
    # cc = OpenCC('t2s')  # 繁轉簡
    # simplified = cc.convert(text)
    # return text != simplified  # 若轉換後不同，代表原本有繁體字
    # 簡單判斷是否含有任一 CJK 統一漢字
    return bool(re.search(r'[\u4E00-\u9FFF]', text))

# ========= 使用者設定 =========
wav_dir   = "/tmp2/b10902031/AICUP/test/audio"
json_out  = "transcriptions.json"
# cool-whisper 模型
cw_model_id = "models/cool-whisper/snapshots/9243ae0ca0cff575d2ca8a5c5698e232bf47821c"
cw_asr      = WhisperModel(cw_model_id, device="cuda", compute_type="float16")
# ==============================

transcriptions = {}

for fname in tqdm(sorted(os.listdir(wav_dir))):
    if not fname.lower().endswith(".wav"):
        continue
    wav_path = os.path.join(wav_dir, fname)

    # 自動偵測語言 + word timestamps
    segments, info = cw_asr.transcribe(
        wav_path,
        beam_size=5,
        language="zh",
        word_timestamps=True,
        condition_on_previous_text=True
    )

    # 組 word-level timestamps
    words = []
    full_text = ""
    for seg in segments:
        for w in seg.words:
            words.append({
                "word":  w.word,
                "start": round(w.start, 3),
                "end":   round(w.end,   3),
            })
            full_text += w.word

    # 存入 dict
    transcriptions[fname] = {
        "language": "zh" if contains_traditional_chinese(full_text) else "en",
        "text":     full_text,        # 完整文字（中文一字/英文一詞）
        "words":    words,            # word-level timestamps
    }

# 寫出 JSON
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(transcriptions, f, ensure_ascii=False, indent=2)

print("✅ Done: generated", json_out)
