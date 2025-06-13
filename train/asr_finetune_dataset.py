import re
import os
import json
import wave
import contextlib

def contains_traditional_chinese(text):
    # 簡單判斷是否含有任一 CJK 統一漢字
    return bool(re.search(r'[\u4E00-\u9FFF]', text))

def get_wav_duration(filepath):
    with contextlib.closing(wave.open(filepath, 'r')) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

def txt_to_jsonl(txt_path, audio_dir, output_jsonl_path):
    with open(txt_path, 'r', encoding='utf-8') as txt_file, \
         open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:

        for line in txt_file:
            if not line.strip():
                continue
            try:
                audio_id, text = line.strip().split('\t', 1)

                if contains_traditional_chinese(text): continue

                audio_filename = f"{audio_id}.wav"
                audio_path = os.path.join(audio_dir, audio_filename)
                abs_audio_path = os.path.abspath(audio_path)

                if not os.path.isfile(audio_path):
                    print(f"音檔不存在：{audio_path}")
                    continue

                duration = get_wav_duration(audio_path)

                entry = {
                    "audio_filepath": abs_audio_path,
                    "duration": round(duration, 3),
                    "text": text
                }
                jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
            except ValueError:
                print(f"格式錯誤：{line.strip()}")

txt_to_jsonl("task1_answer.txt", "/tmp2/b10902031/AICUP/train/audio", "asr_finetune_trainset.jsonl")