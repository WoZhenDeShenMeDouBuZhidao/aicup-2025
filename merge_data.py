def merge_asr_files(file1_path, file2_path, output_path):
    def read_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            if line.strip():  # 忽略空行
                try:
                    idx, content = line.strip().split('\t', 1)
                    data.append((int(idx), content))
                except ValueError:
                    print(f"格式錯誤: {line.strip()}")
        return data

    data1 = read_file(file1_path)
    data2 = read_file(file2_path)

    # 合併並排序
    merged_data = sorted(data1 + data2, key=lambda x: x[0])

    # 寫入輸出檔案
    with open(output_path, 'w', encoding='utf-8') as out:
        for idx, content in merged_data:
            out.write(f"{idx}\t{content}\n")

# 使用範例
merge_asr_files("train01/task1_answer.txt", "train02/task1_answer.txt", "train/task1_answer.txt")

def merge_ner_files(file1_path, file2_path, output_path):
    def parse_line(line):
        parts = line.strip().split('\t')
        if len(parts) < 5:
            raise ValueError(f"格式錯誤: {line}")
        return (int(parts[0]), float(parts[2]), line.strip())  # 以編號、start time 排序

    def read_file(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [parse_line(line) for line in f if line.strip()]

    entries = read_file(file1_path) + read_file(file2_path)
    entries.sort(key=lambda x: (x[0], x[1]))

    with open(output_path, 'w', encoding='utf-8') as out:
        for _, _, line in entries:
            out.write(line + '\n')

# 使用範例
merge_ner_files("train01/task2_answer.txt", "train02/task2_answer.txt", "train/task2_answer.txt")

import os
import shutil

def merge_wav_files(source_dir1, source_dir2, target_dir):
    # 確保目標目錄存在
    os.makedirs(target_dir, exist_ok=True)

    def copy_wavs_from_dir(src_dir):
        for filename in os.listdir(src_dir):
            if filename.lower().endswith('.wav'):
                src_path = os.path.join(src_dir, filename)
                dst_path = os.path.join(target_dir, filename)
                
                # 若目標已有相同檔名，避免覆蓋可加上編號或跳過
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(filename)
                    count = 1
                    while os.path.exists(dst_path):
                        new_filename = f"{base}_{count}{ext}"
                        dst_path = os.path.join(target_dir, new_filename)
                        count += 1

                shutil.copy2(src_path, dst_path)

    copy_wavs_from_dir(source_dir1)
    copy_wavs_from_dir(source_dir2)

# 使用方式
merge_wav_files("train01/audio_mono", "train02/audio_mono", "train/audio")