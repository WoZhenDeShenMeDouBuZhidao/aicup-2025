import os
import soundfile as sf
from tqdm import tqdm

def stereo_to_mono_wavs(input_dir, output_dir=None):
    """
    Converts all stereo WAV files in input_dir to mono.
    If output_dir is None, overwrites original files.
    Otherwise, saves mono files to output_dir.
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for fname in tqdm(os.listdir(input_dir)):
        if fname.lower().endswith('.wav'):
            in_path = os.path.join(input_dir, fname)
            # 載入音檔
            data, samplerate = sf.read(in_path)
            # 若為雙聲道，轉為單聲道
            if len(data.shape) == 2 and data.shape[1] == 2:
                mono_data = data.mean(axis=1)  # 兩聲道取平均
                print(f"Converting {fname} from stereo to mono...")
            else:
                mono_data = data  # 已是單聲道
            # 決定輸出位置
            out_path = os.path.join(output_dir or input_dir, fname)
            # 儲存單聲道音檔
            sf.write(out_path, mono_data, samplerate)

# 用法
# 只要改 input_dir 跟 output_dir
input_dir = '/tmp2/b10902031/AICUP/test/private'
output_dir = '/tmp2/b10902031/AICUP/test/audio'  # 或指定新的資料夾，例如 '/tmp2/b10902031/AICUP/valid/audio_mono'
stereo_to_mono_wavs(input_dir, output_dir)
