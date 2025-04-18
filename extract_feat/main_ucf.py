import subprocess
import os

mp4_path = "/local_datasets/UCF_Crimes/Origin/Videos" # Categories
audio_path = "/local_datasets/UCF_Crimes/Recon/Mel/CFG4.5_dopri5_26_gen_wav_16k_80"
wav2clip_feat_path = "/local_datasets/UCF_Crimes/features/audio/wav2clip-features" #"/local_datasets/XD-violence/features/audio/wav2clip-features/test" # "/local_datasets/XD-violence/features/audio/wav2clip-features/train"

# ----- 메모 ------------------------------
# 1. Video: 24fps, 16framse = 1segment. that is, 3segments = 2seconds
# 2. Audio: 16kHz, 1초당 16kHz개의 신호 존재. 


# ----- 메모 ------------------------------
"""
# 1. extract audio 
def extract_audio(mp4_path, output_wav_path):
    command = [
        "ffmpeg", "-i", mp4_path,
        "-ac", "1",              # mono
        "-ar", "16000",          # 16kHz
        "-vn",                   # no video
        "-y",                    # overwrite
        output_wav_path
    ]
    subprocess.run(command)


if __name__ == '__main__':
    categories = os.listdir(mp4_path) # [Abuse, ...] - Categories
    
    for category in categories:
        dir_path = os.path.join(mp4_path, category) # /local_datasets/.../Abuse/
        files = os.listdir(dir_path) # [*.mp4]
        for file in files: # *.mp4
            file = file[:-4]
            file_path = os.path.join(dir_path, f'{file}.mp4') # /local_datasets/.../Abuse/*.mp4
            save_dir = os.path.join(audio_path, category) # /local_datasets/.../Abuse/
            os.makedirs(save_dir, exist_ok=True)
            output_wav_path = os.path.join(save_dir, f'{file}.wav') # /local_dataset/..../*.wav
            extract_audio(file_path, output_wav_path)
"""


# 2. wav2clip features extraction
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import torch
import torchaudio
import numpy as np
import sys
import wav2clip
import librosa
import gc
from torch.cuda.amp import autocast

# 630 / 3 * 2 = 420초 -> 20개 -> 20초당 1개?
import os, gc, sys
import numpy as np
import torch, torchaudio
from torch.cuda.amp import autocast
import wav2clip

def custom_forward(self, x):
    if self.frame_length and self.hop_length:
        if x.shape[0] == 1:
            frames = np.expand_dims(
                librosa.util.frame(
                    x.cpu().numpy()[0], frame_length = self.frame_length, hop_length = self.hop_length
                ),
                axis=0,
            )
        else:
            frames = librosa.util.frame(
                x.cpu().numpy(), frame_length = self.frame_length, hop_length = self.hop_length
            )
        batch_size, frame_size, frame_num = frames.shape
        feature = self.encoder(
            torch.swapaxes(torch.from_numpy(frames), 1, 2)
            .reshape(batch_size * frame_num, frame_size)
            .to(next(self.encoder.parameters()).device)
        )
        _, embedding_size = feature.shape
        if self.transform:
            feature = self.transform(feature)
        feature = torch.swapaxes(
            feature.reshape(batch_size, frame_num, embedding_size), 1, 2
        )
        if self.scenario == "frozen":
            feature.detach()
        return feature
    else:  # utterance
        feature = self.encoder(x)
        if self.transform:
            feature = self.transform(feature)
        if self.scenario == "frozen":
            feature.detach()
    return feature

if __name__ == '__main__':
    # 1) 모델을 두 개 만들어 각각의 GPU에 올려 둡니다
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    device2 = torch.device("cuda:2")
    device3 = torch.device("cuda:3")

    model0 = wav2clip.get_model(
        frame_length=16000,
        hop_length=int(16000 * 2 / 3)
    ).to(device0)
    model1 = wav2clip.get_model(
        frame_length=16000,
        hop_length=int(16000 * 2 / 3)
    ).to(device1)
    model2 = wav2clip.get_model(
        frame_length=16000,
        hop_length=int(16000 * 2 / 3)
    ).to(device2)
    model3 = wav2clip.get_model(
        frame_length=16000,
        hop_length=int(16000 * 2 / 3)
    ).to(device3)
    

    # custom_forward 를 두 모델에 바인딩
    model0.forward = custom_forward.__get__(model0)
    model1.forward = custom_forward.__get__(model1)
    model2.forward = custom_forward.__get__(model2)
    model3.forward = custom_forward.__get__(model3)

    catecories = sorted(os.listdir(audio_path)) # Abuse..

    with torch.no_grad():
        for category in catecories:
            wav_files = os.listdir(os.path.join(audio_path, category)) # *.mp4
            save_path = os.path.join(wav2clip_feat_path, category)
            os.makedirs(save_path, exist_ok=True)

            for idx, fname in enumerate(wav_files):
                # 2) 짝수 idx → GPU0, 홀수 idx → GPU1
                wav_path = os.path.join(audio_path, category, fname)
                audio, sr = torchaudio.load(wav_path)     # (1, L) CPU tensor

                audio_np = audio.cpu().numpy().astype(np.float32)
                total_samples = audio_np.shape[-1]

                if total_samples >= 16000 * 4:
                #─── 3) 시간축 반반으로 split ───
                    sec = total_samples // 4
                    
                    first_half  = audio_np[..., :sec]
                    second_half = audio_np[..., sec:2*sec]
                    third_half = audio_np[..., 2*sec:3*sec]
                    four_half = audio_np[..., 3*sec:]

                    emb0 = wav2clip.embed_audio(first_half,  model0)  # on cuda:0
                    emb1 = wav2clip.embed_audio(second_half, model1)  # on cuda:1
                    emb2 = wav2clip.embed_audio(third_half, model2)  # on cuda:1
                    emb3 = wav2clip.embed_audio(four_half, model3)  # on cuda:1

                    emb = np.concatenate([emb0, emb1, emb2, emb3], axis=2) 
                    emb = emb.transpose(0,2,1).squeeze(0) # [T,512]
                else:
                    emb = wav2clip.embed_audio(audio_np,  model0)
                    emb = emb.transpose(0,2,1).squeeze(0)  # [T,512]

                # 6) 저장
                out_path = os.path.join(save_path, f"{fname[:-4]}.npy")
                np.save(out_path, emb)

                # 7) 로그
                sys.stdout.write(f"{fname} → {emb.shape}\n")
                sys.stdout.flush()

                # 8) 메모리 정리
                del audio, emb
                gc.collect()
                torch.cuda.empty_cache()





"""
def custom_forward(self, x):
    if self.frame_length and self.hop_length:
        if x.shape[0] == 1:
            frames = np.expand_dims(
                librosa.util.frame(
                    x.cpu().numpy()[0], frame_length = self.frame_length, hop_length = self.hop_length
                ),
                axis=0,
            )
        else:
            frames = librosa.util.frame(
                x.cpu().numpy(), frame_length = self.frame_length, hop_length = self.hop_length
            )
        batch_size, frame_size, frame_num = frames.shape
        feature = self.encoder(
            torch.swapaxes(torch.from_numpy(frames), 1, 2)
            .reshape(batch_size * frame_num, frame_size)
            .to(next(self.encoder.parameters()).device)
        )
        _, embedding_size = feature.shape
        if self.transform:
            feature = self.transform(feature)
        feature = torch.swapaxes(
            feature.reshape(batch_size, frame_num, embedding_size), 1, 2
        )
        if self.scenario == "frozen":
            feature.detach()
        return feature
    else:  # utterance
        feature = self.encoder(x)
        if self.transform:
            feature = self.transform(feature)
        if self.scenario == "frozen":
            feature.detach()
    return feature


if __name__ == '__main__':
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = wav2clip.get_model(frame_length=16000, hop_length=int(16000 * 2 / 3)).to(device) # hop_length: segments 단위 
        model.forward = custom_forward.__get__(model)

        categories = os.listdir(audio_path) # [Abuse, ...] - Categories
        for category in categories:
            dir_path = os.path.join(audio_path, category) # /local_datasets/.../Abuse/
            wav_files = os.listdir(dir_path) # [*.wav]
            save_dir = os.path.join(wav2clip_feat_path, category)
            os.makedirs(save_dir, exist_ok=True)

            for file in wav_files:
                wav_path = os.path.join(dir_path, file) # /local_dataset/..../Abuse/*.wav
                audio, sr = torchaudio.load(wav_path)
                with autocast():
                    embeddings = wav2clip.embed_audio(audio.cpu().numpy(), model) # [B, 512, T]
                embeddings = np.squeeze(embeddings.transpose(0, 2, 1), axis=0).astype(np.float32) # [T, 512]
                
                np.save(os.path.join(save_dir, f"{file[:-4]}.npy"), embeddings)
                sys.stdout.write(f'{file}: {embeddings.shape}\n')
                sys.stdout.flush()
                del audio
                gc.collect()
                torch.cuda.empty_cache()

"""