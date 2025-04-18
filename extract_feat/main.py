import subprocess
import os

mp4_path = "/local_datasets/XD-violence/videos/" #"/local_datasets/XD-violence/test_videos" #"/local_datasets/XD-violence/videos/"
audio_path = "/data/datasets/XD-violence/audios/test" #"/data/datasets/XD-violence/audios/test" #"/local_datasets/XD-violence/audios/train"
wav2clip_feat_path = "/local_datasets/XD-violence/features/audio/wav2clip-features/test" #"/local_datasets/XD-violence/features/audio/wav2clip-features/test" # "/local_datasets/XD-violence/features/audio/wav2clip-features/train"


#####
CLIP_Train_video_feat_path = "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/" # Train
CLIP_Test_video_feat_path = "/local_datasets/XD-violence/features/image/XDTestClipFeatures/" # Test


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
        "-n", #"-y",                    # overwrite
        output_wav_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL)

if __name__ == '__main__':
    mp4_files = os.listdir(mp4_path) # [*.mp4]
    for file in mp4_files:
        input_path = os.path.join(mp4_path, file) # /local_dataset/..../*.mp4
        output_wav_path = os.path.join(audio_path, f'{file[:-4]}.wav') # /local_dataset/..../*.wav
        extract_audio(input_path, output_wav_path)
"""

# test
"""
f = "/local_datasets/XD-violence/features/audio/vggish-features/train/Love.Death.and.Robots.S01E15__#0-02-51_0-03-05_label_G-0-0__vggish.npy" # [20, 128]
k = [
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Love.Death.and.Robots.S01E15__#0-02-51_0-03-05_label_G-0-0__0.npy,G-0-0",# [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Love.Death.and.Robots.S01E15__#0-02-51_0-03-05_label_G-0-0__1.npy,G-0-0",# [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Love.Death.and.Robots.S01E15__#0-02-51_0-03-05_label_G-0-0__2.npy,G-0-0",# [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Love.Death.and.Robots.S01E15__#0-02-51_0-03-05_label_G-0-0__3.npy,G-0-0",# [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Love.Death.and.Robots.S01E15__#0-02-51_0-03-05_label_G-0-0__4.npy,G-0-0",# [70, 512]
        '/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Love.Death.and.Robots.S01E15__#0-02-51_0-03-05_label_G-0-0__5.npy,G-0-0',# [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Love.Death.and.Robots.S01E15__#0-02-51_0-03-05_label_G-0-0__6.npy,G-0-0",# [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Love.Death.and.Robots.S01E15__#0-02-51_0-03-05_label_G-0-0__7.npy,G-0-0",# [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Love.Death.and.Robots.S01E15__#0-02-51_0-03-05_label_G-0-0__8.npy,G-0-0",# [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Love.Death.and.Robots.S01E15__#0-02-51_0-03-05_label_G-0-0__9.npy,G-0-0",# [70, 512]
    ]

    l = [
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Mindhunters.2004__#00-07-51_00-08-38_label_B2-0-0__0.npy", # [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Mindhunters.2004__#00-07-51_00-08-38_label_B2-0-0__1.npy", # [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Mindhunters.2004__#00-07-51_00-08-38_label_B2-0-0__2.npy", # [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Mindhunters.2004__#00-07-51_00-08-38_label_B2-0-0__3.npy", # [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Mindhunters.2004__#00-07-51_00-08-38_label_B2-0-0__4.npy", # [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Mindhunters.2004__#00-07-51_00-08-38_label_B2-0-0__5.npy", # [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Mindhunters.2004__#00-07-51_00-08-38_label_B2-0-0__6.npy", # [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Mindhunters.2004__#00-07-51_00-08-38_label_B2-0-0__7.npy", # [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Mindhunters.2004__#00-07-51_00-08-38_label_B2-0-0__8.npy", # [70, 512]
        "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/Mindhunters.2004__#00-07-51_00-08-38_label_B2-0-0__9.npy", # [70, 512]
    ]
    
    for f in l:
        video_feat = np.load(f)
        sys.stdout.write(f'{video_feat.shape}\n')
"""

# /local_datasets/XD-violence/videos/



# 2. wav2clip features extraction

import torch
import torchaudio
import numpy as np
import sys
import wav2clip
import librosa
import gc

from torch.cuda.amp import autocast
# 630 / 3 * 2 = 420초 -> 20개 -> 20초당 1개?


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
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    model0 = wav2clip.get_model(
        frame_length=16000,
        hop_length=10500
    ).to(device0)
    model1 = wav2clip.get_model(
        frame_length=16000,
        hop_length=10500
    ).to(device1)

    model0.forward = custom_forward.__get__(model0)
    model1.forward = custom_forward.__get__(model1)
    
    wav_files = sorted(os.listdir(audio_path)) # [*.wav]

    with torch.no_grad():
        for idx, file in enumerate(wav_files):
            wav_path = os.path.join(audio_path, file) # /local_dataset/..../*.wav
            audio, sr = torchaudio.load(wav_path)

            audio_np = audio.cpu().numpy().astype(np.float32)
            total_samples = audio_np.shape[-1]

            if total_samples >= 32000:
                #─── 3) 시간축 반반으로 split ───
                mid = total_samples // 2
                first_half  = audio_np[..., :mid]
                second_half = audio_np[..., mid:]

                emb0 = wav2clip.embed_audio(first_half,  model0)  # on cuda:0
                emb1 = wav2clip.embed_audio(second_half, model1)  # on cuda:1

                emb = np.concatenate([emb0, emb1], axis=2) 
                emb = emb.transpose(0,2,1).squeeze(0)  # [T,512]
                
            else:
                emb = wav2clip.embed_audio(audio_np,  model0)
                emb = emb.transpose(0,2,1).squeeze(0)  # [T,512]
                
            np.save(os.path.join(wav2clip_feat_path, f"{file[:-4]}.npy"), emb)

            sys.stdout.write(f'{file}: {emb.shape}\n')
            sys.stdout.flush()

            del audio
            gc.collect()
            torch.cuda.empty_cache()


    """
    wav_files = os.listdir(audio_path) # [*.wav]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = wav2clip.get_model(frame_length=16000, hop_length=int(16000 * 2 / 3)).to(device) # hop_length: segments 단위 
    model.forward = custom_forward.__get__(model)

    with torch.no_grad():
        for file in wav_files:
            wav_path = os.path.join(audio_path, file) # /local_dataset/..../*.wav
            audio, sr = torchaudio.load(wav_path)
            #with autocast():
            embeddings = wav2clip.embed_audio(audio.cpu().numpy(), model) # [B, 512, T]
            embeddings = np.squeeze(embeddings.transpose(0, 2, 1), axis=0).astype(np.float32) # [T, 512]

            np.save(os.path.join(wav2clip_feat_path, f"{file[:-4]}.npy"), embeddings)

            sys.stdout.write(f'{file}: {embeddings.shape}\n')
            sys.stdout.flush()

            del audio
            gc.collect()
            torch.cuda.empty_cache()
    """