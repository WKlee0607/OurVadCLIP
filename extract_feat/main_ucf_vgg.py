import subprocess
import os

mp4_path = "/local_datasets/UCF_Crimes/Origin/Videos" # Categories
audio_path = "/local_datasets/UCF_Crimes/Recon/Mel/CFG4.5_dopri5_26_gen_wav_16k_80"
wav2clip_feat_path = "/local_datasets/UCF_Crimes/features/audio/vggish" #"/local_datasets/XD-violence/features/audio/wav2clip-features/test" # "/local_datasets/XD-violence/features/audio/wav2clip-features/train"

# ----- 메모 ------------------------------
# 1. Video: 24fps, 16framse = 1segment. that is, 3segments = 2seconds
# 2. Audio: 16kHz, 1초당 16kHz개의 신호 존재. 


# ----- 메모 ------------------------------


# 2. wav2clip features extraction
import torch
import torchaudio
import numpy as np
import sys
import gc
from torch.cuda.amp import autocast

# 630 / 3 * 2 = 420초 -> 20개 -> 20초당 1개?
import os, gc, sys
import numpy as np
import torch, torchaudio
from torch.cuda.amp import autocast
import wav2clip
import resampy

if __name__ == '__main__':
    # 1) 모델을 두 개 만들어 각각의 GPU에 올려 둡니다
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    device2 = torch.device("cuda:2")
    device3 = torch.device("cuda:3")

    model0 = torch.hub.load('harritaylor/torchvggish', 'vggish', device=device0, postprocess=False)
    model1 = torch.hub.load('harritaylor/torchvggish', 'vggish', device=device1, postprocess=False)
    model2 = torch.hub.load('harritaylor/torchvggish', 'vggish', device=device2, postprocess=False)
    model3 = torch.hub.load('harritaylor/torchvggish', 'vggish', device=device3, postprocess=False)

    model0.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    
    catecories = sorted(os.listdir(audio_path)) # Abuse..

    with torch.no_grad():
        for category in catecories:
            wav_files = os.listdir(os.path.join(audio_path, category)) # *.mp4
            save_path = os.path.join(wav2clip_feat_path, category)
            os.makedirs(save_path, exist_ok=True)

            for idx, fname in enumerate(wav_files):
                # 2) 짝수 idx → GPU0, 홀수 idx → GPU1
                wav_path = os.path.join(audio_path, category, fname)
                audio, sr = torchaudio.load(wav_path) # (1, L) CPU tensor

                audio_np = audio.cpu().numpy().astype(np.float32) # [1, T]
                total_samples = audio_np.shape[-1]
                
                if total_samples >= 16000 * 4:
                #─── 3) 시간축 반반으로 split ───
                    sec = total_samples // 4
                    
                    first_half  = audio_np[..., :sec]
                    second_half = audio_np[..., sec:2*sec]
                    third_half = audio_np[..., 2*sec:3*sec]
                    four_half = audio_np[..., 3*sec:]

                    emb0 = model0.forward(first_half[0], fs=16000)
                    emb1 = model1.forward(second_half[0], fs=16000)
                    emb2 = model2.forward(third_half[0], fs=16000)
                    emb3 = model3.forward(four_half[0], fs=16000)

                    emb0 = emb0.cpu().numpy() # [T/4, 128]
                    emb1 = emb1.cpu().numpy() 
                    emb2 = emb2.cpu().numpy() 
                    emb3 = emb3.cpu().numpy() 

                    #sys.stdout.write(f"{fname} → {emb0.shape}\n")
                    #sys.stdout.flush()

                    emb = np.concatenate([emb0, emb1, emb2, emb3], axis=0) # [T, 128]
                else:
                    emb = model0.forward(audio_np[0], fs=16000) # [T, 128]
                    emb = emb.cpu().numpy()

                    #sys.stdout.write(f"{fname} → {emb.shape}\n")
                    #sys.stdout.flush()

                # 6) 저장
                out_path = os.path.join(save_path, f"{fname[1:-4]}.npy")
                np.save(out_path, emb)

                # 7) 로그
                sys.stdout.write(f"{fname[1:-4]} → {emb.shape}\n")
                sys.stdout.flush()

                # 8) 메모리 정리
                del audio, emb
                gc.collect()
                torch.cuda.empty_cache()