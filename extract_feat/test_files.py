import subprocess
import os

mp4_path = "/local_datasets/XD-violence/videos/" #"/local_datasets/XD-violence/test_videos" #"/local_datasets/XD-violence/videos/"
audio_path = "/data/datasets/XD-violence/audios/train" #"/data/datasets/XD-violence/audios/test" #"/local_datasets/XD-violence/audios/train"
wav2clip_feat_path = "/local_datasets/XD-violence/features/audio/wav2clip-features/test" #"/local_datasets/XD-violence/features/audio/wav2clip-features/test" # "/local_datasets/XD-violence/features/audio/wav2clip-features/train"

#####
CLIP_Train_video_feat_path = "/local_datasets/XD-violence/features/image/XDTrainClipFeatures/" # Train
CLIP_Test_video_feat_path = "/local_datasets/XD-violence/features/image/XDTestClipFeatures/" # Test

vggish_feat_path = "/local_datasets/XD-violence/features/audio/vggish-features/test/"


# ----- 메모 ------------------------------
# 1. Video: 24fps, 16framse = 1segment. that is, 3segments = 2seconds
# 2. Audio: 16kHz, 1초당 16kHz개의 신호 존재. 


# 2. Nan 뜨는 애 판별 -> NaN인 애는 없음
import numpy as np
import sys
if __name__ == '__main__':
    files = sorted(os.listdir(vggish_feat_path)) # *.npy
    for file in files:
        file_path = os.path.join(vggish_feat_path, file)
        audio = np.load(file_path)
        print(audio.shape)

        """
        has_nan = np.isnan(audio).any()
        has_inf = np.isinf(audio).any()

        #print(has_nan, has_inf)
        if has_nan or has_inf:
            sys.stdout.write(f'file: {file}\n')
            sys.stdout.flush()
        
        #print(audio)
        """
        


# 1. 없는 애 판별
"""
if __name__ == '__main__':
    files = os.listdir(mp4_path) # [*.mp4]
    for file in files:
        file = file[:-4]
        feat_path = os.path.join(wav2clip_feat_path, f'{file}.npy') # /local_datasets/XD-violence/features/audio/wav2clip-features/train/*.npy
        if os.path.isfile(feat_path):
            print("Ok")
        else:
            print(f'{file}.mp4') # v=8cTqh9tMz_I__#1_label_A.mp4
            break
"""