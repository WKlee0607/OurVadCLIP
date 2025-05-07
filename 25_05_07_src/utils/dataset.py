import numpy as np
import sys
import torch
import torch.utils.data as data
import pandas as pd
import os
import re
import utils.tools as tools
import json

import sys
class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, audio_list: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.audio_df = pd.read_csv(audio_list)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        
        # Create a mapping from video file key to audio path
        self.video_to_audio = {}
        
        # Format: movie__#timestamp_label_X__vggish.npy
        for _, row in self.audio_df.iterrows():
            audio_path = row['path']
            # Extract the base key (without __vggish.npy part)
            # Example: Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0
            audio_key = os.path.basename(audio_path).replace('__vggish.npy', '') # vggish
            #audio_key = os.path.basename(audio_path).replace('.npy', '') # wav2clip
            self.video_to_audio[audio_key] = audio_path
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        visual_path = self.df.loc[index]['path']
        clip_feature = np.load(visual_path) # [T, 512]
        # Extract visual file key (removing the __number.npy part)
        # Example: /data/.../Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0__0.npy
        # -> Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0
        visual_basename = os.path.basename(visual_path)
        visual_key = re.sub(r'__\d+\.npy$', '', visual_basename)
        
        # Try to get corresponding audio feature
        if visual_key in self.video_to_audio:
            audio_path = self.video_to_audio[visual_key]
            try:
                audio_feature = np.load(audio_path)
            except:
                sys.stdout.write(f"Warning: Failed to load audio feature from {audio_path}\n")
                sys.stdout.flush()
                audio_feature = np.zeros((clip_feature.shape[0], 128)) # 128: vggish / 512: wav2clip
        else:
            sys.stdout.write(f"Warning: No matching audio for visual key {visual_key}\n")
            sys.stdout.flush()
            audio_feature = np.zeros((clip_feature.shape[0], 128)) # 128: vggish / 512: wav2clip
            
        if self.test_mode == False:
            clip_feature, audio_feature, clip_length = tools.process_feat_audio(clip_feature, audio_feature, self.clip_dim)
        else:
            clip_feature, audio_feature, clip_length = tools.process_split_audio(clip_feature, audio_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        audio_feature = torch.tensor(audio_feature)
        clip_label = self.df.loc[index]['label']

        # 추가
        #file_name = os.path.splitext(visual_basename)[0]
        return clip_feature, audio_feature, clip_label, clip_length#, file_name

class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, audio_list: str, test_mode: bool, label_map: dict, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.audio_df = pd.read_csv(audio_list)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()
        
        # Create a mapping similar to XDDataset
        self.video_to_audio = {}
        for _, row in self.audio_df.iterrows():
            audio_path = row['path']
            audio_key = re.sub(r'__\d+\.npy$', '', os.path.basename(audio_path))#.replace('_0.npy', '')
            audio_path = re.sub(r'__\d+\.npy$', '_0.npy', audio_path)
            self.video_to_audio[audio_key] = audio_path
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        visual_path = self.df.loc[index]['path']
        clip_feature = np.load(visual_path)
        
        # Extract visual file key (removing the __number.npy part)
        visual_basename = os.path.basename(visual_path)
        visual_key = re.sub(r'__\d+\.npy$', '', visual_basename)

        #sys.stdout.write(f'visual_key: {visual_key} \n')
        #sys.stdout.flush()
        
        # Try to get corresponding audio feature
        if visual_key in self.video_to_audio:
            audio_path = self.video_to_audio[visual_key]
            try:
                audio_feature = np.load(audio_path)
            except:
                print(f"Warning: Failed to load audio feature from {audio_path}")
                audio_feature = np.zeros((clip_feature.shape[0], 128), dtype=np.float32)
        else:
            print(f"Warning: No matching audio for visual key {visual_key}")
            audio_feature = np.zeros((clip_feature.shape[0], 128), dtype=np.float32)
        
        if self.test_mode == False:
            clip_feature, audio_feature, clip_length = tools.process_feat_audio(clip_feature, audio_feature, self.clip_dim)
        else:
            clip_feature, audio_feature, clip_length = tools.process_split_audio(clip_feature, audio_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        audio_feature = torch.tensor(audio_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, audio_feature, clip_label, clip_length



##### CCTV Dataset
class CCTVDataset(data.Dataset):
    def __init__(self, clip_dim, visual_dir, audio_dir, annotation_file, subset="training"):
        self.clip_dim = clip_dim
        self.visual_dir = visual_dir
        self.audio_dir = audio_dir
        self.subset = subset # "training" or "testing"

        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)["database"]

        self.samples = []
        for video_name, meta in self.annotations.items(): # {video_name : {values}}
            if meta["subset"] != self.subset:
                continue
            base_name = video_name + ".npy"
            visual_path = os.path.join(visual_dir, base_name)
            audio_path = os.path.join(audio_dir, base_name)
            label = "Fight"

            if os.path.exists(visual_path) and os.path.exists(audio_path):
                self.samples.append((visual_path, audio_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        visual_path, audio_path, label = self.samples[idx]

        visual_feat = np.load(visual_path)  # [T, D_v]
        audio_feat = np.load(audio_path)    # [T, D_a]

        if self.subset == "training":
            clip_feature, audio_feature, clip_length = tools.process_feat_audio(visual_feat, audio_feat, self.clip_dim)
        else:
            clip_feature, audio_feature, clip_length = tools.process_split_audio(visual_feat, audio_feat, self.clip_dim)

        clip_feature = torch.tensor(clip_feature, dtype=torch.float32)
        audio_feature = torch.tensor(audio_feature, dtype=torch.float32)

        if label == "Fight":
            label = 1  # since all annotations are "Fight"
        else:
            label = 0
        return clip_feature, audio_feature, label, clip_length
