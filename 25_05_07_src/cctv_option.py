
import argparse

parser = argparse.ArgumentParser(description='VadCLIP + MACIL-SD')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--audio-dim', default=128, type=int) # vggish

# 이거 변경(UCF)
parser.add_argument('--visual-layers', default=1, type=int) # ucf랑 다름 -> 원본: 1
parser.add_argument('--attn-window', default=4, type=int) # ucf랑 다름 -> 원본: 4
parser.add_argument('--classes-num', default=7, type=int) # ucf랑 다름 -> 원본: 7
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)

parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--batch-size', default=96, type=int)  # Reduced batch size for dual model training

# path
parser.add_argument('--model-path', default='./xd_save/OurBestXdModel.pth') # save path
parser.add_argument('--checkpoint-path', default='./xd_save/OurBestXdModel.pth') # save path
parser.add_argument('--test-path', default='/data/datasets/CCTV_Fights/features/videos/CLIP_4feat_mid') # data_path
parser.add_argument('--test-audio-path', default='/local_datasets/CCTV_Fights/features/audios/vggish_ori')  # vggish_ori: 4frame 기준으로 뽑음
parser.add_argument('--gt-path', default='./list/gt_cctv_ori.npy') # gt_cctv_ori.npy: 4frame 기준으로 뽑음