
import argparse

parser = argparse.ArgumentParser(description='VadCLIP + MACIL-SD')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)

# 이거 변경(UCF)
#parser.add_argument('--model-path', default='./ucf_save/model_ucf.pth') # UCF
#parser.add_argument('--checkpoint-path', default='./ucf_save/checkpoint.pth') # UCF
#parser.add_argument('--model-path', default='./ucf_save/OurBestModel/OurBestModel.pth') # './ucf_save/OurBestModel/OurBestModel.pth'
#parser.add_argument('--checkpoint-path', default='./ucf_save/OurBestModel/OurBestModel.pth') # './ucf_save/OurBestModel/OurBestModel.pth'
parser.add_argument('--visual-layers', default=1, type=int) # ucf랑 다름 -> 원본: 1
parser.add_argument('--attn-window', default=4, type=int) # ucf랑 다름 -> 원본: 4

parser.add_argument('--classes-num', default=7, type=int) # ucf랑 다름 -> 원본: 7
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)

# Self-distillation parameters
parser.add_argument('--m', default=0.91, type=float, help='Self-distillation mixing coefficient')
parser.add_argument('--distill-weight', default=1.0, type=float, help='Weight for distillation loss')
parser.add_argument('--max-epoch', default=10, type=int)

parser.add_argument('--model-path', default='./save_folder4/model_xd.pth') # save path
parser.add_argument('--checkpoint-path', default='./save_folder4/checkpoint.pth') # save path
#parser.add_argument('--model-path', default='./baseline/OurBestModel.pth') # save path
#parser.add_argument('--checkpoint-path', default='./baseline/OurBestModel.pth') # save path
#parser.add_argument('--model-path', default='./No_CMAL/No_CmalXdModel.pth') # save path
#parser.add_argument('--checkpoint-path', default='./No_CMAL/checkpoint.pth') # save path
#parser.add_argument('--model-path', default='./xd_save/OurBestXdModel.pth') # save path
#parser.add_argument('--checkpoint-path', default='./xd_save/OurBestXdModel.pth') # save path

parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--batch-size', default=96, type=int) # 원본: 96

# vggish
parser.add_argument('--train-list', default='./list/xd_CLIP_rgb.csv') # data_path
parser.add_argument('--test-list', default='./list/xd_CLIP_rgbtest.csv') # data_path
parser.add_argument('--audio-list', default='./list/xd_vgg_audio.csv')  # Added for MACIL-SD
parser.add_argument('--test-audio-list', default='./list/xd_vgg_audiotest.csv')  # Added for MACIL-SD
parser.add_argument('--audio-dim', default=128, type=int) # vggish

# wav2clip
"""
parser.add_argument('--train-list', default='./list/xd_CLIP_rgb_sub.csv') # data_path
parser.add_argument('--test-list', default='./list/xd_CLIP_rgbtest.csv') # data_path
parser.add_argument('--audio-list', default='./list/xd_wav2clip_audio_sub.csv')
parser.add_argument('--test-audio-list', default='./list/xd_wav2clip_audiotest.csv')
parser.add_argument('--audio-dim', default=512, type=int)
"""

parser.add_argument('--gt-path', default='./list/gt.npy')
parser.add_argument('--gt-segment-path', default='./list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='./list/gt_label.npy')
parser.add_argument('--lr', default=1e-5) #원본
parser.add_argument('--v-lr', default=2e-6, help='Learning rate for visual teacher model')
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[5, 10, 15])
