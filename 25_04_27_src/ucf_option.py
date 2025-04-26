import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

# model
parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--classes-num', default=14, type=int) # xd랑 다름 -> 노상관
parser.add_argument('--audio-dim', default=128, type=int) # vggish

# ----------- 이거 변경 for Cross-dataset eval -----------
# 원본
parser.add_argument('--model-path', default='./ucf_save/model_ucf.pth')
parser.add_argument('--checkpoint-path', default='./ucf_save/checkpoint.pth')
#parser.add_argument('--visual-layers', default=2, type=int) # xd랑 다름 -> xd = 1 / 원본
#parser.add_argument('--attn-window', default=8, type=int) # xd랑 다름 -> xd = 64 / 원본
#parser.add_argument('--lr', default=2e-5) # 원본
#parser.add_argument('--batch-size', default=64, type=int) # 원본
#parser.add_argument('--scheduler-milestones', default=[4, 8]) # 원본
parser.add_argument('--visual-layers', default=3, type=int) # 수정본
parser.add_argument('--attn-window', default=8, type=int) # 수정본
parser.add_argument('--lr', default=2e-4) # 수정본
parser.add_argument('--batch-size', default=64, type=int) # 수정본
parser.add_argument('--scheduler-milestones', default=[7, 14]) # 수정본

"""
# for Cross-dataset eval
parser.add_argument('--model-path', default='./baseline/OurBestModel.pth') # save path
parser.add_argument('--checkpoint-path', default='./baseline/OurBestModel.pth') # save path
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=64, type=int)
"""
# ----------- 이거 변경 -----------


parser.add_argument('--train-list', default='./list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='./list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='./list/gt_ucf.npy')
parser.add_argument('--audio-list', default='./list/ucf_vgg_audio.csv')  
parser.add_argument('--test-audio-list', default='./list/ucf_vgg_audiotest.csv')  
parser.add_argument('--gt-segment-path', default='./list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='./list/gt_label_ucf.npy')
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)

parser.add_argument('--max-epoch', default=15, type=int)
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--scheduler-rate', default=0.1) 



