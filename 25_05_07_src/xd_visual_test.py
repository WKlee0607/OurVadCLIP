# xd_test.py
import torch
import sys
import os
import matplotlib.pyplot as plt  # 시각화를 위한 임포트
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D  
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CLIPVAD
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option

# -------------------- 추가 --------------------
class GaussianSmoothing1D_AutoKernel(nn.Module):
    def __init__(self, sigma: float, truncate: float = 4.0):
        """
        sigma를 기준으로 kernel_size를 자동 설정하는 Gaussian Smoothing
        Args:
            sigma (float): standard deviation of the Gaussian
            truncate (float): 몇 sigma까지만 사용할지 (default=4.0, scipy와 같음)
        """
        super().__init__()
        
        # kernel_size 자동 계산: kernel_size = 2 * truncate * sigma + 1
        kernel_size = int(2 * truncate * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1  # kernel_size는 무조건 홀수로
        
        # 1D Gaussian kernel 생성
        half = kernel_size // 2
        x = torch.arange(-half, half + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()

        # conv1d용 필터 weight 등록
        self.register_buffer('weight', kernel.view(1, 1, -1))  # (out_channels=1, in_channels=1, kernel_size)
        self.padding = half

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): shape (N,) or (batch_size, N)
        Returns:
            smoothed_x (torch.Tensor)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, N)

        smoothed = F.conv1d(x, self.weight, padding=self.padding)
        return smoothed.squeeze(0).squeeze(0)
# -------------------- 추가 --------------------


#def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):    
    model.to(device)
    model.eval()

    # -------------------- 추가 --------------------
    #smoother = GaussianSmoothing1D_AutoKernel(sigma=3.5).to(device)
    # -------------------- 추가 --------------------

    output_dir_global = "/data/fovert/repos/repos/OurVadCLIP/sample_plots"
    if not os.path.exists(output_dir_global):
        os.makedirs(output_dir_global)

    element_logits2_stack = []
    gt_len = 0
    

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            audio = item[1].squeeze(0)
            length = item[3]
            file_name = item[4]

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
                audio = audio.unsqueeze(0)

            visual = visual.to(device)
            audio = audio.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            text_features, logtis1, logits2, v_logits, a_logits, logits_av = model(visual, audio, padding_mask, prompt_text, lengths) # for Fine
            
            logits_av = logits_av.reshape(-1) 
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2]) # [B * 256, 7]
        
            prob_av = torch.sigmoid(logits_av[0:len_cur]) # [B*256, 7]
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1)) 

            if i == 0:
                ap_av = prob_av # 원본
                ap2 = prob2 # 원본
                #ap_av = smoother(prob_av) # 수정본
                #ap2 = smoother(prob2) # 수정본
            else:
                ap_av = torch.cat([ap_av, prob_av], dim=0) # 원본
                ap2 = torch.cat([ap2, prob2], dim=0) # 원본
                #ap_av = torch.cat([ap_av, smoother(prob_av)], dim=0) # 수정본
                #ap2 = torch.cat([ap2, smoother(prob2)], dim=0) # 수정본
            
            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)
    
            # visualization            
            intp_ap2 = np.repeat(prob2.cpu().numpy(), 16, 0)
            time_len = len(intp_ap2)
            slice_gt = gt[gt_len:gt_len + time_len]
            time_axis = np.arange(time_len)

            plt.figure(figsize=(12, 6))
            plt.fill_between(time_axis, 0, slice_gt, alpha=0.2, label='Ground Truth', color='red')
            plt.plot(time_axis, intp_ap2, label='Anomaly Score', color='blue', linewidth=3)
            plt.xlabel('Time Index', fontsize='xx-large')
            plt.ylabel('Anomaly Score', fontsize='xx-large')
            plt.ylim(0.0, 1.0) 
            plt.xticks(fontsize='large')  # y축 폰트 크기 설정
            plt.yticks(fontsize='xx-large')  # y축 폰트 크기 설정
            #plt.title(f"{file_name[0]}")
            #plt.legend(loc='upper left', fontsize='x-large')
            plt.tight_layout()

            save_path = os.path.join(output_dir_global, f"{file_name[0]}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()

            gt_len += time_len


        # 기존 플롯에서 사용된 색과 라벨을 정의
        legend_elements = [
            mpatches.Patch(color='red', alpha=0.2, label='Ground Truth'),
            Line2D([0], [0], color='blue', linewidth=3, label='Anomaly Score')
        ]
        # 새로운 Figure에 범례만 그림
        fig_legend = plt.figure(figsize=(4, 2))
        ax = fig_legend.add_subplot(111)
        ax.axis('off')  # 축 제거
        legend = ax.legend(handles=legend_elements, loc='center', fontsize='x-large')

        # 저장 경로 설정 및 저장
        legend_save_path = os.path.join(output_dir_global, f"legend.png")
        fig_legend.savefig(legend_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig_legend)

            
    ap_av = ap_av.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap_av = ap_av.tolist()
    ap2 = ap2.tolist()

    #sys.stdout.write(f'ap_av: {len(ap_av)}\n') # 145706
    #sys.stdout.write(f'ap2: {len(ap2)}\n') # 145706
    #sys.stdout.write(f'gt: {len(gt)}\n') # 2331296 = 145706 * 16
    #sys.stdout.flush()

    # logits_av를 사용한 분류 성능 계산
    ROC_av = roc_auc_score(gt, np.repeat(ap_av, 16))
    AP_av = average_precision_score(gt, np.repeat(ap_av, 16)) # y_true, y_score 
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    print("AUC (using logits_av): ", ROC_av, " AP (using logits_av): ", AP_av)
    print("AUC2: ", ROC2, " AP2:", AP2)

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))

    return ROC_av, AP2, averageMAP


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()

    label_map = dict({'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 
                      'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'})

    test_dataset = XDDataset(args.visual_length, args.test_list, args.test_audio_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, 
                    args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, 
                    args.prompt_postfix, args.audio_dim, device)
                   
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param['av_model_state_dict'])

    #model_param = torch.load(args.checkpoint_path) # for UCF
    #model.load_state_dict(model_param['model_state_dict']) # for UCF

    test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)



















'''
# xd_test.py
import torch
import sys
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import matplotlib.pyplot as plt  # 시각화를 위한 임포트
import os  # 파일 시스템 작업을 위한 임포트

from model import CLIPVAD
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option

def upsample_array(arr, target_length):
    """
    입력 배열(arr)을 target_length만큼 선형 보간(interpolation)하여 업샘플링합니다.
    """
    current_length = len(arr)
    if current_length == target_length:
        return arr
    orig_x = np.arange(current_length)
    new_x = np.linspace(0, current_length - 1, target_length)
    return np.interp(new_x, orig_x, arr)

def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device, visual_ds_name, audio_ds_name):
    
    model.to(device)
    model.eval()

    element_logits2_stack = []
    total_len = 0

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            audio = item[1].squeeze(0)
            length = item[3]

            length = int(length)
            len_cur = length
            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
                audio = audio.unsqueeze(0)

            visual = visual.to(device)
            audio = audio.to(device)

            lengths = torch.zeros(int(length / maxlen) + 1)
            for j in range(int(length / maxlen) + 1):
                if j == 0 and length < maxlen:
                    lengths[j] = length
                elif j == 0 and length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                elif length > maxlen:
                    lengths[j] = maxlen
                    length -= maxlen
                else:
                    lengths[j] = length
            lengths = lengths.to(int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            
            # 모델 출력: logits1 대신 logits_av를 사용하도록 수정
            text_features, logtis1, logits2, v_logits, a_logits, logits_av = model(
                visual, audio, padding_mask, prompt_text, lengths)
            # logits_av: [B, 256]
            # a_logits: [B, 256, 1]
            # v_logits: [B, 256, 1]
            # logits2: [B, 256, N]
                
            # logits_av의 shape를 1차원으로 reshape하여 확률 계산에 사용
            logits_av = logits_av.reshape(-1) # [256]
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2]) # [B * 256, N] -> [256, N]
            
            # logits_av를 sigmoid를 통해 이진 분류 확률로 변환
            prob_av = torch.sigmoid(logits_av[0:len_cur]) # [256]
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1)) # [256]

            if i == 0:
                ap_av = prob_av
                ap2 = prob2
            else:
                ap_av = torch.cat([ap_av, prob_av], dim=0)
                ap2 = torch.cat([ap2, prob2], dim=0)

            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy() # [256, N]
            element_logits2 = np.repeat(element_logits2, 16, 0) # [256 * 16, N]
            element_logits2_stack.append(element_logits2)
            
            # ------------------ 개별 데이터 샘플 시각화 ------------------
            # 각 모달리티별 프레임 단위 스코어 계산 (sigmoid 적용)
            prob_visual = torch.sigmoid(v_logits[0:len_cur]).cpu().numpy().squeeze()
            prob_audio = torch.sigmoid(a_logits[0:len_cur]).cpu().numpy().squeeze()
            prob_av_sample = prob2.cpu().numpy().squeeze()  # 결합된(visual–audio) 스코어

            # x축: visual 점수의 길이를 기준으로 함
            target_length = len(prob_visual)
            x_axis = range(target_length)

            ######################### gt_score 추가
            gt_score = gt[total_len:total_len + len_cur*16]
            total_len += len_cur*16
            sys.stdout.write(f'{i}th target_length: {target_length}\n')
            sys.stdout.write(f'{i}th total_len: {total_len}/{gt.shape[0]}\n')
            sys.stdout.write(f'{i}th gt_score: {gt_score.shape[0]}\n')
            sys.stdout.flush()

            if len(gt_score) != target_length:
                gt_score = upsample_array(gt_score, target_length)

            # 만약 다른 모달리티의 점수 배열 길이가 다르면 업샘플링 처리
            if len(prob_audio) != target_length:
                prob_audio = upsample_array(prob_audio, target_length)
            if len(prob_av_sample) != target_length:
                prob_av_sample = upsample_array(prob_av_sample, target_length)
            
            # ----- 각 샘플에 해당하는 영화 제목 추출 -----
            # DataLoader의 dataset 객체가 XDDataset라면 df에 접근 가능
            if hasattr(testdataloader.dataset, 'df'):
                # CSV 파일에서 해당 샘플의 'path' 열 값을 가져와서,
                # 파일명에서 첫 번째 '__' 이전 부분을 영화 제목으로 사용합니다.
                movie_full_path = testdataloader.dataset.df.iloc[i]['path']
                movie_title = os.path.basename(movie_full_path).split('__')[0]
            else:
                movie_title = f"Sample_{i}"
            # ------------------------------------------------

            plt.figure(figsize=(10, 5))
            plt.plot(x_axis, gt_score, label='gt Score')
            plt.plot(x_axis, prob_visual, label='Visual Score')
            plt.plot(x_axis, prob_audio, label='Audio Score')
            plt.plot(x_axis, prob_av_sample, label='Visual-Audio Score')
            plt.xlabel("Frame Index")
            plt.ylabel("Score")
            # 제목에 영화 제목과 데이터셋명을 포함합니다.
            plt.title(f"{movie_title} - Dataset: {visual_ds_name} & {audio_ds_name} - Sample {i}")
            plt.legend()

            # 결과 파일을 저장할 폴더 생성 (예: "sample_plots")
            output_dir = "sample_plots"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # 파일명에도 영화 제목을 포함시킵니다.
            save_filename = f"{movie_title}_sample_{i}_scores.png"
            plt.savefig(os.path.join(output_dir, save_filename), dpi=300)
            plt.close()
            # -------------------------------------------------------------
            
    ap_av = ap_av.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap_av = ap_av.tolist()
    ap2 = ap2.tolist()

    # logits_av를 사용한 분류 성능 계산
    ROC_av = roc_auc_score(gt, np.repeat(ap_av, 16))
    AP_av = average_precision_score(gt, np.repeat(ap_av, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    print("AUC (using logits_av): ", ROC_av, " AP (using logits_av): ", AP_av)
    print("AUC2: ", ROC2, " AP2:", AP2)

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} = {1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))
    
    # ------------------ 전체 점수 시각화 (전체 샘플 기반) ------------------
    repeated_ap_av = np.repeat(ap_av, 16)
    repeated_ap2 = np.repeat(ap2, 16)
    x_axis = np.arange(len(repeated_ap_av))

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, repeated_ap_av, label='Logits AV (sigmoid)')
    plt.plot(x_axis, repeated_ap2, label='Logits2 (softmax)')
    plt.xlabel('Time Index')
    plt.ylabel('Score')
    plt.title(f"Aggregated Logit Scores - Visual: {visual_ds_name}, Audio: {audio_ds_name}")
    plt.legend()
    plt.tight_layout()

    output_dir_global = "plots"
    if not os.path.exists(output_dir_global):
        os.makedirs(output_dir_global)
    save_path = os.path.join(output_dir_global, "logit_scores.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    # --------------------------------------------------------------------
    
    return ROC_av, AP2, averageMAP


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()

    label_map = {
        'A': 'normal', 
        'B1': 'fighting', 
        'B2': 'shooting', 
        'B4': 'riot', 
        'B5': 'abuse', 
        'B6': 'car accident', 
        'G': 'explosion'
    }

    test_dataset = XDDataset(args.visual_length, args.test_list, args.test_audio_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width, 
        args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, 
        args.prompt_postfix, args.audio_dim, device
    )
                   
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param['av_model_state_dict'])

    # 데이터셋 파일 경로에서 파일명을 추출하여 데이터셋 이름(제목)에 사용합니다.
    visual_ds_name = os.path.basename(args.test_list).split('.')[0]
    audio_ds_name = os.path.basename(args.test_audio_list).split('.')[0]

    test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device, visual_ds_name, audio_ds_name)

'''