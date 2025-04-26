import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.ndimage import gaussian_filter1d

from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import ucf_option

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



def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels, device):
    model.to(device)
    model.eval()

    # -------------------- 추가 --------------------
    smoother = GaussianSmoothing1D_AutoKernel(sigma=5.0).to(device)
    # -------------------- 추가 --------------------

    element_logits2_stack = []
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

            #_, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
            text_features, logtis1, logits2, v_logits, a_logits, logits_av = model(visual, audio, None, prompt_text, lengths)

            logits_av = logits_av.unsqueeze(-1)
            logits1 = logits_av.reshape(logits_av.shape[0] * logits_av.shape[1], logits_av.shape[2])
            logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
            prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1))
            prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            if i == 0:
                #ap1 = prob1 # 원본
                #ap2 = prob2 # 원본
                ap1 = smoother(prob1) # smoothing
                ap2 = smoother(prob2) # smoothing
            else:
                #ap1 = torch.cat([ap1, prob1], dim=0) # 원본
                #ap2 = torch.cat([ap2, prob2], dim=0) # 원본
                ap1 = torch.cat([ap1, smoother(prob1)], dim=0) # smoothing
                ap2 = torch.cat([ap2, smoother(prob2)], dim=0) # smoothing


            element_logits2 = logits2[0:len_cur].softmax(dim=-1).detach().cpu().numpy()
            element_logits2 = np.repeat(element_logits2, 16, 0)
            element_logits2_stack.append(element_logits2)

    ap1 = ap1.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap1 = ap1.tolist()
    ap2 = ap2.tolist()

    ROC1 = roc_auc_score(gt, np.repeat(ap1, 16))
    AP1 = average_precision_score(gt, np.repeat(ap1, 16))
    ROC2 = roc_auc_score(gt, np.repeat(ap2, 16))
    AP2 = average_precision_score(gt, np.repeat(ap2, 16))

    print("AUC1: ", ROC1, " AP1: ", AP1)
    print("AUC2: ", ROC2, " AP2:", AP2)

    dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
    averageMAP = 0
    for i in range(5):
        print('mAP@{0:.1f} ={1:.2f}%'.format(iou[i], dmap[i]))
        averageMAP += dmap[i]
    averageMAP = averageMAP/(i+1)
    print('average MAP: {:.2f}'.format(averageMAP))

    return ROC2, AP1

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()

    label_map = dict({'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson', 'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion', 'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery', 'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing', 'Vandalism': 'Vandalism'})

    testdataset = UCFDataset(args.visual_length, args.test_list, args.test_audio_list, True, label_map)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, 
                    args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, 
                    args.prompt_postfix, args.audio_dim, device)
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param['av_model_state_dict']) # for XD

    test(model, testdataloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)