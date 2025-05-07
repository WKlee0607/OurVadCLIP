# xd_test.py
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import CLIPVAD
from utils.dataset import CCTVDataset
from utils.tools import get_batch_mask, get_prompt_text
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import cctv_option

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
def test(model, testdataloader, maxlen, prompt_text, gt, device):    
    model.to(device)
    model.eval()

    # -------------------- 추가 --------------------
    #smoother = GaussianSmoothing1D_AutoKernel(sigma=3.5).to(device)
    # -------------------- 추가 --------------------

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
            #import sys
            #sys.stdout.write(f"v: {visual.shape}\n")
            #sys.stdout.write(f"a: {audio.shape}\n")
            #sys.stdout.flush()

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

    ap_av = ap_av.cpu().numpy()
    ap2 = ap2.cpu().numpy()
    ap_av = ap_av.tolist()
    ap2 = ap2.tolist()

    # logits_av를 사용한 분류 성능 계산
    clip_len = 4
    ROC_av = roc_auc_score(gt, np.repeat(ap_av, clip_len))
    AP_av = average_precision_score(gt, np.repeat(ap_av, clip_len)) # y_true, y_score 
    ROC2 = roc_auc_score(gt, np.repeat(ap2, clip_len))
    AP2 = average_precision_score(gt, np.repeat(ap2, clip_len))

    print("AUC (using logits_av): ", ROC_av, " AP (using logits_av): ", AP_av)
    print("AUC2: ", ROC2, " AP2:", AP2)

    return ROC_av, AP2


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = cctv_option.parser.parse_args()

    test_dataset = CCTVDataset(args.visual_length, args.test_path, args.test_audio_path, "list/cctv_groundtruth.json", "testing")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = ['Normal', 'Fight']
    gt = np.load(args.gt_path)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, 
                    args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, 
                    args.prompt_postfix, args.audio_dim, device)
                   
    model_param = torch.load(args.model_path)
    model.load_state_dict(model_param['av_model_state_dict'])

    #model_param = torch.load(args.checkpoint_path) # for UCF
    #model.load_state_dict(model_param['model_state_dict']) # for UCF

    test(model, test_loader, args.visual_length, prompt_text, gt, device)




