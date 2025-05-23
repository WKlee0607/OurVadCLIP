import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.InfoNCE import InfoNCE

# CMAL - 합집합(logits_av기준)
'''
def CMAL(mmil_logits, audio_logits, visual_logits, seq_len, audio_rep, visual_rep):
    """
    - mmil_logits: [B, 256]
    - audio_logits: [B, 256, 1]
    - visual_logits: [B, 256, 1]

    """
    audio_abn = torch.zeros(0).cuda()  # tensor([])
    visual_abn = torch.zeros(0).cuda()  # tensor([])
    audio_bgd = torch.zeros(0).cuda()  # tensor([])
    visual_bgd = torch.zeros(0).cuda()  # tensor([])
    audio_nor = torch.zeros(0).cuda()
    visual_nor = torch.zeros(0).cuda()
    for i in range(mmil_logits.shape[0]):
        if mmil_logits[i] > 0.5:
            # extract all bottom-k index
            cur_visual_inverse_topk, cur_visual_inverse_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_audio_inverse_topk, cur_audio_inverse_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_all_inverse_topk_indices = torch.unique(torch.cat([cur_visual_inverse_topk_indices, cur_audio_inverse_topk_indices]))

            # extract all top-k index
            cur_visual_topk, cur_visual_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_audio_topk, cur_audio_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_all_topk_indices = torch.unique(torch.cat([cur_visual_topk_indices, cur_audio_topk_indices]))

            # get bottom-k features
            cur_visual_inverse_rep_topk = visual_rep[i][cur_all_inverse_topk_indices]
            cur_audio_inverse_rep_topk = audio_rep[i][cur_all_inverse_topk_indices]
            visual_bgd = torch.cat((visual_bgd, cur_visual_inverse_rep_topk), 0)
            audio_bgd = torch.cat((audio_bgd, cur_audio_inverse_rep_topk), 0)

            # get top-k features
            cur_visual_rep_topk = visual_rep[i][cur_all_topk_indices]
            cur_dim = cur_visual_rep_topk.size()
            cur_visual_rep_topk = torch.mean(cur_visual_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_abn = torch.cat((visual_abn, cur_visual_rep_topk), 0)

            cur_audio_rep_topk = audio_rep[i][cur_all_topk_indices]            
            cur_dim = cur_audio_rep_topk.size()
            cur_audio_rep_topk = torch.mean(cur_audio_rep_topk, 0, keepdim=True).expand(cur_dim)
            audio_abn = torch.cat((audio_abn, cur_audio_rep_topk), 0)
        else:
            # extract all bottom-k index
            cur_visual_inverse_topk, cur_visual_inverse_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_audio_inverse_topk, cur_audio_inverse_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_all_inverse_topk_indices = torch.unique(torch.cat([cur_visual_inverse_topk_indices, cur_audio_inverse_topk_indices]))
            
            # extract all top-k index
            cur_visual_topk, cur_visual_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_audio_topk, cur_audio_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_all_topk_indices = torch.unique(torch.cat([cur_visual_topk_indices, cur_audio_topk_indices]))

            # get bottom-k features
            cur_visual_inverse_rep_topk = visual_rep[i][cur_all_inverse_topk_indices]
            cur_audio_inverse_rep_topk = audio_rep[i][cur_all_inverse_topk_indices]
            visual_bgd = torch.cat((visual_bgd, cur_visual_inverse_rep_topk), 0)
            audio_bgd = torch.cat((audio_bgd, cur_audio_inverse_rep_topk), 0)
            
            # get top-k features
            cur_visual_rep_topk = visual_rep[i][cur_all_topk_indices]
            cur_audio_rep_topk = audio_rep[i][cur_all_topk_indices]
            audio_nor = torch.cat((audio_nor, cur_audio_rep_topk), 0)
            visual_nor = torch.cat((visual_nor, cur_visual_rep_topk), 0)
    cmals = InfoNCE(negative_mode='unpaired')
    if audio_nor.size(0) == 0 or audio_abn.size(0) == 0:
        return 0.0, 0.0, 0.0, 0.0
    else:
        loss_a2v_a2b = cmals(audio_abn, visual_abn, visual_bgd)
        loss_a2v_a2n = cmals(audio_abn, visual_abn, visual_nor)
        loss_v2a_a2b = cmals(visual_abn, audio_abn, audio_bgd)
        loss_v2a_a2n = cmals(visual_abn, audio_abn, audio_nor)
        return loss_a2v_a2b, loss_a2v_a2n, loss_v2a_a2b, loss_v2a_a2n
'''





# CMAL - 합집합(레이블 기준)
'''
def CMAL(labels, audio_logits, visual_logits, seq_len, audio_rep, visual_rep):
    """
    label 기준

    - labels : [B, 7]
    - logits_av: [B, 256] 
    - audio_logits: [B, 256]
    - visual_logits: [B, 256]
    - visual_rep, audio_rep: B, 256, 512]
    """
    audio_abn = torch.zeros(0).cuda()  # tensor([])
    visual_abn = torch.zeros(0).cuda()  # tensor([])
    audio_bgd = torch.zeros(0).cuda()  # tensor([])
    visual_bgd = torch.zeros(0).cuda()  # tensor([])
    audio_nor = torch.zeros(0).cuda()
    visual_nor = torch.zeros(0).cuda()
    for i in range(labels.shape[0]):
        if labels[i][0] == 0.: # Abnormal
            # extract all bottom-k index
            cur_visual_inverse_topk, cur_visual_inverse_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_audio_inverse_topk, cur_audio_inverse_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_all_inverse_topk_indices = torch.unique(torch.cat([cur_visual_inverse_topk_indices, cur_audio_inverse_topk_indices]))

            # extract all top-k index
            cur_visual_topk, cur_visual_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_audio_topk, cur_audio_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_all_topk_indices = torch.unique(torch.cat([cur_visual_topk_indices, cur_audio_topk_indices]))

            # get bottom-k features
            cur_visual_inverse_rep_topk = visual_rep[i][cur_all_inverse_topk_indices]
            cur_audio_inverse_rep_topk = audio_rep[i][cur_all_inverse_topk_indices]
            visual_bgd = torch.cat((visual_bgd, cur_visual_inverse_rep_topk), 0)
            audio_bgd = torch.cat((audio_bgd, cur_audio_inverse_rep_topk), 0)

            # get top-k features
            cur_visual_rep_topk = visual_rep[i][cur_all_topk_indices]
            cur_dim = cur_visual_rep_topk.size()
            cur_visual_rep_topk = torch.mean(cur_visual_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_abn = torch.cat((visual_abn, cur_visual_rep_topk), 0)

            cur_audio_rep_topk = audio_rep[i][cur_all_topk_indices]            
            cur_dim = cur_audio_rep_topk.size()
            cur_audio_rep_topk = torch.mean(cur_audio_rep_topk, 0, keepdim=True).expand(cur_dim)
            audio_abn = torch.cat((audio_abn, cur_audio_rep_topk), 0)
        else:
            # extract all bottom-k index
            cur_visual_inverse_topk, cur_visual_inverse_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_audio_inverse_topk, cur_audio_inverse_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_all_inverse_topk_indices = torch.unique(torch.cat([cur_visual_inverse_topk_indices, cur_audio_inverse_topk_indices]))
            
            # extract all top-k index
            cur_visual_topk, cur_visual_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_audio_topk, cur_audio_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_all_topk_indices = torch.unique(torch.cat([cur_visual_topk_indices, cur_audio_topk_indices]))

            # get bottom-k features
            cur_visual_inverse_rep_topk = visual_rep[i][cur_all_inverse_topk_indices]
            cur_audio_inverse_rep_topk = audio_rep[i][cur_all_inverse_topk_indices]
            visual_bgd = torch.cat((visual_bgd, cur_visual_inverse_rep_topk), 0)
            audio_bgd = torch.cat((audio_bgd, cur_audio_inverse_rep_topk), 0)
            
            # get top-k features
            cur_visual_rep_topk = visual_rep[i][cur_all_topk_indices]
            cur_audio_rep_topk = audio_rep[i][cur_all_topk_indices]
            audio_nor = torch.cat((audio_nor, cur_audio_rep_topk), 0)
            visual_nor = torch.cat((visual_nor, cur_visual_rep_topk), 0)
    cmals = InfoNCE(negative_mode='unpaired')
    if audio_nor.size(0) == 0 or audio_abn.size(0) == 0:
        return 0.0, 0.0, 0.0, 0.0
    else:
        loss_a2v_a2b = cmals(audio_abn, visual_abn, visual_bgd) # audio_abn: [All-K, 512] ex: [675, 512], [654, 512],....
        loss_a2v_a2n = cmals(audio_abn, visual_abn, visual_nor)
        loss_v2a_a2b = cmals(visual_abn, audio_abn, audio_bgd)
        loss_v2a_a2n = cmals(visual_abn, audio_abn, audio_nor)

        # nor-bgd 비슷하도록 학습.
        return loss_a2v_a2b, loss_a2v_a2n, loss_v2a_a2b, loss_v2a_a2n
'''



'''
# CMAL 수정 2
def CMAL(sample_level_preds, audio_logits, visual_logits, seq_len, audio_rep, visual_rep):
    """
    - sample_level_preds : [B, 256]
    - logits_av: [B, 256] 
    - audio_logits: [B, 256]
    - visual_logits: [B, 256]
    - visual_rep, audio_rep: B, 256, 512]
    """
    audio_abn = torch.zeros(0).cuda()  # tensor([])
    visual_abn = torch.zeros(0).cuda()  # tensor([])
    audio_bgd = torch.zeros(0).cuda()  # tensor([])
    visual_bgd = torch.zeros(0).cuda()  # tensor([])
    
    audio_nor = torch.zeros(0).cuda()
    visual_nor = torch.zeros(0).cuda()
    audio_nor_bgd = torch.zeros(0).cuda()  # tensor([])
    visual_nor_bgd = torch.zeros(0).cuda()  # tensor([])

    for i in range(sample_level_preds.shape[0]):
        #if labels[i][0] == 0.: # Abnormal
        if sample_level_preds[i] > 0.5:
            cur_visual_inverse_topk, cur_visual_inverse_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_visual_inverse_rep_topk = visual_rep[i][cur_visual_inverse_topk_indices]
            # cur_dim = cur_visual_inverse_rep_topk.size()
            # cur_visual_inverse_rep_topk = torch.mean(cur_visual_inverse_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_bgd = torch.cat((visual_bgd, cur_visual_inverse_rep_topk), 0)

            cur_audio_inverse_topk, cur_audio_inverse_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_audio_inverse_rep_topk = audio_rep[i][cur_audio_inverse_topk_indices]
            # cur_dim = cur_audio_inverse_rep_topk.size()
            # cur_audio_inverse_rep_topk = torch.mean(cur_audio_inverse_rep_topk, 0, keepdim=True).expand(cur_dim)
            audio_bgd = torch.cat((audio_bgd, cur_audio_inverse_rep_topk), 0)

            cur_audio_topk, cur_audio_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_audio_rep_topk = audio_rep[i][cur_audio_topk_indices]
            cur_dim = cur_audio_rep_topk.size()
            cur_audio_rep_topk = torch.mean(cur_audio_rep_topk, 0, keepdim=True).expand(cur_dim)
            audio_abn = torch.cat((audio_abn, cur_audio_rep_topk), 0)

            cur_visual_topk, cur_visual_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_visual_rep_topk = visual_rep[i][cur_visual_topk_indices]
            cur_dim = cur_visual_rep_topk.size()
            cur_visual_rep_topk = torch.mean(cur_visual_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_abn = torch.cat((visual_abn, cur_visual_rep_topk), 0)
        else:
            cur_visual_inverse_topk, cur_visual_inverse_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_visual_inverse_rep_topk = visual_rep[i][cur_visual_inverse_topk_indices]
            # cur_dim = cur_visual_inverse_rep_topk.size()
            # cur_visual_inverse_rep_topk = torch.mean(cur_visual_inverse_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_nor_bgd = torch.cat((visual_nor_bgd, cur_visual_inverse_rep_topk), 0)

            cur_audio_inverse_topk, cur_audio_inverse_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_audio_inverse_rep_topk = audio_rep[i][cur_audio_inverse_topk_indices]
            # cur_dim = cur_audio_inverse_rep_topk.size()
            # cur_audio_inverse_rep_topk = torch.mean(cur_audio_inverse_rep_topk, 0, keepdim=True).expand(cur_dim)
            audio_nor_bgd = torch.cat((audio_nor_bgd, cur_audio_inverse_rep_topk), 0)

            cur_audio_topk, cur_audio_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_audio_rep_topk = audio_rep[i][cur_audio_topk_indices]
            # cur_dim = cur_audio_rep_topk.size()
            # cur_audio_rep_topk = torch.mean(cur_audio_rep_topk, 0, keepdim=True).expand(cur_dim)
            audio_nor = torch.cat((audio_nor, cur_audio_rep_topk), 0)

            cur_visual_topk, cur_visual_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_visual_rep_topk = visual_rep[i][cur_visual_topk_indices]
            # cur_dim = cur_visual_rep_topk.size()
            # cur_visual_rep_topk = torch.mean(cur_visual_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_nor = torch.cat((visual_nor, cur_visual_rep_topk), 0)
    cmals = InfoNCE(negative_mode='unpaired')
    if audio_nor.size(0) == 0 or audio_abn.size(0) == 0:
        return 0.0, 0.0, 0.0, 0.0
    else:
        loss_a2v_a2b = cmals(audio_abn, visual_abn, visual_bgd) # audio_abn: [All-K, 512] ex: [675, 512], [654, 512],....
        loss_v2a_a2b = cmals(visual_abn, audio_abn, audio_bgd)
        loss_a = cmals(audio_nor, visual_nor_bgd, visual_abn) # 수정
        loss_b = cmals(visual_nor, audio_nor_bgd, audio_abn) # 수정
        return loss_a2v_a2b, loss_a, loss_v2a_a2b, loss_b
'''


# CMAL 수정 1 

def CMAL(labels, audio_logits, visual_logits, seq_len, audio_rep, visual_rep):
    """
    label 기준

    - labels : [B, 7]
    - logits_av: [B, 256] 
    - audio_logits: [B, 256]
    - visual_logits: [B, 256]
    - visual_rep, audio_rep: B, 256, 512]
    """
    audio_abn = torch.zeros(0).cuda()  # tensor([])
    visual_abn = torch.zeros(0).cuda()  # tensor([])
    audio_bgd = torch.zeros(0).cuda()  # tensor([])
    visual_bgd = torch.zeros(0).cuda()  # tensor([])
    audio_nor = torch.zeros(0).cuda()
    visual_nor = torch.zeros(0).cuda()
    for i in range(labels.shape[0]):
        if labels[i][0] == 0.: # Abnormal
            cur_visual_inverse_topk, cur_visual_inverse_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_visual_inverse_rep_topk = visual_rep[i][cur_visual_inverse_topk_indices]
            # cur_dim = cur_visual_inverse_rep_topk.size()
            # cur_visual_inverse_rep_topk = torch.mean(cur_visual_inverse_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_bgd = torch.cat((visual_bgd, cur_visual_inverse_rep_topk), 0)

            cur_audio_inverse_topk, cur_audio_inverse_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_audio_inverse_rep_topk = audio_rep[i][cur_audio_inverse_topk_indices]
            # cur_dim = cur_audio_inverse_rep_topk.size()
            # cur_audio_inverse_rep_topk = torch.mean(cur_audio_inverse_rep_topk, 0, keepdim=True).expand(cur_dim)
            audio_bgd = torch.cat((audio_bgd, cur_audio_inverse_rep_topk), 0)

            cur_audio_topk, cur_audio_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_audio_rep_topk = audio_rep[i][cur_audio_topk_indices]
            cur_dim = cur_audio_rep_topk.size()
            cur_audio_rep_topk = torch.mean(cur_audio_rep_topk, 0, keepdim=True).expand(cur_dim)
            audio_abn = torch.cat((audio_abn, cur_audio_rep_topk), 0)

            cur_visual_topk, cur_visual_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_visual_rep_topk = visual_rep[i][cur_visual_topk_indices]
            cur_dim = cur_visual_rep_topk.size()
            cur_visual_rep_topk = torch.mean(cur_visual_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_abn = torch.cat((visual_abn, cur_visual_rep_topk), 0)
        else:
            cur_visual_inverse_topk, cur_visual_inverse_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_visual_inverse_rep_topk = visual_rep[i][cur_visual_inverse_topk_indices]
            # cur_dim = cur_visual_inverse_rep_topk.size()
            # cur_visual_inverse_rep_topk = torch.mean(cur_visual_inverse_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_bgd = torch.cat((visual_bgd, cur_visual_inverse_rep_topk), 0)

            cur_audio_inverse_topk, cur_audio_inverse_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_audio_inverse_rep_topk = audio_rep[i][cur_audio_inverse_topk_indices]
            # cur_dim = cur_audio_inverse_rep_topk.size()
            # cur_audio_inverse_rep_topk = torch.mean(cur_audio_inverse_rep_topk, 0, keepdim=True).expand(cur_dim)
            audio_bgd = torch.cat((audio_bgd, cur_audio_inverse_rep_topk), 0)

            cur_audio_topk, cur_audio_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_audio_rep_topk = audio_rep[i][cur_audio_topk_indices]
            # cur_dim = cur_audio_rep_topk.size()
            # cur_audio_rep_topk = torch.mean(cur_audio_rep_topk, 0, keepdim=True).expand(cur_dim)
            audio_nor = torch.cat((audio_nor, cur_audio_rep_topk), 0)

            cur_visual_topk, cur_visual_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_visual_rep_topk = visual_rep[i][cur_visual_topk_indices]
            # cur_dim = cur_visual_rep_topk.size()
            # cur_visual_rep_topk = torch.mean(cur_visual_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_nor = torch.cat((visual_nor, cur_visual_rep_topk), 0)
    cmals = InfoNCE(negative_mode='unpaired')
    if audio_nor.size(0) == 0 or audio_abn.size(0) == 0:
        return 0.0, 0.0, 0.0, 0.0
    else:
        loss_a2v_a2b = cmals(audio_abn, visual_abn, visual_bgd) # audio_abn: [All-K, 512] ex: [675, 512], [654, 512],....
        loss_a2v_a2n = cmals(audio_abn, visual_abn, visual_nor)
        loss_v2a_a2b = cmals(visual_abn, audio_abn, audio_bgd)
        loss_v2a_a2n = cmals(visual_abn, audio_abn, audio_nor)

        # nor-bgd 비슷하도록 학습.
        return loss_a2v_a2b, loss_a2v_a2n, loss_v2a_a2b, loss_v2a_a2n





# 원본
'''
def CMAL(mmil_logits, audio_logits, visual_logits, seq_len, audio_rep, visual_rep):
    """
    - mmil_logits: [B, 256]
    - audio_logits: [B, 256, 1]
    - visual_logits: [B, 256, 1]

    """

    audio_abn = torch.zeros(0).cuda()  # tensor([])
    visual_abn = torch.zeros(0).cuda()  # tensor([])
    audio_bgd = torch.zeros(0).cuda()  # tensor([])
    visual_bgd = torch.zeros(0).cuda()  # tensor([])
    audio_nor = torch.zeros(0).cuda()
    visual_nor = torch.zeros(0).cuda()
    for i in range(mmil_logits.shape[0]):
        if mmil_logits[i] > 0.5:
            cur_visual_inverse_topk, cur_visual_inverse_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_visual_inverse_rep_topk = visual_rep[i][cur_visual_inverse_topk_indices]
            # cur_dim = cur_visual_inverse_rep_topk.size()
            # cur_visual_inverse_rep_topk = torch.mean(cur_visual_inverse_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_bgd = torch.cat((visual_bgd, cur_visual_inverse_rep_topk), 0)

            cur_audio_inverse_topk, cur_audio_inverse_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_audio_inverse_rep_topk = audio_rep[i][cur_audio_inverse_topk_indices]
            # cur_dim = cur_audio_inverse_rep_topk.size()
            # cur_audio_inverse_rep_topk = torch.mean(cur_audio_inverse_rep_topk, 0, keepdim=True).expand(cur_dim)
            audio_bgd = torch.cat((audio_bgd, cur_audio_inverse_rep_topk), 0)

            cur_audio_topk, cur_audio_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_audio_rep_topk = audio_rep[i][cur_audio_topk_indices]
            cur_dim = cur_audio_rep_topk.size()
            cur_audio_rep_topk = torch.mean(cur_audio_rep_topk, 0, keepdim=True).expand(cur_dim)
            audio_abn = torch.cat((audio_abn, cur_audio_rep_topk), 0)

            cur_visual_topk, cur_visual_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_visual_rep_topk = visual_rep[i][cur_visual_topk_indices]
            cur_dim = cur_visual_rep_topk.size()
            cur_visual_rep_topk = torch.mean(cur_visual_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_abn = torch.cat((visual_abn, cur_visual_rep_topk), 0)
        else:
            cur_visual_inverse_topk, cur_visual_inverse_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_visual_inverse_rep_topk = visual_rep[i][cur_visual_inverse_topk_indices]
            # cur_dim = cur_visual_inverse_rep_topk.size()
            # cur_visual_inverse_rep_topk = torch.mean(cur_visual_inverse_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_bgd = torch.cat((visual_bgd, cur_visual_inverse_rep_topk), 0)

            cur_audio_inverse_topk, cur_audio_inverse_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=False)
            cur_audio_inverse_rep_topk = audio_rep[i][cur_audio_inverse_topk_indices]
            # cur_dim = cur_audio_inverse_rep_topk.size()
            # cur_audio_inverse_rep_topk = torch.mean(cur_audio_inverse_rep_topk, 0, keepdim=True).expand(cur_dim)
            audio_bgd = torch.cat((audio_bgd, cur_audio_inverse_rep_topk), 0)

            cur_audio_topk, cur_audio_topk_indices = torch.topk(audio_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_audio_rep_topk = audio_rep[i][cur_audio_topk_indices]
            # cur_dim = cur_audio_rep_topk.size()
            # cur_audio_rep_topk = torch.mean(cur_audio_rep_topk, 0, keepdim=True).expand(cur_dim)
            audio_nor = torch.cat((audio_nor, cur_audio_rep_topk), 0)

            cur_visual_topk, cur_visual_topk_indices = torch.topk(visual_logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            cur_visual_rep_topk = visual_rep[i][cur_visual_topk_indices]
            # cur_dim = cur_visual_rep_topk.size()
            # cur_visual_rep_topk = torch.mean(cur_visual_rep_topk, 0, keepdim=True).expand(cur_dim)
            visual_nor = torch.cat((visual_nor, cur_visual_rep_topk), 0)
    cmals = InfoNCE(negative_mode='unpaired')
    if audio_nor.size(0) == 0 or audio_abn.size(0) == 0:
        return 0.0, 0.0, 0.0, 0.0
    else:
        loss_a2v_a2b = cmals(audio_abn, visual_abn, visual_bgd)
        loss_a2v_a2n = cmals(audio_abn, visual_abn, visual_nor)
        loss_v2a_a2b = cmals(visual_abn, audio_abn, audio_bgd)
        loss_v2a_a2n = cmals(visual_abn, audio_abn, audio_nor)
        return loss_a2v_a2b, loss_a2v_a2n, loss_v2a_a2b, loss_v2a_a2n
'''