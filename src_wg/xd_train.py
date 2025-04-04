import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import random
import os
import datetime
import sys

from model import CLIPVAD
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_prompt_text, get_batch_label, cosine_scheduler
import xd_option
import tqdm

"""
def loss_vf(img_token:torch.Tensor, aud_token:torch.Tensor, beta: float = 1 / 0.07, **kwargs) -> torch.Tensor:
    '''
    img_embs : [B, 768]
    audio_embs : [B, 768]
    '''
    B = img_token.shape[0]

    sim1 = torch.einsum('nc,mc->nm', F.normalize(img_token, dim=-1), F.normalize(aud_token, dim=-1)) * beta # [B, B]
    sim2 = torch.einsum('nc,mc->nm', F.normalize(aud_token, dim=-1), F.normalize(img_token, dim=-1)) * beta # [B, B]
    labels = torch.arange(B).long().to(img_token.device)

    loss = 0.5 * (F.cross_entropy(sim1, labels) + F.cross_entropy(sim2, labels))
    return loss
"""

"""
def Distll(logits_visual, logits_audio, logits_av, temperature):
    logits_av = logits_av.unsqueeze(-1) # [B, 256, 1]
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    source_audio_student = F.log_softmax(logits_audio/temperature, dim=1)
    target_audio_student = F.softmax(logits_audio/temperature, dim=1)
    source_visual_student = F.log_softmax(logits_visual/temperature, dim=1)
    target_visual_student = F.softmax(logits_visual/temperature, dim=1)
    source_teacher = F.log_softmax(logits_av/temperature, dim=1)
    target_teacher = F.softmax(logits_av/temperature, dim=1)

    distill_loss = kl_loss(source_visual_student, target_teacher) + kl_loss(source_audio_student, target_teacher) + kl_loss(source_teacher, target_visual_student)# + kl_loss(source_teacher, target_audio_student)
    return distill_loss
"""

def Distll(logits_visual, logits_audio, temperature):
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    source_audio_student = F.log_softmax(logits_audio/temperature, dim=1)
    target_visual_student = F.softmax(logits_visual/temperature, dim=1)

    return kl_loss(source_audio_student, target_visual_student)


def update_ema(student: nn.Module, ema_model: nn.Module, decay: float):
    """
    학생 모델(student)의 파라미터로 EMA 모델(ema_model)을 업데이트합니다.
    decay는 EMA decay 값 (예: 0.99)
    """
    with torch.no_grad():
        student_params = dict(student.named_parameters())
        for name, ema_param in ema_model.named_parameters():
            student_param = student_params[name]
            ema_param.data.mul_(decay).add_(student_param.data, alpha=1 - decay)


def CLASM(logits, labels, lengths, device): # fine
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device): # coarse
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))
    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def train(model, train_loader, test_loader, args, label_map: dict, device, rank):
    model.to(device)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer_av = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler_av = MultiStepLR(optimizer_av, args.scheduler_milestones, args.scheduler_rate)
    momentum_schedule = cosine_scheduler(base_value=args.m, final_value=1.0, epochs=args.max_epoch, niter_per_ep=len(train_loader))

    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_av.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("Checkpoint info:")
        print("Epoch:", epoch+1, " AP:", ap_best)

    for e in range(epoch, args.max_epoch):
        model.train()
        loss_total_coarse = 0
        loss_total_fine = 0
        
        for i, item in enumerate(train_loader):
        #for item in tqdm(train_loader, desc=f"Extracting features in {rank}..."):
            it = len(train_loader) * e + i  # global training iteration
            step = i * train_loader.batch_size
            visual_feat, audio_feat, text_labels, feat_lengths = item
            visual_feat = visual_feat.to(device) # [B, 256, 512]
            audio_feat = audio_feat.to(device) # [B, 256, 128]
            feat_lengths = feat_lengths.to(device)
            # text_labels를 dataset에서 그대로 사용 (get_prompt_text 대신)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            # Audio-visual model forward
            #text_features, logits_coarse, logits_fine, logits_visual, logits_audio, logits_av = model(visual_feat, audio_feat, None, prompt_text, feat_lengths) # logits_av -> 이거 사용해야 함!!
            #text_features, logits_coarse, logits_fine, logits_visual, logits_audio = model(visual_feat, audio_feat, None, prompt_text, feat_lengths) 
            #text_features, logits_coarse, logits_fine, logits_visual, logits_audio, visual_features, audio_features = model(visual_feat, audio_feat, None, prompt_text, feat_lengths) 
            text_features, logits_coarse, logits_fine, logits_visual, logits_audio, logits_av = model(visual_feat, audio_feat, None, prompt_text, feat_lengths) 


            #sys.stdout.write(f"visual_features.shape: {visual_features.shape}") # [B, 256, 512]
            #sys.stdout.write(f"audio_features.shape: {audio_features.shape}") # [B, 256, 512]

            #print(f"logits_visual.shape: {logits_visual.shape}") # [B, 256, 1]
            #print(f"logits_audio.shape: {logits_audio.shape}") # [B, 256, 1]
            #print(f"logits_av.shape: {logits_av.shape}") # [B, 256]

            ## Coarse Loss
            loss1 = CLAS2(logits_coarse, text_labels, feat_lengths, device) # audio-visual
            loss_total_coarse += loss1.item()

            ## Fine Loss
            loss2 = CLASM(logits_fine, text_labels, feat_lengths, device)
            loss_total_fine += loss2.item()

            ## Other Loss
            loss3 = torch.zeros(1, dtype=torch.float32).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 6.0
            
            ## Self-Distillation Loss
            #distill_loss = Distll(logits_visual, logits_audio, logits_av, args.temperature)
            #distill_loss = Distll(logits_visual, logits_av, args.temperature)

            ## Params Update 
            #loss_all = loss1 + loss2 + loss3 * 1e-4 + distill_loss * 2e-1
            #loss_all = loss1 + loss2 + loss3 + loss4 + loss5 * 1e-4 + distill_loss * 2e-1
            loss4 = Distll(logits_visual, logits_av, args.temperature)
            loss_all = loss1 + loss2 + loss3 * 1e-4 + loss4 * 2e-1
            optimizer_av.zero_grad()
            loss_all.backward()
            optimizer_av.step()

            ## EMA Update
            m = momentum_schedule[it]
            # ver4
            update_ema(student=model.av_classifier, ema_model=model.visual_classifier, decay=m)
            update_ema(student=model.fuse_LGT, ema_model=model.video_LGT, decay=m)
            
            if step % 4800 == 0 and step != 0:
                sys.stdout.write(f"Epoch {e+1}, Step {step}:")
                sys.stdout.write("  Coarse: {:.4f}, Fine Loss: {:.4f}, Loss3: {:.4f}".format(
                    loss_total_coarse/(i+1), loss_total_fine/(i+1), loss3.item()))
                
                # 중간 디버깅 평가: 테스트 데이터셋으로 AUC, AP, mAP 출력
                print("  --> Running mid-epoch evaluation ...")
                auc, ap, mAP = test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                sys.stdout.write(f"      [Mid-Epoch] AUC: {auc:.4f}, AP: {ap:.4f}, mAP: {mAP:.4f}")
                sys.stdout.flush()
                if ap > ap_best:
                    ap_best = ap 
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer_av.state_dict(),
                        'ap': ap_best
                    }
                    torch.save(checkpoint, args.checkpoint_path)
        
        dist.barrier() if use_ddp else None # Distributed GPU
        scheduler_av.step()

        # Epoch 종료 후 평가
        #if (use_ddp and rank == 0) or (rank==None):
        auc, ap, mAP = test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
        if ap > ap_best:
            ap_best = ap 
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_av.state_dict(),
                'ap': ap_best
            }
        torch.save(checkpoint, args.checkpoint_path)
        sys.stdout.write(f"Epoch {e+1}: New best AP = {ap_best:.4f}")
        sys.stdout.flush()
        #print(f"Epoch {e+1}: New best AP = {ap_best:.4f}")
        

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)
    print(f"Best model saved with AP: {ap_best:.4f}")
    dist.destroy_process_group() if use_ddp else None

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # Distributed GPU
    USE_CUDA = torch.cuda.is_available()
    num_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    use_ddp = True if num_gpus > 1 else False
    rank = 0 if not use_ddp else None
    if use_ddp:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=9000))
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        world_size = dist.get_world_size()
        print(f'Rank: {rank}, World size: {world_size}') if rank == 0 else None

    #device = torch.cuda.current_device() if USE_CUDA else torch.device('cpu')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = {
        'A': 'normal',
        'B1': 'fighting',
        'B2': 'shooting',
        'B4': 'riot',
        'B5': 'abuse',
        'B6': 'car accident',
        'G': 'explosion'
    }

    train_dataset = XDDataset(args.visual_length, args.train_list, args.audio_list, False, label_map)
    #sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else train_dataset
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = XDDataset(args.visual_length, args.test_list, args.test_audio_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, 
                       args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, 
                       args.prompt_postfix, args.audio_dim, True, device)
    
    train(model, train_loader, test_loader, args, label_map, device, rank)
