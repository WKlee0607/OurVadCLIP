import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
import sys

from model import CLIPVAD
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
from utils.CMA_MIL import CMAL
import ucf_option

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

def DISTILL(logits_target, logits_source, temperature):
    """
    - logits_target: [B, 256]
    - logits_source: [B, 256]
    """
    
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    source_audio_student = F.log_softmax(logits_source/temperature, dim=1)
    target_visual_student = F.softmax(logits_target/temperature, dim=1)
    #source_audio_student = torch.log(torch.clamp(F.softmax(logits_source/temperature, dim=1), 1e-6, 1))
    #target_visual_student = torch.clamp(F.softmax(logits_target / temperature, dim=1), 1e-6, 1)
    return kl_loss(source_audio_student, target_visual_student)

def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        sys.stdout.write(f"checkpoint info: \n")
        sys.stdout.write(f"epoch: {epoch+1},  ap: {ap_best}\n")
        sys.stdout.flush()

    for e in range(args.max_epoch):
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_total_cmal = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            #normal_features, normal_label, normal_lengths = next(normal_iter)
            #anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            normal_visual_feat, normal_audio_feat, normal_text_labels, normal_feat_lengths = next(normal_iter)
            abnormal_visual_feat, abnormal_audio_feat, abnormal_text_labels, abnormal_feat_lengths = next(anomaly_iter)

            visual_feat = torch.cat([normal_visual_feat, abnormal_visual_feat], dim=0).to(device) # [B, 256, 512]
            audio_feat = torch.cat([normal_audio_feat, abnormal_audio_feat], dim=0).to(device) # [B, 256, 128]
            text_labels = list(normal_text_labels) + list(abnormal_text_labels)
            feat_lengths = torch.cat([normal_feat_lengths, abnormal_feat_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)
            
            # 추가
            text_features, logtis1, logits2, v_logits, a_logits, logits_av = model(visual_feat, audio_feat, None, prompt_text, feat_lengths)

            # 추가
            loss1 = CLAS2(logits_av, text_labels, feat_lengths, device) # Coarse
            loss_total1 += loss1.item()

            loss2 = CLASM(logits2, text_labels, feat_lengths, device) # Fine
            loss_total2 += loss2.item()

            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13 * 1e-1

            # 추가
            loss4_1 = DISTILL(v_logits.squeeze(-1), a_logits.squeeze(-1), 3.0) # Chain-each branches
            loss4_2 = DISTILL(a_logits.squeeze(-1), v_logits.squeeze(-1), 3.0) # Chain-each branches
            loss5 = DISTILL(logits_av, v_logits.squeeze(-1), 3.0) # Distill_av_to_v
            loss6 = CLAS2(a_logits.squeeze(-1), text_labels, feat_lengths, device) # BCE_Audio

            added_loss = loss6 + loss4_1 + loss4_2 + loss5 #* 5e-1

            """
            # CMAL loss
            # CMAL 수정 ------------------------------------------------------------------------
            visual_features, audio_features = model.encode_video(visual_feat, audio_feat, None, feat_lengths)
            loss_a2v_a2b, loss_a2v_a2n, loss_v2a_a2b, loss_v2a_a2n = CMAL(
                text_labels, 
                a_logits.squeeze(-1), 
                v_logits.squeeze(-1), 
                feat_lengths, 
                audio_features, 
                visual_features
            )
            # CMAL 수정 ------------------------------------------------------------------------
            
            # CMAL 손실 합산 (가중치는 실험에 따라 조정)
            cmal_loss = (loss_a2v_a2b + loss_a2v_a2n + loss_v2a_a2b + loss_v2a_a2n) * 0.25 
            
            # 수정: cmal_loss가 tensor인지 float인지 확인하고 처리
            if isinstance(cmal_loss, torch.Tensor):
                loss_total_cmal += cmal_loss.item()
            else:
                loss_total_cmal += cmal_loss  # float인 경우 직접 더함
            """
            
            # 추가
            loss = loss1 + loss2 + loss3 * 1e-4 + added_loss #+ cmal_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += i * normal_loader.batch_size * 2

            if step % 1280 == 0 and step != 0:
                #print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item(), ' | cmal', cmal_loss.item())
                print('epoch: ', e+1, '| step: ', step, '| loss1: ', loss_total1 / (i+1), '| loss2: ', loss_total2 / (i+1), '| loss3: ', loss3.item())
                AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
                AP = AUC
                sys.stdout.write(f"[Mid-Epoch:{e+1}] AUC: {AUC:.4f}, AP: {AP:.4f}\n")
                sys.stdout.flush()

                if AP > ap_best:
                    ap_best = AP 
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'ap': ap_best}
                    torch.save(checkpoint, args.checkpoint_path)
                    sys.stdout.write(f"-----Current Best Epoch:{e+1}, AUC: {AUC:.4f}, AP: {AP:.4f}-----\n")
                    sys.stdout.flush()
                
        scheduler.step()

        # 추가
        AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, device)
        if AP > ap_best:
            ap_best = AP 
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ap': ap_best}
            torch.save(checkpoint, args.checkpoint_path)
            sys.stdout.write(f"-----Current Best Epoch:{e+1}, AUC: {AUC:.4f}, AP: {AP:.4f}-----\n")
            sys.stdout.flush()
        
            torch.save(checkpoint, args.checkpoint_path)
            #checkpoint = torch.load(args.checkpoint_path)
            #model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    # train
    normal_dataset = UCFDataset(args.visual_length, args.train_list, args.audio_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, args.audio_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # test
    test_dataset = UCFDataset(args.visual_length, args.test_list, args.test_audio_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, 
                    args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, 
                    args.prompt_postfix, args.audio_dim, device)

    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)