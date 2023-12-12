import os
import sys
import json
import pickle
import random
from utils.functions import calculate_loss

import torch
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import numpy as np
import random
import sys
import torchvision.models as models
from losses.losses import FocalLoss
from args import get_args
from utils.evaluate import compute_confusion_matrix,compute_indexes
from sklearn.metrics import roc_auc_score

# from seresnext import se_resnext50_32x4d
args=get_args()
save_pth = args.res_savedir

if not os.path.exists(save_pth):
    os.makedirs(save_pth)
    print(f'making save dir  {save_pth}-------------')
if args.backbone == 'vit':
    resnet = models.__dict__[args.arch](pretrained=True).to('cuda')
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).to('cuda')
    if os.path.isfile(args.res_weights):
        resnet.load_state_dict(torch.load(args.res_weights))


# resnet = torch.nn.Sequential(*(list(resnet.children())[:-1])).to('cuda')
def pre_cls_model(x):
    features = resnet(x)
    #print('sad',features.size())
    return features.squeeze()


softmax = nn.Softmax(dim=1)

device_gpu = torch.device('cuda')
device_cpu = torch.device('cpu')


def get_lossfn(name,use_reduction=False):
    if name == 'CELoss':
        return nn.CrossEntropyLoss(reduction = args.reduction if use_reduction else 'mean')
    elif name == 'FocalLoss':
        return FocalLoss(gamma=2,alpha=0.25,size_average=False)

# Choosing `num_centers` random data points as the initial centers
def random_init(dataset, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    used = torch.zeros(num_points, dtype=torch.long)
    indices = torch.zeros(num_centers, dtype=torch.long)
    for i in range(num_centers):
        while True:
            cur_id = random.randint(0, num_points - 1)
            if used[cur_id] > 0:
                continue
            used[cur_id] = 1
            indices[i] = cur_id
            break
    indices = indices.to(device_gpu)
    centers = torch.gather(dataset, 0, indices.view(-1, 1).expand(-1, dimension))
    return centers


# Compute for each data point the closest center
def compute_codes(dataset, centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long, device=device_gpu)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes


# Compute new centers as means of the data points forming the clusters
def update_centers(dataset, codes, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float, device=device_gpu)
    cnt = torch.zeros(num_centers, dtype=torch.float, device=device_gpu)
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), dataset)
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float, device=device_gpu))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float, device=device_gpu))
    centers /= cnt.view(-1, 1)
    return centers


def cluster(dataset, num_centers):
    centers = random_init(dataset, num_centers)
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers)
        new_codes = compute_codes(dataset, centers)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        if torch.equal(codes, new_codes):
            break
        codes = new_codes
    return codes



@torch.no_grad()
def evaluate_auc(model, data_loader, device, epoch, multi_tasks, name='valid'):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    resnet.eval()
    #torch.save(resnet.state_dict(), f'{save_pth}/resnet.pth')
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    accu_loss = [0] * multi_tasks
    accu_num = [0] * multi_tasks
    loss_functions = args.loss_fns

    prob_all = [[] for i in range(multi_tasks)]
    label_all = [[] for i in range(multi_tasks)]
    roc_auc = [0] * multi_tasks
    accuracy_total, sensitivity_total, specificity_total, F1_total = [], [], [], []

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        x = images.to(device)
        B = x.shape[0]
        
        if args.backbone != 'vit_res':
            x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
            features = pre_cls_model(x)
            features = torch.reshape(features, (B, int(features.shape[0] / B), features.shape[1]))
        elif args.backbone == 'vit_res':
            features = x

        labels = list(labels.values())
        preds = model(features)

        for i in range(multi_tasks):
            loss_function = get_lossfn(loss_functions[i])
            pred = preds[i]
            label = labels[i]
            label = label.long().to(device)
            loss = loss_function(pred, label)
            pred_classes = torch.max(torch.softmax(pred,dim=1), dim=1)[1]
            accu_num[i] += torch.eq(pred_classes, label.to(device)).sum().item()
            accu_loss[i] += loss.detach().item()
            pred = torch.softmax(pred, dim=1).cpu().numpy()
 
            prob_all[i].extend(pred[:,1])
            label_all[i].extend(label.cpu().numpy())


        s = ''.join([' loss_{}: {:.3f}, acc_{}: {:.3f}'.format(i,acc_l/ (step + 1),i,acc_n/ sample_num) for i,(acc_l,acc_n) in enumerate(zip(accu_loss,accu_num))])

        s_desc = f'[{name} epoch {epoch}] ' + s
        data_loader.desc = s_desc
    # label_all=np.array(label_all)
    # prob_all=np.array(prob_all)
    for i in range(multi_tasks):
        if args.num_classes[i] != 2:
            hot = np.eye(args.num_classes[i])[label_all[i]]
            #print(hot.shape)
            roc_auc[i] = roc_auc_score(hot, np.array(prob_all[i]).reshape(-1,1), multi_class='ovo')
            #roc_auc[i] = roc_auc_score(label_all[i], prob_all[i])
        
        else:roc_auc[i] = roc_auc_score(label_all[i], prob_all[i])

        prob_all[i] = np.array(prob_all[i]).round().astype(int)
        label_all[i] = np.array(label_all[i]).astype(int)
        tp, fp, tn, fn = compute_confusion_matrix(prob_all[i], label_all[i])

       # print("tp,fn,tn,fp:", tp, fp, tn, fn)

        accuracy, sensitivity, specificity, F1 = compute_indexes(tp, fp, tn, fn)

        accuracy_total.append(accuracy)
        sensitivity_total.append(sensitivity)
        specificity_total.append(specificity)
        F1_total.append(F1)

    print("accuracy, sensitivity, specificity, F1,roc_auc:", accuracy_total, sensitivity_total, specificity_total, F1_total, roc_auc)
    return np.array(accu_loss) / (step + 1), np.array(accu_num) / sample_num,roc_auc, F1_total, accuracy_total, sensitivity_total, specificity_total


def train_one_epoch_multi_longtail_weight(model, optimizer, data_loader, device, epoch, multi_tasks,istrain=[True]*args.multi_tasks,cont=False):
    model.train()
    
    # loss_function = torch.nn.BCELoss()#torch.nn.CrossEntropyLoss()
    loss_functions = args.loss_fns
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    accu_num = [0] * multi_tasks
    accu_loss = [0] * multi_tasks
    errors_nums = [0] * multi_tasks
    errors_lists = [ [] for i in range(multi_tasks) ]
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        x = images.to(device)
        B = x.shape[0]

        if args.backbone != 'vit_res':
            resnet.eval()
            x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
            features = pre_cls_model(x)
            features = torch.reshape(features, (B, int(features.shape[0] / B), features.shape[1]))
        elif args.backbone == 'vit_res':
            features = x


        preds = model(features)

        loss_total = 0
    
        labels = list(labels.values())

        for i in range(multi_tasks):
            # print(i,args.loss_fns[i],labels[-1])
            loss_function = get_lossfn(loss_functions[i],use_reduction=True)
            pred = preds[i].to(device)
            label = labels[i].to(device).long()
            # pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

            pred_classes = torch.max(torch.softmax(pred, dim=1), dim=1)[1]
            accu_num[i] += torch.eq(pred_classes, label.to(device)).sum().item()
            if istrain[i]:
                loss,errors_num,error_name = calculate_loss(loss_function,pred,label,args.tasks[i],args.loss_weights[i],args.reduction,labels,cont)
                loss_total += loss
                accu_loss[i] += loss.detach().item()
                errors_nums[i] += errors_num
                errors_lists[i].extend(error_name)

        loss_total.backward()
        

        # s = ''.join([' loss_{}: {:.3f}, acc_{}: {:.3f}'.format(i,acc_l/ (step + 1),i,acc_n/ sample_num) for i,(acc_l,acc_n) in enumerate(zip(accu_loss,accu_num))])
        s = ''.join([' loss_{}: {:.3f}, acc_{}: {:.3f}'.format(i,accu_loss[i]/ (step + 1),i,accu_num[i]/ sample_num) for i in args.show_tasks])
 
        s_desc = f'[train epoch {epoch}] '+ s 
        if cont:
            s_desc += f' errors: {errors_nums}'
        data_loader.desc = s_desc

        if not torch.isfinite(loss_total):
            print('WARNING: non-finite loss, ending training ', loss_total)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    
    if cont:
        err_res = ''.join([f'total_errors{i} : {errors_nums[i]} {len(errors_lists[i])} ' for i in args.show_tasks])
        print(err_res)
        #print(f' total errors :{errors_nums[1]} ,{len(errors_lists[1])}')
    return np.array(accu_loss) / (step + 1), np.array(accu_num) / sample_num, errors_lists

    


@torch.no_grad()
def evaluate_multi_longtail_weight(model, data_loader, device, epoch, multi_tasks,name='valid',cont=False):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    zhenyin = 0
    gjb = 0
    sample_num = 0
    louzhen = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    accu_loss = [0] * multi_tasks
    accu_num = [0] * multi_tasks
    loss_functions = args.loss_fns
    errors_nums = [0] * multi_tasks
    errors_lists = [ [] for i in range(multi_tasks) ]

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        
        x = images.to(device)
        
        B = x.shape[0]
        if args.backbone != 'vit_res':
            resnet.eval()
            torch.save(resnet.state_dict(), f'{save_pth}/resnet.pth')
            x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
            features = pre_cls_model(x)
            features = torch.reshape(features, (B, int(features.shape[0] / B), features.shape[1]))
        elif args.backbone == 'vit_res':
            features = x
 
        labels = list(labels.values())
        preds = model(features)

        for i in range(multi_tasks):
            loss_function = get_lossfn(loss_functions[i])
            pred = preds[i]
            label = labels[i]
            label = label.long().to(device)
            loss = loss_function(pred, label)

            pred_classes = torch.max(pred, dim=1)[1]
            
            if args.tasks[i] == 'level':
                accu_num[i] += (torch.abs(pred_classes-label) <=1).sum().item()  # torch.eq(pred_classes, label.to(device)).sum().item()
                #print('sdsa')
            else:
                accu_num[i] += torch.eq(pred_classes, label.to(device)).sum().item()
                loss,error_num,error_name = calculate_loss(loss_function,pred,label,args.tasks[i],args.loss_weights[i],args.reduction,labels,cont)
                # print(loss,error_num,error_name)
                errors_nums[i] += error_num
                errors_lists[i].extend(error_name)

            accu_loss[i] += loss.detach().item()

        # s = ''.join([' loss_{}: {:.3f}, acc_{}: {:.3f}'.format(i,acc_l/ (step + 1),i,acc_n/ sample_num) for i,(acc_l,acc_n) in enumerate(zip(accu_loss,accu_num))])
        s = ''.join([' loss_{}: {:.3f}, acc_{}: {:.3f}'.format(i,accu_loss[i]/ (step + 1),i,accu_num[i]/ sample_num) for i in args.show_tasks])
        s_desc = f'[{name} epoch {epoch}] '+ s
        if cont :
            s_desc += f' error: {errors_nums}'
        data_loader.desc = s_desc
    if cont:
        err_res = ''.join([f'total_errors{i} : {errors_nums[i]} {len(errors_lists[i])} ' for i in args.show_tasks])
        print(err_res)
        
    return np.array(accu_loss) / (step + 1), np.array(accu_num) / sample_num,  errors_lists